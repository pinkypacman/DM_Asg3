"""Per-fold LightGBM on the rich 116-d feature set.

For each GroupKFold(user) fold k:
    - Train LGBMClassifier on the 4 other folds (~8800 samples)
    - Predict probabilities on the held-out fold k's samples → OOF row
    - Predict probabilities on the full test set
Aggregate:
    OOF (11020, 6)             — stitched across folds, aligned to original train order
    test (6849, 6)             — averaged across the 5 fold-models
Saves to cache/gbdt.npz alongside per-fold val macroF1 / per-class F1 for diagnostics.
"""
from __future__ import annotations

import argparse
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score

from ._progress import tqdm
from .cv import build as build_folds
from .data import CACHE_DIR
from .features import build_rich, build_richer
from .features_richest import build as build_richest

NUM_CLASSES = 6
def _gbdt_cache(tag: str):
    return CACHE_DIR / f"gbdt{tag}.npz"


def train_fold(
    F_all: np.ndarray, y_all: np.ndarray, fold_of: np.ndarray, fold: int,
    F_test: np.ndarray, params: dict, early_stopping: int, seed: int,
) -> dict:
    tr_idx = np.flatnonzero(fold_of != fold)
    va_idx = np.flatnonzero(fold_of == fold)

    model = lgb.LGBMClassifier(
        random_state=seed, n_jobs=-1, verbose=-1, **params,
    )
    model.fit(
        F_all[tr_idx], y_all[tr_idx],
        eval_set=[(F_all[va_idx], y_all[va_idx])],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )

    val_proba = model.predict_proba(F_all[va_idx])
    val_pred = val_proba.argmax(axis=1)
    macro = float(f1_score(y_all[va_idx], val_pred, average="macro", zero_division=0))
    per_class = f1_score(
        y_all[va_idx], val_pred,
        labels=list(range(NUM_CLASSES)), average=None, zero_division=0,
    )

    test_proba = model.predict_proba(F_test)
    return {
        "val_idx": va_idx,
        "val_proba": val_proba.astype(np.float32),
        "test_proba": test_proba.astype(np.float32),
        "macro_f1": macro,
        "per_class_f1": per_class,
        "n_train": int(len(tr_idx)),
        "best_iter": int(model.best_iteration_ or model.n_estimators),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tag", default="", help="Suffix on cache/gbdt{tag}.npz output.")
    parser.add_argument("--features-set", choices=["rich", "richer", "richest"], default="rich",
                        help="Which cached feature set to load (rich=116-d, richer=172-d, richest=196-d).")
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=40)
    parser.add_argument("--feature-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-freq", type=int, default=5)
    parser.add_argument("--early-stopping", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rich = {"rich": build_rich, "richer": build_richer, "richest": build_richest}[args.features_set](force=False)
    print(f"features-set: {args.features_set}  → loaded {rich['F_train'].shape[1]} features")
    F_all = rich["F_train"].astype(np.float32)
    F_test = rich["F_test"].astype(np.float32)
    y_all = rich["y_train"]
    fold_of = build_folds()["fold_of"]
    file_id_test = rich["file_id_test"]
    print(f"rich features: train {F_all.shape}  test {F_test.shape}")
    print(f"folds: {args.folds}  class counts: {np.bincount(y_all, minlength=NUM_CLASSES).tolist()}")

    params = dict(
        objective="multiclass", num_class=NUM_CLASSES,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        class_weight="balanced",
    )
    print(f"params: {params}")

    n_train = len(y_all)
    n_test = len(F_test)
    oof = np.zeros((n_train, NUM_CLASSES), dtype=np.float32)
    test_sum = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
    fold_macros = []
    fold_per_class = []
    fold_best_iters = []

    for f in tqdm(args.folds, desc="GBDT folds", unit="fold"):
        print(f"\n--- fold {f} ---")
        out = train_fold(F_all, y_all, fold_of, f, F_test, params, args.early_stopping, args.seed)
        oof[out["val_idx"]] = out["val_proba"]
        test_sum += out["test_proba"].astype(np.float64)
        fold_macros.append(out["macro_f1"])
        fold_per_class.append(out["per_class_f1"])
        fold_best_iters.append(out["best_iter"])
        pc = " ".join(f"{v:.3f}" for v in out["per_class_f1"])
        print(f"  best_iter={out['best_iter']:4d}  val macroF1={out['macro_f1']:.4f}  per-class [{pc}]")

    test_avg = (test_sum / len(args.folds)).astype(np.float32)

    # Stitched OOF macroF1 (using only the folds we actually trained).
    seen = np.isin(fold_of, args.folds)
    pred = oof.argmax(axis=1)
    oof_macro = float(f1_score(y_all[seen], pred[seen], average="macro", zero_division=0))
    oof_per_class = f1_score(
        y_all[seen], pred[seen],
        labels=list(range(NUM_CLASSES)), average=None, zero_division=0,
    )

    print()
    print("=" * 72)
    print(f"5-fold OOF macroF1            : {oof_macro:.4f}")
    print(f"OOF per-class F1              : {[round(float(v), 3) for v in oof_per_class]}")
    print(f"Per-fold val macroF1 spread   : "
          f"min {min(fold_macros):.4f}  mean {np.mean(fold_macros):.4f}  max {max(fold_macros):.4f}")
    print(f"OOF pred class counts         : {np.bincount(pred[seen], minlength=NUM_CLASSES).tolist()}")
    print(f"Test pred (mean-prob argmax)  : {np.bincount(test_avg.argmax(1), minlength=NUM_CLASSES).tolist()}")

    out_path = _gbdt_cache(args.tag)
    np.savez_compressed(
        out_path,
        oof=oof, test=test_avg,
        y_oof=y_all,
        file_id_test=file_id_test,
        fold_macros=np.array(fold_macros, dtype=np.float32),
        fold_per_class=np.stack(fold_per_class).astype(np.float32),
        fold_best_iters=np.array(fold_best_iters, dtype=np.int32),
        oof_macro=np.float32(oof_macro),
        oof_per_class=oof_per_class.astype(np.float32),
    )
    print(f"\nsaved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
