"""Per-fold XGBoost on the rich 116-d feature set (c2-leaning, diversified vs LightGBM).

Diversification ingredients vs the default `src/gbdt.py`:
  - Different framework math (split scoring, L2 regularization, histogram method)
  - Lower colsample_bytree (0.6 vs LGBM 0.85)  — strongest column-diversity knob
  - Slightly slower learning rate (0.03 vs LGBM 0.05) for lower per-tree variance

C2-leaning ingredients (deliberately gentle to avoid memorization):
  - balanced sample weights × 1.5 multiplier on c2 only
  - matched max_depth=6 to LightGBM (NOT deeper) — see depth/leaf analysis: deeper risks
    memorization of the ~280 c2 train samples per fold

Effective per-sample gradient weights (full-train counts; per-fold values are within ~1% of these):
    formula: balanced[c] = N_total / (n_classes × count_c),  then c2 ×= 1.5

    class   count   balanced   ×c2-boost   note
    c0      4643    0.396      0.396       majority
    c1      4695    0.391      0.391       majority
    c2       358    5.131      7.697  ←    *1.5× emphasis* (vs c4 12.94, still well below)
    c3       656    2.799      2.799       moderate
    c4       142   12.939     12.939       rarest class — already heavily up-weighted by 'balanced'
    c5       526    3.491      3.491       moderate

    ratio interpretation: each c2 sample contributes ~19× the gradient/hessian of a c0 sample
    to the split-finding objective (7.697 / 0.396 ≈ 19), pushing the tree to carve out
    c2-discriminating splits without crossing into c4's natural emphasis territory.

Saves to cache/xgb{tag}.npz alongside per-fold val macroF1 / per-class F1 for diagnostics.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from ._progress import tqdm
from .cv import build as build_folds
from .data import CACHE_DIR
from .features import build_rich, build_richer
from .features_richest import build as build_richest

NUM_CLASSES = 6


def _xgb_cache(tag: str):
    return CACHE_DIR / f"xgb{tag}.npz"


def _class_weights_from(y: np.ndarray, c2_multiplier: float) -> np.ndarray:
    """Compute balanced class-weight vector (length NUM_CLASSES) from `y` and apply c2 boost.

    Requires all NUM_CLASSES classes to be present in `y`. Use this on the REAL training fold,
    then look up `class_w[y_pseudo]` separately for pseudo rows — never recompute from `y_pseudo`
    (which may be missing rare classes due to strict thresholds).
    """
    classes = np.arange(NUM_CLASSES)
    class_w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_w[2] *= c2_multiplier
    return class_w


def _build_sample_weights(y: np.ndarray, c2_multiplier: float) -> np.ndarray:
    """sklearn 'balanced' weights, then multiply c2 (and only c2) by c2_multiplier."""
    return _class_weights_from(y, c2_multiplier)[y].astype(np.float64)


def train_fold(
    F_all: np.ndarray, y_all: np.ndarray, fold_of: np.ndarray, fold: int,
    F_test: np.ndarray, params: dict, c2_multiplier: float,
    early_stopping: int, seed: int,
    F_pseudo: np.ndarray | None = None, y_pseudo: np.ndarray | None = None,
    pseudo_weight_scale: float = 1.0,
) -> dict:
    tr_idx = np.flatnonzero(fold_of != fold)
    va_idx = np.flatnonzero(fold_of == fold)

    # Real training samples — compute class weights here (all 6 classes present).
    X_tr = F_all[tr_idx]
    y_tr = y_all[tr_idx]
    class_w = _class_weights_from(y_tr, c2_multiplier)
    sample_w = class_w[y_tr].astype(np.float64)

    # Append pseudo-labeled test samples (if provided).
    # Reuse the SAME class_w vector (don't recompute from y_pseudo — pseudo may be missing
    # rare classes due to per-class strict thresholds, which would crash compute_class_weight).
    if F_pseudo is not None and y_pseudo is not None and len(F_pseudo) > 0:
        X_tr = np.concatenate([X_tr, F_pseudo], axis=0)
        sw_pseudo = class_w[y_pseudo].astype(np.float64) * pseudo_weight_scale
        y_tr = np.concatenate([y_tr, y_pseudo], axis=0)
        sample_w = np.concatenate([sample_w, sw_pseudo], axis=0)

    model = xgb.XGBClassifier(
        random_state=seed, n_jobs=-1, **params,
        eval_metric="mlogloss", early_stopping_rounds=early_stopping,
        verbosity=0,
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_w,
        eval_set=[(F_all[va_idx], y_all[va_idx])],
        verbose=False,
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
        "best_iter": int(model.best_iteration if model.best_iteration is not None else model.n_estimators),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tag", default="", help="Suffix on cache/xgb{tag}.npz output.")
    parser.add_argument("--n-estimators", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.6)
    parser.add_argument("--colsample-bylevel", type=float, default=0.7)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--c2-multiplier", type=float, default=1.5,
                        help="Multiplier on c2's balanced sample weight (1.0 = no extra boost).")
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-pseudo", action="store_true",
                        help="Load cache/pseudo{PSEUDO_TAG}.npz and concat pseudo-labeled test rows to training.")
    parser.add_argument("--pseudo-tag", default="",
                        help="Suffix on cache/pseudo{TAG}.npz when --use-pseudo is set.")
    parser.add_argument("--pseudo-weight-scale", type=float, default=1.0,
                        help="Scale factor on pseudo-row sample weights (1.0 = same as real, <1.0 = down-weight).")
    parser.add_argument("--features-set", choices=["rich", "richer", "richest"], default="rich",
                        help="Which cached feature set to load (rich=116-d, richer=172-d, richest=196-d).")
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
        objective="multi:softprob", num_class=NUM_CLASSES,
        tree_method="hist",
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        colsample_bylevel=args.colsample_bylevel,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
    )
    print(f"params: {params}")

    # Show effective per-class weights (after balanced + c2 boost) for sanity.
    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_all)
    cw_boost = cw.copy(); cw_boost[2] *= args.c2_multiplier
    print(f"per-class weights (balanced)            : {[round(float(v),3) for v in cw]}")
    print(f"per-class weights (balanced × c2×{args.c2_multiplier}): {[round(float(v),3) for v in cw_boost]}")

    # Load pseudo-labels if requested.
    F_pseudo, y_pseudo = None, None
    if args.use_pseudo:
        pseudo_path = CACHE_DIR / f"pseudo{args.pseudo_tag}.npz"
        if not pseudo_path.exists():
            raise FileNotFoundError(
                f"--use-pseudo set but {pseudo_path} missing. "
                f"Run: python -m src.pseudo --tag '{args.pseudo_tag}'"
            )
        pseudo = np.load(pseudo_path)
        idx = pseudo["indices"]
        F_pseudo = F_test[idx].astype(np.float32)
        y_pseudo = pseudo["labels"].astype(y_all.dtype)
        print(f"\nPseudo-labels loaded from {pseudo_path.name}:")
        print(f"  n_pseudo: {len(F_pseudo)} ({len(F_pseudo)/len(F_test)*100:.1f}% of test)")
        print(f"  class counts: {np.bincount(y_pseudo, minlength=NUM_CLASSES).tolist()}")
        print(f"  thresholds:   {pseudo['thresholds'].tolist()}")
        print(f"  weight scale: {args.pseudo_weight_scale}")

    n_train = len(y_all)
    n_test = len(F_test)
    oof = np.zeros((n_train, NUM_CLASSES), dtype=np.float32)
    test_sum = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
    fold_macros = []
    fold_per_class = []
    fold_best_iters = []

    for f in tqdm(args.folds, desc="XGB folds", unit="fold"):
        print(f"\n--- fold {f} ---", flush=True)
        out = train_fold(
            F_all, y_all, fold_of, f, F_test, params,
            c2_multiplier=args.c2_multiplier,
            early_stopping=args.early_stopping, seed=args.seed,
            F_pseudo=F_pseudo, y_pseudo=y_pseudo,
            pseudo_weight_scale=args.pseudo_weight_scale,
        )
        oof[out["val_idx"]] = out["val_proba"]
        test_sum += out["test_proba"].astype(np.float64)
        fold_macros.append(out["macro_f1"])
        fold_per_class.append(out["per_class_f1"])
        fold_best_iters.append(out["best_iter"])
        pc = " ".join(f"{v:.3f}" for v in out["per_class_f1"])
        print(f"  best_iter={out['best_iter']:4d}  val macroF1={out['macro_f1']:.4f}  per-class [{pc}]", flush=True)

    test_avg = (test_sum / len(args.folds)).astype(np.float32)

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

    out_path = _xgb_cache(args.tag)
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
        c2_multiplier=np.float32(args.c2_multiplier),
    )
    print(f"\nsaved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
