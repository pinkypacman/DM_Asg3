"""Phase-A polish pipeline → submission CSV.

Pipeline (in order):
    (a) OOF construction        — KNN per fold on its own (train → val) split
    (b) KNN k/weights/metric sweep — pick the (k, weights, metric) maximizing OOF macroF1
    (c) Test-time augmentation  — average each fold's encoder's embedding over N augmented forwards
    (d) Threshold/prior correction — learn per-class additive logit biases on OOF, apply to test

Operates on the cache produced by src.tcn + src.embed for a given --tag (e.g. '_la' for
logit-adjusted checkpoints), so v2 (tag '') and v3 (tag '_la') can co-exist.

Safeguards:
    - aborts if c0 + c1 predicted share drops below 80% (threshold tuning over-corrected)
    - asserts OOF rows are aligned to original train order via fold_of indexing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ._progress import tqdm
from .cv import build as build_folds
from .data import CACHE_DIR, build as build_windows
from .tcn import CHECKPOINT_TEMPLATE, DEVICE, HARDataset, TCN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
SAMPLE_PATH = PROJECT_ROOT / "sample_submission.csv"
NUM_CLASSES = 6


# ----------------------------------------------------------------------------- helpers

def _embed_path(fold: int, tag: str) -> Path:
    return CACHE_DIR / f"tcn_embed_fold{fold}{tag}.npz"


def _ckpt_path(fold: int, tag: str) -> Path:
    return CACHE_DIR / CHECKPOINT_TEMPLATE.format(fold=fold, tag=tag)


def _fit_knn(Z_tr, y_tr, k, weights, metric):
    sc = StandardScaler().fit(Z_tr)
    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, n_jobs=-1)
    knn.fit(sc.transform(Z_tr), y_tr)
    return sc, knn


def _proba_to_full(knn, raw):
    """Pad KNN proba to (N, NUM_CLASSES) layout (handles missing class in train)."""
    out = np.zeros((raw.shape[0], NUM_CLASSES), dtype=np.float32)
    for i, c in enumerate(knn.classes_):
        out[:, int(c)] = raw[:, i]
    return out


def _macro_f1(y_true, proba):
    pred = proba.argmax(axis=1)
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def _per_class_f1(y_true, proba):
    pred = proba.argmax(axis=1)
    return f1_score(y_true, pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)


# ----------------------------------------------------------------------------- (a) OOF + (b) KNN sweep

def build_oof(folds, tag_list, k, weights, metric):
    """Per-fold KNN(train → val), averaged across all seed tags; stitched to original order.

    tag_list: list of tags (e.g. ['', '_s7', '_s2024'] or ['_la05', '_la05_s7', ...]).
              Each tag points to a different per-seed model. KNN probabilities are
              averaged across tags for each fold before stitching.
    """
    fold_of = build_folds()["fold_of"]
    y_full = build_windows()["y_train"]
    n_train = len(fold_of)
    oof = np.zeros((n_train, NUM_CLASSES), dtype=np.float32)
    y_aligned = np.zeros(n_train, dtype=np.int64)

    for f in folds:
        val_idx = np.flatnonzero(fold_of == f)
        proba_sum = np.zeros((len(val_idx), NUM_CLASSES), dtype=np.float32)
        for tag in tag_list:
            emb = np.load(_embed_path(f, tag))
            assert np.array_equal(emb["y_val"], y_full[val_idx]), f"y_val misalignment for fold {f}, tag '{tag}'"
            sc, knn = _fit_knn(emb["Z_train"], emb["y_train"], k, weights, metric)
            raw = knn.predict_proba(sc.transform(emb["Z_val"]))
            proba_sum += _proba_to_full(knn, raw)
        oof[val_idx] = proba_sum / len(tag_list)
        y_aligned[val_idx] = y_full[val_idx]

    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-5), "OOF rows don't sum to 1"
    assert np.isfinite(oof).all()
    return oof, y_aligned


def sweep_knn(folds, tag_list, ks, weights_list, metrics):
    n_configs = len(ks) * len(weights_list) * len(metrics)
    n_seeds = len(tag_list)
    print(f"\n--- (b) KNN sweep over k×weights×metric ({n_configs} configs × {n_seeds} seeds) ---")
    best = None
    rows = []
    for k in ks:
        for w in weights_list:
            for m in metrics:
                oof, y = build_oof(folds, tag_list, k, w, m)
                macro = _macro_f1(y, oof)
                rows.append({"k": k, "weights": w, "metric": m, "macroF1": macro})
                if best is None or macro > best["macroF1"]:
                    best = {"k": k, "weights": w, "metric": m, "macroF1": macro,
                            "oof": oof, "y": y}
    df = pd.DataFrame(rows).sort_values("macroF1", ascending=False)
    print(df.head(8).to_string(index=False))
    print(f"  ✔ best: k={best['k']} weights={best['weights']} metric={best['metric']} "
          f"→ OOF macroF1 {best['macroF1']:.4f}")
    return best


# ----------------------------------------------------------------------------- (c) TTA on test

@torch.no_grad()
def _tta_test_embed(fold, tag, X_test, tta_n, jitter_std, max_shift, batch_size, num_workers):
    """Forward X_test through fold's encoder under tta_n random augmentations; return averaged (N, dim)."""
    ckpt_path = _ckpt_path(fold, tag)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = TCN(channels=tuple(cfg["channels"]), dilations=tuple(cfg["dilations"]),
                embed_dim=cfg["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = ckpt["mean"].astype(np.float32)
    std = ckpt["std"].astype(np.float32)

    Z_sum = np.zeros((len(X_test), cfg["embed_dim"]), dtype=np.float64)
    for pass_i in range(tta_n):
        ds = HARDataset(
            X_test, np.zeros(len(X_test), dtype=np.int64), mean, std,
            augment=True, jitter_std=jitter_std, max_shift=max_shift,
            scale_jitter=0.0, rng_seed=1000 + pass_i,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        chunks = []
        for xb, _ in tqdm(loader, desc=f"fold {fold} tta {pass_i+1}/{tta_n}",
                          leave=False, unit="batch"):
            xb = xb.to(DEVICE, non_blocking=True)
            z = model.embed(xb)
            chunks.append(z.detach().cpu().numpy())
        Z_sum += np.concatenate(chunks, axis=0)
    return (Z_sum / tta_n).astype(np.float32)


def predict_test_probs(folds, tag_list, best_knn, X_test, tta_n, jitter_std, max_shift,
                       batch_size=256, num_workers=2):
    """Ensemble test probabilities across (folds × seed tags) using per-fold encoder + TTA.

    For each (fold, tag) pair:
        - KNN database = concat of fold's train + val embeddings (under that tag's encoder)
        - Test side = TTA-averaged embedding (same TTA pattern, same encoder)
        - Predict class probabilities
    All 5×|tag_list| probability arrays are averaged into one (n_test, 6) result.
    """
    k, weights, metric = best_knn["k"], best_knn["weights"], best_knn["metric"]
    proba_per_unit = []
    for f in folds:
        for tag in tag_list:
            emb = np.load(_embed_path(f, tag))
            Z_db = np.concatenate([emb["Z_train"], emb["Z_val"]], axis=0)
            y_db = np.concatenate([emb["y_train"], emb["y_val"]], axis=0)

            if tta_n > 0:
                Z_test_f = _tta_test_embed(
                    f, tag, X_test, tta_n, jitter_std, max_shift,
                    batch_size=batch_size, num_workers=num_workers,
                )
            else:
                Z_test_f = emb["Z_test"]

            sc, knn = _fit_knn(Z_db, y_db, k, weights, metric)
            raw = knn.predict_proba(sc.transform(Z_test_f))
            proba = _proba_to_full(knn, raw)
            proba_per_unit.append(proba)
            shares = (np.bincount(proba.argmax(1), minlength=6) / len(proba) * 100).round(2).tolist()
            print(f"  fold {f}, tag '{tag}': predicted class shares {shares}")
    return np.mean(np.stack(proba_per_unit, axis=0), axis=0)


# ----------------------------------------------------------------------------- (d) threshold tuning

def _objective(b, oof_logp, y):
    shifted = oof_logp + b[None, :]
    pred = shifted.argmax(axis=1)
    return -float(f1_score(y, pred, average="macro", zero_division=0))


def tune_threshold(oof, y, init=None, verbose=True):
    """Powell minimize over 6-d additive logit bias to maximize OOF macroF1."""
    # log-probs (with safety floor) for additive shifts.
    oof_logp = np.log(np.clip(oof, 1e-8, 1.0))
    if init is None:
        init = np.zeros(NUM_CLASSES, dtype=np.float64)
    baseline = -_objective(init, oof_logp, y)
    res = minimize(_objective, x0=init, args=(oof_logp, y),
                   method="Powell", options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 200})
    b_star = res.x
    after = -_objective(b_star, oof_logp, y)
    if verbose:
        print(f"  OOF macroF1 baseline {baseline:.4f} → tuned {after:.4f} (Δ={after - baseline:+.4f})")
        print(f"  biases per class: {[round(float(v), 3) for v in b_star]}")
    return b_star, after


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed-tags", nargs="*", default=[],
                        help="Additional embed-cache tags to ensemble with --tag (one per seed). "
                             "KNN probabilities are averaged across all (fold, tag) pairs.")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--weights", nargs="+", default=["uniform", "distance"])
    parser.add_argument("--metrics", nargs="+", default=["cosine", "euclidean"])
    parser.add_argument("--tta-n", type=int, default=5)
    parser.add_argument("--tta-jitter", type=float, default=0.01)
    parser.add_argument("--tta-shift", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--name", default="", help="Submission filename.")
    args = parser.parse_args()

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    tag_list = [args.tag, *args.seed_tags]
    print(f"  primary tag: '{args.tag}'   seed tags: {args.seed_tags or '(none)'}")
    print(f"  → {len(tag_list)} encoders per fold (×{len(args.folds)} folds = {len(tag_list)*len(args.folds)} models)")

    # ===== (a) baseline OOF with current default KNN (k=15, uniform, cosine) =====
    print("--- (a) baseline OOF (k=15, uniform, cosine) ---")
    base_oof, y_oof = build_oof(args.folds, tag_list, k=15, weights="uniform", metric="cosine")
    base_macro = _macro_f1(y_oof, base_oof)
    base_pc = _per_class_f1(y_oof, base_oof)
    print(f"  baseline OOF macroF1 {base_macro:.4f}  per-class {base_pc.round(3).tolist()}")

    # ===== (b) KNN sweep =====
    best = sweep_knn(args.folds, tag_list, args.ks, args.weights, args.metrics)
    oof_b = best["oof"]
    y_b = best["y"]
    pc_b = _per_class_f1(y_b, oof_b)
    print(f"  after sweep: OOF macroF1 {best['macroF1']:.4f}  per-class {pc_b.round(3).tolist()}")

    # ===== (c) Test-time augmentation, predict test probs =====
    arrays = build_windows()
    print(f"\n--- (c) predicting test with TTA (N={args.tta_n}, σ={args.tta_jitter}, shift=±{args.tta_shift}) ---")
    test_proba = predict_test_probs(
        args.folds, tag_list, best, arrays["X_test"],
        tta_n=args.tta_n, jitter_std=args.tta_jitter, max_shift=args.tta_shift,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    base_pred_share = (np.bincount(test_proba.argmax(1), minlength=6) / len(test_proba) * 100)
    print(f"  test pred share after TTA: {base_pred_share.round(2).tolist()}")

    # ===== (d) threshold tuning on OOF, apply biases to test probs =====
    print("\n--- (d) threshold/prior correction (Powell on OOF) ---")
    b_star, oof_macro_d = tune_threshold(oof_b, y_b)
    oof_pc_d = _per_class_f1(y_b,
        np.exp(np.log(np.clip(oof_b, 1e-8, 1.0)) + b_star[None, :]))
    print(f"  OOF per-class after biases {oof_pc_d.round(3).tolist()}")

    test_logp = np.log(np.clip(test_proba, 1e-8, 1.0))
    test_proba_final = test_logp + b_star[None, :]  # additive shift in log-space
    pred = test_proba_final.argmax(axis=1)
    shares = (np.bincount(pred, minlength=6) / len(pred) * 100)
    print(f"\nfinal test pred class shares: {shares.round(2).tolist()}")

    # ===== (e) safeguards =====
    majority_share = shares[0] + shares[1]
    if majority_share < 80.0:
        # Revert biases — they over-corrected.
        print(f"  ⚠ c0+c1 share dropped to {majority_share:.1f}% (<80%). Reverting threshold biases.")
        b_star = np.zeros_like(b_star)
        pred = test_proba.argmax(axis=1)
        shares = (np.bincount(pred, minlength=6) / len(pred) * 100)
        print(f"  reverted pred shares: {shares.round(2).tolist()}")

    # ===== (e') Save OOF + test (calibrated) probabilities for downstream ensemble =====
    # Normalize the shifted log-probs back to a proper probability distribution
    # so blending against e.g. GBDT softmax outputs is on the same scale.
    def _shifted_softmax(p_in, biases):
        lp = np.log(np.clip(p_in, 1e-8, 1.0)) + biases[None, :]
        lp = lp - lp.max(axis=1, keepdims=True)
        ep = np.exp(lp)
        return (ep / ep.sum(axis=1, keepdims=True)).astype(np.float32)

    oof_calibrated = _shifted_softmax(oof_b, b_star)
    test_calibrated = _shifted_softmax(test_proba, b_star)

    pp_cache = CACHE_DIR / f"postprocess_tcn{args.tag}.npz"
    np.savez_compressed(
        pp_cache,
        oof=oof_calibrated, test=test_calibrated,
        oof_raw=oof_b.astype(np.float32), test_raw=test_proba.astype(np.float32),
        y_oof=y_b,
        biases=b_star.astype(np.float32),
        best_knn=np.array([best["k"], best["weights"], best["metric"]], dtype=object),
        oof_macro_baseline=np.float32(base_macro),
        oof_macro_sweep=np.float32(best["macroF1"]),
        oof_macro_tuned=np.float32(oof_macro_d),
        tta_n=np.int32(args.tta_n),
    )
    print(f"  saved → {pp_cache}  (oof {oof_calibrated.shape}, test {test_calibrated.shape})")

    # ===== (f) write submission =====
    fold_first_emb = np.load(_embed_path(args.folds[0], args.tag))
    file_id_test = fold_first_emb["file_id_test"].astype(int)
    sample = pd.read_csv(SAMPLE_PATH)
    id_to_pred = dict(zip(file_id_test.tolist(), pred.astype(int).tolist()))
    missing = [int(i) for i in sample["Id"] if int(i) not in id_to_pred]
    if missing:
        raise ValueError(f"{len(missing)} Ids missing from prediction; first 5: {missing[:5]}")

    out_name = args.name or f"submission_postprocess{args.tag or '_v3'}.csv"
    out_path = SUBMISSION_DIR / out_name
    sample = sample.copy()
    sample["Label"] = sample["Id"].astype(int).map(id_to_pred).astype(int)
    sample.to_csv(out_path, index=False)

    # ===== (g) summary =====
    print()
    print("=" * 64)
    print(f"OOF macroF1 progression:")
    print(f"  (a) baseline                : {base_macro:.4f}")
    print(f"  (b) KNN sweep best          : {best['macroF1']:.4f}  ({best['k']}, {best['weights']}, {best['metric']})")
    print(f"  (d) +threshold biases       : {oof_macro_d:.4f}  biases={b_star.round(3).tolist()}")
    print(f"submission: {out_path}")
    print(f"  rows : {len(sample)}")
    print(f"  label distribution : {sample['Label'].value_counts().sort_index().to_dict()}")
    print(f"  share (%)          : "
          f"{(sample['Label'].value_counts(normalize=True).sort_index() * 100).round(2).to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
