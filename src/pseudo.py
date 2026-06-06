"""Generate pseudo-labels for the test set from a high-quality source blend.

Goal: extract the *confident-enough* test predictions and treat them as
additional labeled training data for a re-trained XGB ("v8" attempt).

Why per-class thresholds:
  - c0 / c1 (majority, easy) — moderate threshold; many confident predictions.
  - c3 / c5 (moderate, larger sample base)                — moderate threshold.
  - c2 / c4 (rare; high backfire risk)                    — strict threshold so
        we only inject rows the model is very sure about. Adding wrong c2 / c4
        rows would lock in past biases (we saw v6a over-predict c2 to 3.72 %).

Source blend (default): v5's recipe — 0.40 · TCN_raw + 0.60 · LGBM — applied to
the cached test probabilities, with NO threshold biases. v5 had the best
public-LB so its uncalibrated blend is our most-validated probability source.

Saves cache/pseudo{TAG}.npz with:
    indices      — (n_pseudo,) int      — test row indices that passed threshold
    labels       — (n_pseudo,) int      — argmax class for those rows
    probs        — (n_pseudo,) float    — max probability for those rows
    blend_w      — float array          — weights used to construct source probs
    thresholds   — float array          — per-class threshold used
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .data import CACHE_DIR
from .features import build_rich

NUM_CLASSES = 6


def _load_source(tcn_tag: str = "_la05") -> dict:
    """Load TCN, LGBM, XGB test probabilities from caches. Each is (n_test, NUM_CLASSES)."""
    tcn = np.load(CACHE_DIR / f"postprocess_tcn{tcn_tag}.npz")
    gbdt = np.load(CACHE_DIR / "gbdt.npz")
    xgb_cache_p = CACHE_DIR / "xgb.npz"
    xgb = np.load(xgb_cache_p) if xgb_cache_p.exists() else None
    return {
        "tcn_test": tcn["test_raw"].astype(np.float32),
        "lgbm_test": gbdt["test"].astype(np.float32),
        "xgb_test": xgb["test"].astype(np.float32) if xgb is not None else None,
        "file_id_test": gbdt["file_id_test"].astype(int),
    }


def make_blend(src: dict, w_tcn: float, w_lgbm: float, w_xgb: float) -> np.ndarray:
    """Return weighted blend of source probabilities; weights are normalized."""
    parts, weights = [], []
    if w_tcn > 0:
        parts.append(src["tcn_test"]); weights.append(w_tcn)
    if w_lgbm > 0:
        parts.append(src["lgbm_test"]); weights.append(w_lgbm)
    if w_xgb > 0:
        if src["xgb_test"] is None:
            raise ValueError("w_xgb > 0 but cache/xgb.npz not found")
        parts.append(src["xgb_test"]); weights.append(w_xgb)
    weights = np.array(weights) / np.sum(weights)
    return sum(w * p for w, p in zip(weights, parts))


def select(probs: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (indices, labels, max_probs) for rows passing per-class threshold.

    A row passes if `max(probs[i]) >= thresholds[argmax(probs[i])]`.
    """
    preds = probs.argmax(axis=1)
    max_p = probs.max(axis=1)
    per_row_thresh = thresholds[preds]
    mask = max_p >= per_row_thresh
    return np.flatnonzero(mask), preds[mask], max_p[mask]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="Suffix on cache/pseudo{TAG}.npz output.")
    parser.add_argument("--tcn-tag", default="_la05", help="Postprocess cache tag to use.")
    # Blend weights (v5 recipe by default: 0.40 TCN + 0.60 LGBM, no XGB).
    parser.add_argument("--w-tcn", type=float, default=0.40)
    parser.add_argument("--w-lgbm", type=float, default=0.60)
    parser.add_argument("--w-xgb", type=float, default=0.00)
    # Per-class thresholds — rare classes stricter to prevent backfire.
    parser.add_argument("--thresh-c0", type=float, default=0.80)
    parser.add_argument("--thresh-c1", type=float, default=0.80)
    parser.add_argument("--thresh-c2", type=float, default=0.95)   # strict
    parser.add_argument("--thresh-c3", type=float, default=0.85)
    parser.add_argument("--thresh-c4", type=float, default=0.90)   # strict
    parser.add_argument("--thresh-c5", type=float, default=0.85)
    args = parser.parse_args()

    print(f"Loading source probabilities (tcn-tag '{args.tcn_tag}')...")
    src = _load_source(args.tcn_tag)
    print(f"  TCN  test : {src['tcn_test'].shape}")
    print(f"  LGBM test : {src['lgbm_test'].shape}")
    print(f"  XGB  test : {'present' if src['xgb_test'] is not None else '(missing — w_xgb must be 0)'}")

    probs = make_blend(src, args.w_tcn, args.w_lgbm, args.w_xgb)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), "blended probs don't sum to 1"

    thresholds = np.array(
        [args.thresh_c0, args.thresh_c1, args.thresh_c2,
         args.thresh_c3, args.thresh_c4, args.thresh_c5],
        dtype=np.float32,
    )
    print(f"\nBlend weights : TCN={args.w_tcn}  LGBM={args.w_lgbm}  XGB={args.w_xgb}")
    print(f"Thresholds    : {thresholds.tolist()}")

    indices, labels, max_probs = select(probs, thresholds)
    n_pseudo = len(indices)
    n_test = len(probs)
    print(f"\nSelected {n_pseudo}/{n_test} ({n_pseudo/n_test*100:.1f}%) test rows as pseudo-labels.")

    # Per-class breakdown.
    print("\nPer-class breakdown:")
    print(f"{'class':>5} | {'threshold':>9} | {'argmax-c':>10} | {'kept':>5} | {'avg conf':>8} | {'kept %':>7}")
    print("-" * 65)
    all_preds = probs.argmax(axis=1)
    for c in range(NUM_CLASSES):
        total = int((all_preds == c).sum())
        kept = int((labels == c).sum())
        if kept == 0:
            avg_conf = float("nan")
        else:
            avg_conf = max_probs[labels == c].mean()
        pct = kept / total * 100 if total else 0.0
        print(f"   c{c} | {thresholds[c]:>9.2f} | {total:>10d} | {kept:>5d} | {avg_conf:>8.3f} | {pct:>6.1f}%")

    # Sanity guard for c2 backfire.
    c2_kept = int((labels == 2).sum())
    c4_kept = int((labels == 4).sum())
    if c2_kept > 200:
        print(f"\n  ⚠ {c2_kept} c2 pseudo-labels — relatively many for a rare class. Verify thresholds.")
    if c4_kept > 100:
        print(f"\n  ⚠ {c4_kept} c4 pseudo-labels — relatively many for the rarest class. Verify thresholds.")

    # Also report what rich features we'll add (for sanity).
    rich = build_rich(force=False)
    X_pseudo = rich["F_test"][indices].astype(np.float32)
    print(f"\nFeature rows that will be added: X_pseudo shape = {X_pseudo.shape}")

    out_path = CACHE_DIR / f"pseudo{args.tag}.npz"
    np.savez_compressed(
        out_path,
        indices=indices.astype(np.int64),
        labels=labels.astype(np.int64),
        max_probs=max_probs.astype(np.float32),
        blend_w=np.array([args.w_tcn, args.w_lgbm, args.w_xgb], dtype=np.float32),
        thresholds=thresholds,
        n_pseudo=np.int32(n_pseudo),
        n_test=np.int32(n_test),
    )
    print(f"\nsaved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
