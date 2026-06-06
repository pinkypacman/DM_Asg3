"""GroupKFold(user) fold indices for the modeling phase.

The held-out test set has 40 disjoint users, so we mirror that by holding out
whole users per fold during validation. Indices are cached so downstream
modeling scripts can plug them in without re-importing data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .data import CACHE_DIR, build as build_windows

N_SPLITS = 5
FOLDS_PATH = CACHE_DIR / "folds.npz"


def build(force: bool = False, n_splits: int = N_SPLITS) -> dict:
    if FOLDS_PATH.exists() and not force:
        with np.load(FOLDS_PATH) as f:
            return {k: f[k] for k in f.files}

    arrays = build_windows(force=False)
    y = arrays["y_train"]
    users = arrays["user_train"]
    gkf = GroupKFold(n_splits=n_splits)

    # Fold id per training sample (-1 means unused, but every sample appears in exactly one valid fold).
    fold_of = np.full(len(y), -1, dtype=np.int8)
    for fold_idx, (_train_idx, val_idx) in enumerate(gkf.split(np.zeros_like(y), y, groups=users)):
        fold_of[val_idx] = fold_idx
    assert (fold_of >= 0).all(), "GroupKFold did not assign every sample to a validation fold"

    np.savez_compressed(FOLDS_PATH, fold_of=fold_of, n_splits=np.int8(n_splits))
    return {"fold_of": fold_of, "n_splits": np.int8(n_splits)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    out = build(force=args.force, n_splits=args.n_splits)
    fold_of = out["fold_of"]
    arrays = build_windows(force=False)
    y = arrays["y_train"]
    users = arrays["user_train"]

    print(f"Cache: {FOLDS_PATH}")
    print(f"  fold_of: shape={fold_of.shape} dtype={fold_of.dtype} n_splits={int(out['n_splits'])}")

    # Sanity report: per-fold user counts, sample counts, and class distribution.
    print("\nPer-fold breakdown (validation side):")
    print(f"{'fold':>4} | {'#users':>6} | {'#samples':>8} | class counts (0..5)")
    for f in range(int(out["n_splits"])):
        mask = fold_of == f
        n_u = len(np.unique(users[mask]))
        n_s = int(mask.sum())
        counts = np.bincount(y[mask], minlength=6)
        print(f"{f:>4} | {n_u:>6} | {n_s:>8} | {counts.tolist()}")

    # Sanity: no user appears in more than one fold.
    user_to_folds = {}
    for u, f in zip(users, fold_of):
        user_to_folds.setdefault(int(u), set()).add(int(f))
    multi = {u: fs for u, fs in user_to_folds.items() if len(fs) > 1}
    assert not multi, f"Users appearing in multiple folds: {multi}"
    print("\nOK — each user appears in exactly one validation fold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
