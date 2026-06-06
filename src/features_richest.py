"""Build the 196-d "richest" feature set = the 172-d `features_richer` set plus
24 orientation-invariant motion descriptors (jerk + RMS) computed from the raw
window. Output: cache/features_richest.npz — consumed by the GBDT/XGB/CatBoost and
all div_* members of the final blend.

Run:  python -m src.features_richest        (after src.features --richer)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .data import CACHE_DIR
from .features import build_richer

NC = 6


def jerk_rms(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """24 motion features from the raw (N,300,6) window (mean channels = first 3)."""
    mv = X[..., :3]                                  # mean vector over time (N,300,3)
    w1 = np.linalg.norm(mv, axis=2)                  # per-step magnitude
    d1 = np.diff(mv, axis=1); w2 = np.linalg.norm(d1, axis=2)        # jerk (1st diff)
    d2 = np.diff(mv, axis=1, n=2); w3 = np.linalg.norm(d2, axis=2)   # 2nd diff
    a, b = mv[:, :-1], mv[:, 1:]                     # angle between successive vectors
    cos = (a * b).sum(2) / (np.linalg.norm(a, 2, 2) * np.linalg.norm(b, 2, 2) + 1e-9)
    w4 = np.arccos(np.clip(cos, -1, 1))
    rms = np.sqrt((X[..., :3] ** 2 + X[..., 3:] ** 2).mean(1))       # RMS per axis (N,3)

    feats, names = [], []
    for arr, nm in [(w1, "w1"), (w2, "w2_jerk"), (w3, "w3"), (w4, "w4_ang")]:
        for op, on in [(arr.mean(1), "mean"), (arr.std(1), "std"), (arr.max(1), "max"),
                       (np.quantile(arr, 0.9, 1), "q90"), (np.quantile(arr, 0.1, 1), "q10")]:
            feats.append(op); names.append(f"{nm}_{on}")
    for i, ax in enumerate("xyz"):
        feats.append(rms[:, i]); names.append(f"rms_{ax}")
    feats.append(np.linalg.norm(rms, axis=1)); names.append("rms_norm")
    return np.nan_to_num(np.stack(feats, 1).astype(np.float32)), names


def build(force: bool = False) -> dict:
    out_path = CACHE_DIR / "features_richest.npz"
    if out_path.exists() and not force:
        with np.load(out_path, allow_pickle=True) as f:
            return {k: f[k] for k in f.files}

    fr = build_richer(force=False)
    F = np.nan_to_num(fr["F_train"].astype(np.float32))
    Fte = np.nan_to_num(fr["F_test"].astype(np.float32))
    w = np.load(CACHE_DIR / "windows.npz", allow_pickle=True)

    new_tr, names = jerk_rms(w["X_train"])
    new_te, _ = jerk_rms(w["X_test"])
    F_train = np.concatenate([F, new_tr], 1)
    F_test = np.concatenate([Fte, new_te], 1)

    np.savez_compressed(
        out_path,
        F_train=F_train, F_test=F_test,
        feature_names=np.array(list(fr["feature_names"]) + names),
        y_train=fr["y_train"], file_id_train=fr["file_id_train"], user_train=fr["user_train"],
        file_id_test=fr["file_id_test"], user_test=fr["user_test"],
    )
    print(f"saved {out_path}  ({F_train.shape[1]} features = 172 richer + {new_tr.shape[1]} jerk/RMS)")
    return {"F_train": F_train, "F_test": F_test}


if __name__ == "__main__":
    build(force="--force" in sys.argv)
    sys.exit(0)
