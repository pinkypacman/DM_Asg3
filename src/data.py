"""Load HAR accelerometer windows from CSV folders into cached NumPy arrays.

Each CSV is one 5-minute window (300 rows × 6 statistical features).
We collapse each window to shape (300, 6) in column order:
    mean_x, mean_y, mean_z, std_x, std_y, std_z

Outputs cached to cache/windows.npz:
    X_train (N_train, 300, 6) float32
    y_train (N_train,)        int64
    file_id_train (N_train,)  int64
    user_train    (N_train,)  int16
    X_test  (N_test, 300, 6)  float32
    file_id_test  (N_test,)   int64
    user_test     (N_test,)   int16
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ._progress import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = PROJECT_ROOT / "train" / "train"
TEST_ROOT = PROJECT_ROOT / "test" / "test"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_PATH = CACHE_DIR / "windows.npz"

FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
WINDOW_LEN = 300
USER_RE = re.compile(r"User_(\d+)")


def _user_id(folder_name: str) -> int:
    m = USER_RE.match(folder_name)
    if not m:
        raise ValueError(f"Cannot parse user id from folder name {folder_name!r}")
    return int(m.group(1))


def _list_csvs(root: Path) -> list[tuple[int, Path]]:
    """Return [(user_id, csv_path), ...] sorted by (user_id, file name)."""
    out: list[tuple[int, Path]] = []
    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        uid = _user_id(user_dir.name)
        for csv in sorted(user_dir.glob("*.csv")):
            out.append((uid, csv))
    return out


def _load_split(root: Path, with_label: bool, desc: str):
    entries = _list_csvs(root)
    n = len(entries)
    X = np.empty((n, WINDOW_LEN, len(FEATURE_COLS)), dtype=np.float32)
    file_id = np.empty(n, dtype=np.int64)
    user = np.empty(n, dtype=np.int16)
    y = np.empty(n, dtype=np.int64) if with_label else None

    for i, (uid, csv_path) in enumerate(tqdm(entries, desc=desc, unit="file")):
        df = pd.read_csv(csv_path)
        if len(df) != WINDOW_LEN:
            raise ValueError(f"{csv_path}: expected {WINDOW_LEN} rows, got {len(df)}")
        X[i] = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
        # `file_id` is constant within a file; take the first row's value.
        file_id[i] = int(df["file_id"].iloc[0])
        user[i] = uid
        if with_label:
            lbl = df["label"].iloc[0]
            # Sanity: label is constant within the file.
            if (df["label"] != lbl).any():
                raise ValueError(f"{csv_path}: label varies within file")
            y[i] = int(lbl)

    # Light cleaning: clamp tiny negative std values (shouldn't exist) to 0.
    std_block = X[..., 3:]
    if (std_block < 0).any():
        np.clip(std_block, 0.0, None, out=std_block)
    if not np.isfinite(X).all():
        raise ValueError(f"{desc}: non-finite values in X")

    return X, y, file_id, user


def build(force: bool = False) -> dict[str, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists() and not force:
        with np.load(CACHE_PATH) as f:
            return {k: f[k] for k in f.files}

    X_train, y_train, fid_train, u_train = _load_split(TRAIN_ROOT, with_label=True, desc="train")
    X_test, _, fid_test, u_test = _load_split(TEST_ROOT, with_label=False, desc="test")

    np.savez_compressed(
        CACHE_PATH,
        X_train=X_train,
        y_train=y_train,
        file_id_train=fid_train,
        user_train=u_train,
        X_test=X_test,
        file_id_test=fid_test,
        user_test=u_test,
    )
    return dict(
        X_train=X_train,
        y_train=y_train,
        file_id_train=fid_train,
        user_train=u_train,
        X_test=X_test,
        file_id_test=fid_test,
        user_test=u_test,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild cache even if it exists.")
    args = parser.parse_args()

    arrays = build(force=args.force)
    print(f"Cache: {CACHE_PATH}")
    for k, v in arrays.items():
        print(f"  {k:18s} shape={tuple(v.shape)} dtype={v.dtype}")

    assert arrays["X_train"].shape == (11020, 300, 6), arrays["X_train"].shape
    assert arrays["X_test"].shape == (6849, 300, 6), arrays["X_test"].shape
    assert np.isfinite(arrays["X_train"]).all()
    assert np.isfinite(arrays["X_test"]).all()
    assert arrays["y_train"].min() >= 0 and arrays["y_train"].max() <= 5

    sample = pd.read_csv(PROJECT_ROOT / "sample_submission.csv")
    sample_ids = set(sample["Id"].astype(int).tolist())
    test_ids = set(arrays["file_id_test"].astype(int).tolist())
    assert sample_ids == test_ids, (
        f"submission Ids ({len(sample_ids)}) != test file_ids ({len(test_ids)}); "
        f"missing from test: {sorted(sample_ids - test_ids)[:5]}..."
    )
    print(f"OK — file_id_test matches sample_submission.csv (n={len(test_ids)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
