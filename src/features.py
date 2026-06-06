"""Feature engineering for the HAR task.

Two pre-set feature views:
  - `compute`      → ~34 lean features (cached at cache/features.npz)
                     per-channel summary (mean/std/min/max/median) over the 6 series  -> 30
                     movement intensity from magnitude signals                        ->  3
                     orientation hint (mean_z / mag_mean)                             ->  1
  - `compute_rich` → ~116 features (cached at cache/features_rich.npz) — lean set plus:
                     FFT band powers + spectral entropy/centroid/dominant freq        -> 28
                     temporal dynamics (zero-cross, first-diff, autocorr lags)        -> 24
                     higher-order stats (skew, kurtosis, IQR, q10, q90) per channel   -> 30

All features are fully vectorized in NumPy across the window axis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .data import CACHE_DIR, FEATURE_COLS, build as build_windows

FEATURES_PATH = CACHE_DIR / "features.npz"
FEATURES_RICH_PATH = CACHE_DIR / "features_rich.npz"
FEATURES_RICHER_PATH = CACHE_DIR / "features_richer.npz"

_CH_STATS = ["mean", "std", "min", "max", "median"]


def _per_channel_stats(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """X: (N, 300, 6) -> (N, 30) with names like 'mean_x__mean', 'mean_x__std', ..."""
    feats = np.stack(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            np.median(X, axis=1),
        ],
        axis=-1,
    )  # (N, 6, 5)
    n = feats.shape[0]
    out = feats.reshape(n, -1)  # (N, 30) — order: channel0 stats, channel1 stats, ...
    names = [f"{col}__{stat}" for col in FEATURE_COLS for stat in _CH_STATS]
    return out, names


def _magnitude_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Intensity proxies derived from the 3D magnitude signals."""
    # mean_x, mean_y, mean_z -> resultant per-second mean magnitude.
    mag = np.sqrt(np.sum(X[..., 0:3] ** 2, axis=2))  # (N, 300)
    # std_x, std_y, std_z -> per-second 'jitter' magnitude.
    std_mag = np.sqrt(np.sum(X[..., 3:6] ** 2, axis=2))  # (N, 300)

    feats = np.stack(
        [
            mag.mean(axis=1),
            mag.std(axis=1),
            std_mag.mean(axis=1),
        ],
        axis=-1,
    )  # (N, 3)
    names = ["mag__mean", "mag__std", "std_mag__mean"]
    return feats, names


def _orientation_features(X: np.ndarray, mag_mean: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Gravity-direction proxy: how much of |a| sits on the Z axis (lying vs upright)."""
    mean_z_mean = X[..., 2].mean(axis=1)  # (N,)
    ratio = np.where(mag_mean > 1e-8, mean_z_mean / mag_mean, 0.0)
    return ratio[:, None].astype(np.float32), ["orient__mean_z_over_mag"]


def compute(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """X: (N, 300, 6) -> (features (N, F), feature_names (len F))."""
    parts: list[np.ndarray] = []
    names: list[str] = []

    f, n = _per_channel_stats(X)
    parts.append(f); names.extend(n)

    fm, nm = _magnitude_features(X)
    parts.append(fm); names.extend(nm)

    fo, no = _orientation_features(X, mag_mean=fm[:, 0])
    parts.append(fo); names.extend(no)

    feats = np.concatenate(parts, axis=1).astype(np.float32)
    return feats, names


# ----------------------------------------------------------------------------- rich

_FFT_SIGS = ["mean_x", "mean_y", "mean_z", "mag"]
_FFT_BANDS = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.51)]


def _signal_stack(X: np.ndarray) -> np.ndarray:
    """Build (N, T, 4) of mean_x, mean_y, mean_z, and 3-axis magnitude."""
    mag = np.sqrt(np.sum(X[..., 0:3] ** 2, axis=2))  # (N, T)
    return np.stack([X[..., 0], X[..., 1], X[..., 2], mag], axis=-1)  # (N, T, 4)


def _fft_features(signals: np.ndarray, sample_rate: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """signals: (N, T, K) → (N, 7*K) FFT-derived features per signal."""
    N, T, K = signals.shape
    sig = signals - signals.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(sig, axis=1)
    power = np.abs(fft) ** 2                        # (N, T//2+1, K)
    freqs = np.fft.rfftfreq(T, d=1.0 / sample_rate) # (T//2+1,)

    # Skip DC bin for spectral stats.
    P = power[:, 1:, :]                             # (N, F-1, K)
    f = freqs[1:]                                   # (F-1,)
    total = P.sum(axis=1, keepdims=True) + 1e-12    # (N, 1, K)
    Pn = P / total

    band_p = np.stack(
        [(Pn * ((f >= lo) & (f < hi))[None, :, None]).sum(axis=1)
         for lo, hi in _FFT_BANDS],
        axis=1,
    )  # (N, 4, K)
    entropy = -(Pn * np.log(Pn + 1e-12)).sum(axis=1)
    entropy = entropy / np.log(P.shape[1])          # normalize by log(F-1)
    centroid = (Pn * f[None, :, None]).sum(axis=1)  # (N, K)
    dom_idx = Pn.argmax(axis=1)
    dom_freq = f[dom_idx]                           # (N, K)

    feats = np.concatenate(
        [band_p, entropy[:, None], centroid[:, None], dom_freq[:, None]],
        axis=1,
    ).astype(np.float32)  # (N, 7, K)
    stats_names = ["band0", "band1", "band2", "band3", "spec_entropy", "spec_centroid", "dom_freq"]
    names = [f"{sig}__{stat}" for stat in stats_names for sig in _FFT_SIGS]
    return feats.reshape(N, -1), names


def _temporal_dynamics(signals: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """signals: (N, T, K) → (N, 6*K) zcr/first-diff/autocorr features."""
    N, T, K = signals.shape
    detrended = signals - signals.mean(axis=1, keepdims=True)

    sign_changes = np.diff(np.sign(detrended), axis=1) != 0  # (N, T-1, K)
    zcr = sign_changes.sum(axis=1) / max(T - 1, 1)           # (N, K)

    fd = np.diff(signals, axis=1)                            # (N, T-1, K)
    fd_mean = fd.mean(axis=1)
    fd_std = fd.std(axis=1)

    acfs = []
    for lag in (1, 5, 10):
        a = detrended[:, lag:, :]
        b = detrended[:, : T - lag, :]
        num = (a * b).sum(axis=1)
        denom = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1)) + 1e-12
        acfs.append(num / denom)
    acfs = np.stack(acfs, axis=1)                            # (N, 3, K)

    feats = np.concatenate(
        [zcr[:, None], fd_mean[:, None], fd_std[:, None], acfs],
        axis=1,
    ).astype(np.float32)  # (N, 6, K)
    stats_names = ["zcr", "fd_mean", "fd_std", "acf1", "acf5", "acf10"]
    names = [f"{sig}__{stat}" for stat in stats_names for sig in _FFT_SIGS]
    return feats.reshape(N, -1), names


def _higher_order(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """X: (N, T, C) → (N, 5*C) skew/kurtosis/IQR/q10/q90 per channel."""
    N, T, C = X.shape
    skew = sstats.skew(X, axis=1, bias=False, nan_policy="omit")
    kurt = sstats.kurtosis(X, axis=1, bias=False, nan_policy="omit")
    q10 = np.quantile(X, 0.10, axis=1)
    q25 = np.quantile(X, 0.25, axis=1)
    q75 = np.quantile(X, 0.75, axis=1)
    q90 = np.quantile(X, 0.90, axis=1)
    iqr = q75 - q25

    feats = np.stack([skew, kurt, iqr, q10, q90], axis=1).astype(np.float32)  # (N, 5, C)
    stats_names = ["skew", "kurt", "iqr", "q10", "q90"]
    names = [f"{col}__{stat}" for stat in stats_names for col in FEATURE_COLS]
    return feats.reshape(N, -1), names


def compute_rich(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """X: (N, 300, 6) → (features (N, ~116), feature_names).

    Adds FFT, temporal-dynamics, and higher-order stats to the lean 34-d set.
    """
    parts: list[np.ndarray] = []
    names: list[str] = []

    lean, lean_names = compute(X)
    parts.append(lean); names.extend(lean_names)

    sig4 = _signal_stack(X)

    f_fft, n_fft = _fft_features(sig4)
    parts.append(f_fft); names.extend(n_fft)

    f_td, n_td = _temporal_dynamics(sig4)
    parts.append(f_td); names.extend(n_td)

    f_ho, n_ho = _higher_order(X)
    parts.append(f_ho); names.extend(n_ho)

    feats = np.concatenate(parts, axis=1).astype(np.float32)
    return feats, names


# ----------------------------------------------------------------------------- richer (v9)
# Adds two orthogonal feature groups on top of compute_rich:
#   - cross-channel relationships (18 features) — Missing-1
#   - temporal segmentation       (38 features) — Missing-2
# Total: ~116 + 56 = ~172 features. Targets within-window pattern + coordinated motion
# signals that current aggregate-only features can't capture.

_CROSS_PAIRS = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]  # mean and std pairs
_CROSS_PAIR_NAMES = [
    "mean_x__mean_y", "mean_x__mean_z", "mean_y__mean_z",
    "std_x__std_y", "std_x__std_z", "std_y__std_z",
]
_LAG_PAIRS = [(0, 1), (0, 2), (1, 2)]  # mean pairs only for lagged correlation
_LAG_PAIR_NAMES = ["mean_x__mean_y", "mean_x__mean_z", "mean_y__mean_z"]
_LAGS = (1, 5)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Pearson correlation. a, b: (N, T)."""
    ma = a.mean(axis=1, keepdims=True)
    mb = b.mean(axis=1, keepdims=True)
    cov = ((a - ma) * (b - mb)).mean(axis=1)
    sa = a.std(axis=1)
    sb = b.std(axis=1)
    return cov / (sa * sb + 1e-12)


def _cross_channel_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """X: (N, 300, 6) → (N, 18) cross-channel relationships."""
    parts = []
    names = []

    # 6 cross-channel Pearson correlations.
    for (i, j), nm in zip(_CROSS_PAIRS, _CROSS_PAIR_NAMES):
        parts.append(_pearson_corr(X[:, :, i], X[:, :, j]))
        names.append(f"corr__{nm}")

    # 6 channel ratios (intensity & orientation distribution).
    mag = np.sqrt(np.sum(X[..., 0:3] ** 2, axis=2))  # (N, T)
    mag_mean = mag.mean(axis=1) + 1e-12
    std_x_mean = X[..., 3].mean(axis=1)
    std_y_mean = X[..., 4].mean(axis=1)
    std_z_mean = X[..., 5].mean(axis=1)
    mean_x_mean = X[..., 0].mean(axis=1)
    mean_y_mean = X[..., 1].mean(axis=1)
    parts.append(std_x_mean / (std_y_mean + 1e-12)); names.append("ratio__stdx_over_stdy")
    parts.append(std_x_mean / (std_z_mean + 1e-12)); names.append("ratio__stdx_over_stdz")
    parts.append(std_y_mean / (std_z_mean + 1e-12)); names.append("ratio__stdy_over_stdz")
    parts.append(np.abs(mean_x_mean) / mag_mean); names.append("ratio__abs_mx_over_mag")
    parts.append(np.abs(mean_y_mean) / mag_mean); names.append("ratio__abs_my_over_mag")
    parts.append(std_x_mean / mag_mean); names.append("ratio__stdx_over_mag")

    # 6 lagged cross-correlations (3 mean pairs × 2 lags).
    T = X.shape[1]
    for lag in _LAGS:
        for (i, j), nm in zip(_LAG_PAIRS, _LAG_PAIR_NAMES):
            a = X[:, : T - lag, i]
            b = X[:, lag:, j]
            parts.append(_pearson_corr(a, b))
            names.append(f"lagcorr_lag{lag}__{nm}")

    feats = np.stack(parts, axis=1).astype(np.float32)
    return feats, names


_SEG_N = 5  # 5 segments × 60s for T=300


def _temporal_segmentation_features(X: np.ndarray, n_segments: int = _SEG_N) -> tuple[np.ndarray, list[str]]:
    """X: (N, T, 6) → (N, 38) within-window temporal segmentation features."""
    N, T, _ = X.shape
    seg_size = T // n_segments

    mag = np.sqrt(np.sum(X[..., 0:3] ** 2, axis=2))           # (N, T)
    std_mag = np.sqrt(np.sum(X[..., 3:6] ** 2, axis=2))       # (N, T)
    mean_z = X[..., 2]                                        # (N, T)

    # Compute per-segment stats for each of 5 signals/stats combinations.
    # Each stack has shape (N, n_segments).
    def seg(arr, op):
        out = np.empty((N, n_segments), dtype=np.float32)
        for s in range(n_segments):
            lo = s * seg_size
            hi = (s + 1) * seg_size if s < n_segments - 1 else T
            out[:, s] = op(arr[:, lo:hi])
        return out

    seg_mean_mag    = seg(mag,     lambda a: a.mean(axis=1))
    seg_std_mag     = seg(mag,     lambda a: a.std(axis=1))
    seg_mean_stdmag = seg(std_mag, lambda a: a.mean(axis=1))
    seg_max_stdmag  = seg(std_mag, lambda a: a.max(axis=1))
    seg_mean_meanz  = seg(mean_z,  lambda a: a.mean(axis=1))

    all_seg = [
        ("seg_mean_mag",    seg_mean_mag),
        ("seg_std_mag",     seg_std_mag),
        ("seg_mean_stdmag", seg_mean_stdmag),
        ("seg_max_stdmag",  seg_max_stdmag),
        ("seg_mean_meanz",  seg_mean_meanz),
    ]

    parts, names = [], []
    # 25 per-segment features (5 stats × 5 segments)
    for stat_name, arr in all_seg:
        for s in range(n_segments):
            parts.append(arr[:, s])
            names.append(f"{stat_name}__s{s}")

    # 10 across-segment derived features (5 stats × 2 reductions: range, std)
    for stat_name, arr in all_seg:
        parts.append(arr.max(axis=1) - arr.min(axis=1))
        names.append(f"{stat_name}__range_across_segs")
        parts.append(arr.std(axis=1))
        names.append(f"{stat_name}__std_across_segs")

    # 3 segment-count features (how many segments are above/below window quantiles).
    window_median_mag = np.median(mag, axis=1, keepdims=True)
    window_q25_mag = np.quantile(mag, 0.25, axis=1, keepdims=True)
    window_q75_mag = np.quantile(mag, 0.75, axis=1, keepdims=True)
    parts.append((seg_mean_mag > window_median_mag).sum(axis=1).astype(np.float32))
    names.append("nseg__mag_above_window_median")
    parts.append((seg_mean_mag > window_q75_mag).sum(axis=1).astype(np.float32))
    names.append("nseg__mag_above_window_q75")
    parts.append((seg_mean_mag < window_q25_mag).sum(axis=1).astype(np.float32))
    names.append("nseg__mag_below_window_q25")

    feats = np.stack(parts, axis=1).astype(np.float32)
    return feats, names


def compute_richer(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """X: (N, 300, 6) → (features (N, ~172), feature_names).

    Adds two orthogonal feature groups on top of the 116-d `compute_rich` set:
      - cross-channel relationships (Pearson correlations, ratios, lagged correlations) — 18 dims
      - temporal segmentation       (5 segments × 5 stats + cross-segment derived)      — 38 dims
    Target: capture coordinated motion and within-window pattern that aggregate-only
    features can't see.
    """
    rich, rich_names = compute_rich(X)

    f_cc, n_cc = _cross_channel_features(X)
    f_seg, n_seg = _temporal_segmentation_features(X)

    feats = np.concatenate([rich, f_cc, f_seg], axis=1).astype(np.float32)
    names = list(rich_names) + n_cc + n_seg
    return feats, names


def _build(compute_fn, cache_path: Path, force: bool) -> dict:
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=True) as f:
            return {k: f[k] for k in f.files}

    arrays = build_windows(force=False)
    F_train, names = compute_fn(arrays["X_train"])
    F_test, names_test = compute_fn(arrays["X_test"])
    assert names == names_test
    assert np.isfinite(F_train).all() and np.isfinite(F_test).all(), "non-finite features"

    np.savez_compressed(
        cache_path,
        F_train=F_train,
        F_test=F_test,
        feature_names=np.array(names),
        y_train=arrays["y_train"],
        file_id_train=arrays["file_id_train"],
        user_train=arrays["user_train"],
        file_id_test=arrays["file_id_test"],
        user_test=arrays["user_test"],
    )
    return dict(
        F_train=F_train,
        F_test=F_test,
        feature_names=np.array(names),
        y_train=arrays["y_train"],
        file_id_train=arrays["file_id_train"],
        user_train=arrays["user_train"],
        file_id_test=arrays["file_id_test"],
        user_test=arrays["user_test"],
    )


def build(force: bool = False) -> dict:
    return _build(compute, FEATURES_PATH, force)


def build_rich(force: bool = False) -> dict:
    return _build(compute_rich, FEATURES_RICH_PATH, force)


def build_richer(force: bool = False) -> dict:
    return _build(compute_richer, FEATURES_RICHER_PATH, force)


def _separation_check(F: np.ndarray, y: np.ndarray, names: list[str], top: int = 5) -> None:
    """Print per-class means for the features with the largest class-mean spread."""
    classes = np.unique(y)
    class_means = np.stack([F[y == c].mean(axis=0) for c in classes], axis=0)  # (C, F)
    spread = class_means.max(axis=0) - class_means.min(axis=0)
    # Normalize by overall std so units don't dominate.
    overall_std = F.std(axis=0) + 1e-8
    score = spread / overall_std
    order = np.argsort(-score)[:top]
    print(f"\nTop-{top} features by class-mean spread (normalized):")
    header = "feature".ljust(30) + " | " + " ".join(f"c{c}".rjust(10) for c in classes) + " | score"
    print(header)
    print("-" * len(header))
    for i in order:
        per_class = " ".join(f"{class_means[c_idx, i]:+10.4f}" for c_idx in range(len(classes)))
        print(f"{names[i]:30s} | {per_class} | {score[i]:6.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild features cache.")
    parser.add_argument("--rich", action="store_true",
                        help="Build the ~116-d rich feature set (FFT + temporal + higher-order).")
    parser.add_argument("--richer", action="store_true",
                        help="Build the ~172-d richer feature set (rich + cross-channel + temporal-segmentation).")
    args = parser.parse_args()

    if args.richer:
        arrays = build_richer(force=args.force)
        cache_path = FEATURES_RICHER_PATH
    elif args.rich:
        arrays = build_rich(force=args.force)
        cache_path = FEATURES_RICH_PATH
    else:
        arrays = build(force=args.force)
        cache_path = FEATURES_PATH
    F_train = arrays["F_train"]
    F_test = arrays["F_test"]
    names = list(arrays["feature_names"])
    y = arrays["y_train"]

    print(f"Cache: {cache_path}")
    print(f"  F_train: shape={F_train.shape} dtype={F_train.dtype}")
    print(f"  F_test : shape={F_test.shape} dtype={F_test.dtype}")
    print(f"  feature_names: {len(names)} entries (first 5: {names[:5]})")
    assert F_train.shape[1] == F_test.shape[1] == len(names)
    assert np.isfinite(F_train).all() and np.isfinite(F_test).all(), "non-finite features"

    print("\nPer-feature stats (train):")
    df = pd.DataFrame(F_train, columns=names).describe().T[["mean", "std", "min", "max"]]
    print(df.round(4).to_string())

    _separation_check(F_train, y, names, top=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
