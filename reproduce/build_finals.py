"""Reproduce the two FINAL submissions from cached model outputs.

    submission_tcn_gru.csv          (public-LB bet, public 0.8160)
    submission_tcn_robust_v5dist.csv (robust/private bet, public 0.8115)

Both share ONE underlying "gru-blend" probability model (a 12-member diverse
ensemble); they differ only in the final DECISION LAYER applied to the blended
test probabilities:

    gru          : per-class bias calibration that pins the class-2 predicted
                   share to 3.3% and class-5 to 3.6% (the public-LB optima),
                   starting from v5's frozen 6-dim decision biases B0.
    robust_v5dist: prior-match the predictions to v5's full class distribution
                   [41.1, 43.5, 3.6, 7.4, 1.0, 3.4]% (the generalizable / private
                   target), via additive log-prob biases.

Everything is computed from precomputed per-model OOF/test probability caches in
cache/ (the expensive GPU/boosting model fits). No model training happens here.

Run:
    python reproduce/build_finals.py            # writes into submission/
    python reproduce/build_finals.py --out DIR  # writes elsewhere (for verification)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
SAMPLE = ROOT / "sample_submission.csv"
NC = 6

# v5's frozen 6-dim decision biases (the 0.8008 champion's calibration).
B0 = np.array([0.332, 0.06, 0.991, 0.289, -0.077, -0.066])

# --- the 12-member "gru-blend" ensemble: name -> (cache file, test-prob key) ---
MEMBERS = {
    "knn":         ("postprocess_tcn_la05.npz", "test_raw"),  # TCN-encoder + KNN-on-embedding
    "lgb":         ("gbdt_richest.npz",         "test"),      # LightGBM on richest features
    "xgb":         ("xgb_richest.npz",          "test"),      # XGBoost on richest features
    "cat":         ("_catboost_oof.npz",        "test"),      # CatBoost
    "bagging":     ("div_bagging.npz",          "test"),      # bagged trees (diverse family)
    "mlp":         ("div_mlp.npz",              "test"),      # feature-MLP
    "linear":      ("div_linear.npz",           "test"),      # linear / Nystroem-RBF
    "inception":   ("div_inception.npz",        "test"),      # InceptionTime multiscale conv
    "tabular":     ("div_tabular.npz",          "test"),      # TabNet
    "resnet":      ("div_resnet.npz",           "test"),      # 1D-ResNet on raw sequence
    "transformer": ("div_transformer.npz",      "test"),      # Transformer encoder on sequence
    "bigru":       ("bigru.npz",                "test"),      # BiGRU on raw sequence
}

# --- blend weights (frozen from v21/v23/gru tuning) ---
D = 0.45                                              # total weight on the 5 "diverse" families
CORE = {"knn": .30, "lgb": .23, "xgb": .24, "cat": .23}
DIVERSE = ["bagging", "mlp", "linear", "inception", "tabular"]
DEEP = {"resnet": .10, "transformer": .07, "bigru": .08}  # deep-sequence slot (incl. BiGRU)

V5_DIST = np.array([41.1, 43.5, 3.6, 7.4, 1.0, 3.4])     # v5's proven, generalizable distribution


def _load_test(fn: str, key: str) -> np.ndarray:
    p = np.load(CACHE / fn, allow_pickle=True)[key].astype(float)
    return p / p.sum(1, keepdims=True)


def gru_blend_probs() -> np.ndarray:
    """The shared 12-member diverse ensemble, as normalized test probabilities."""
    P = {n: _load_test(fn, k) for n, (fn, k) in MEMBERS.items()}
    # v21 base weights: core scaled by (1-D), each diverse family gets D/5.
    w = {k: v * (1 - D) for k, v in CORE.items()}
    for n in DIVERSE:
        w[n] = D / 5
    # graft the deep slot on top: scale the v21 base by (1 - sum(DEEP)), add DEEP.
    s = 1 - sum(DEEP.values())
    blend = sum(w[n] * s * P[n] for n in w) + sum(DEEP[n] * P[n] for n in DEEP)
    return blend / blend.sum(1, keepdims=True)


def _shifted(p: np.ndarray, b: np.ndarray) -> np.ndarray:
    lp = np.log(np.clip(p, 1e-8, 1)) + b
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


def _solve_share(prob, b, cls, target):
    """Binary-search the class-`cls` additive bias so its predicted share == target."""
    lo, hi = -3.0, 4.0
    for _ in range(40):
        m = (lo + hi) / 2
        bb = b.copy()
        bb[cls] = m
        if (_shifted(prob, bb).argmax(1) == cls).mean() < target:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def decide_gru(prob: np.ndarray) -> np.ndarray:
    """gru decision layer: pin c2 share to 3.3% and c5 to 3.6%, from v5's B0 biases."""
    b = B0.copy()
    for _ in range(15):
        b[2] = _solve_share(prob, b, 2, 0.033)
        b[5] = _solve_share(prob, b, 5, 0.036)
    return _shifted(prob, b).argmax(1)


def decide_prior_match(prob: np.ndarray, target_pct: np.ndarray) -> np.ndarray:
    """robust_v5dist decision layer: additive log-prob biases that drive the predicted
    class distribution to `target_pct`."""
    target = target_pct / target_pct.sum()
    b = np.zeros(NC)
    for _ in range(500):
        pred = (np.log(np.clip(prob, 1e-8, 1)) + b).argmax(1)
        cur = np.bincount(pred, minlength=NC) / len(pred)
        b += 0.5 * (np.log(target + 1e-6) - np.log(cur + 1e-6))
    return (np.log(np.clip(prob, 1e-8, 1)) + b).argmax(1)


def write_submission(pred: np.ndarray, fid: np.ndarray, out: Path) -> None:
    sample = pd.read_csv(SAMPLE)
    mapping = dict(zip(fid.tolist(), pred.tolist()))
    missing = [int(i) for i in sample["Id"] if int(i) not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} Ids missing from prediction (first 5): {missing[:5]}")
    sample = sample.copy()
    sample["Label"] = sample["Id"].astype(int).map(mapping).astype(int)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    sh = (np.bincount(pred, minlength=NC) / len(pred) * 100).round(1).tolist()
    print(f"  wrote {out.name:36s} shares={sh}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "submission"),
                    help="output directory for the two submission CSVs")
    args = ap.parse_args()
    out_dir = Path(args.out)

    # test-window ids (same order as every member cache); read from a small cache so
    # reproduction does not require the bulky windows.npz (raw arrays).
    fid = np.load(CACHE / "gbdt_richest.npz", allow_pickle=True)["file_id_test"].astype(int)
    blend = gru_blend_probs()

    print("Reproducing final submissions from cache/ ...")
    write_submission(decide_gru(blend), fid, out_dir / "submission_tcn_gru.csv")
    write_submission(decide_prior_match(blend, V5_DIST), fid,
                     out_dir / "submission_tcn_robust_v5dist.csv")


if __name__ == "__main__":
    main()
