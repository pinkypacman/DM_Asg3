"""Deterministic reproduction of the two FINAL submissions from cached model predictions.

Given the cached per-model TEST probability matrices (cache/*.npz), this regenerates
the exact submission CSVs via a fully deterministic blend + calibration (no randomness):
  - submission_tcn_gru.csv         : gru-blend model + (c2=3.3%, c5=3.6%) bias calibration
  - submission_tcn_robust_v5dist.csv : gru-blend model + match-to-v5-distribution calibration

The blend weights, v5 frozen biases, and calibration iterations are fixed constants, so
re-running this produces byte-identical CSVs. (The upstream cache/*.npz are produced by the
per-model training scripts; GBDTs are seed-deterministic, deep models are seed-fixed but
subject to GPU non-determinism — hence the *cached predictions* are the reproducible artifact.)

Run:  python -m src.reproduce_submissions [--verify]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "cache"; SUB = ROOT / "submission"; SAMPLE = ROOT / "sample_submission.csv"
NC = 6
B0 = np.array([0.332, 0.06, 0.991, 0.289, -0.077, -0.066])   # v5 frozen per-class log-prob biases

# --- component test-probability sources (name -> file, key) ---
SRC = {
    "knn": ("postprocess_tcn_la05.npz", "test_raw"),
    "lgb": ("gbdt_richest.npz", "test"), "xgb": ("xgb_richest.npz", "test"),
    "cat": ("_catboost_oof.npz", "test"),
    "bagging": ("div_bagging.npz", "test"), "mlp": ("div_mlp.npz", "test"),
    "linear": ("div_linear.npz", "test"), "inception": ("div_inception.npz", "test"),
    "tabular": ("div_tabular.npz", "test"),
    "resnet": ("div_resnet.npz", "test"), "transformer": ("div_transformer.npz", "test"),
    "bigru": ("bigru.npz", "test"),
}


def _load():
    TT = {}
    for n, (fn, k) in SRC.items():
        t = np.load(C / fn, allow_pickle=True)[k].astype(np.float64)
        TT[n] = t / t.sum(1, keepdims=True)
    fid = np.load(C / "windows.npz", allow_pickle=True)["file_id_test"].astype(int)
    return TT, fid


def gru_blend(TT):
    """0.75 * v21(9-model) + 0.10 resnet + 0.07 transformer + 0.08 bigru, renormalized."""
    cw = {"knn": .30, "lgb": .23, "xgb": .24, "cat": .23}
    v21w = {k: v * (1 - 0.45) for k, v in cw.items()}
    for n in ["bagging", "mlp", "linear", "inception", "tabular"]:
        v21w[n] = 0.45 / 5
    deep = {"resnet": .10, "transformer": .07, "bigru": .08}
    s = 1 - sum(deep.values())
    o = sum(w * s * TT[n] for n, w in v21w.items()) + sum(w * TT[n] for n, w in deep.items())
    return o / o.sum(1, keepdims=True)


def _shifted(p, b):
    lp = np.log(np.clip(p, 1e-8, 1.0)) + b
    lp -= lp.max(1, keepdims=True); e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


def _solve_share(prob, b, cls, target):           # 40-iter binary search for the cls bias
    lo, hi = -3.0, 4.0
    for _ in range(40):
        m = (lo + hi) / 2; bb = b.copy(); bb[cls] = m
        if (_shifted(prob, bb).argmax(1) == cls).mean() < target: lo = m
        else: hi = m
    return (lo + hi) / 2


def cal_gru(prob):                                  # c2=3.3%, c5=3.6%, others = v5 bias
    b = B0.copy()
    for _ in range(15):
        b[2] = _solve_share(prob, b, 2, 0.033); b[5] = _solve_share(prob, b, 5, 0.036)
    return _shifted(prob, b).argmax(1)


def cal_v5dist(prob):                               # match v5's class distribution (lr=0.5, 500 iters)
    target = np.array([41.1, 43.5, 3.6, 7.4, 1.0, 3.4]); target = target / target.sum()
    b = np.zeros(NC); lp = np.log(np.clip(prob, 1e-8, 1.0))
    for _ in range(500):
        cur = np.bincount((lp + b).argmax(1), minlength=NC) / len(prob)
        b += 0.5 * (np.log(target + 1e-6) - np.log(cur + 1e-6))
    return (lp + b).argmax(1)


def _write(pred, fid, name, out_dir):
    sample = pd.read_csv(SAMPLE)
    id2 = dict(zip(fid.tolist(), pred.tolist()))
    missing = [int(i) for i in sample["Id"] if int(i) not in id2]
    if missing:
        raise ValueError(f"{len(missing)} ids missing")
    sample = sample.copy(); sample["Label"] = sample["Id"].astype(int).map(id2).astype(int)
    p = Path(out_dir) / name; sample.to_csv(p, index=False)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="write to /tmp and diff vs submission/ (no overwrite)")
    args = ap.parse_args()
    TT, fid = _load()
    blend = gru_blend(TT)
    recipes = [("submission_tcn_gru.csv", cal_gru(blend)),
               ("submission_tcn_robust_v5dist.csv", cal_v5dist(blend))]
    out_dir = "/tmp" if args.verify else SUB
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ok_all = True
    for name, pred in recipes:
        p = _write(pred, fid, name, out_dir)
        if args.verify:
            ref = SUB / name
            if ref.exists():
                a = pd.read_csv(p); b = pd.read_csv(ref)
                merged = a.merge(b, on="Id", suffixes=("_new", "_ref"))
                ndiff = int((merged["Label_new"] != merged["Label_ref"]).sum())
                ok = (ndiff == 0); ok_all &= ok
                print(f"  {name:38s} reproduced; diff vs existing = {ndiff} rows  {'EXACT MATCH ✓' if ok else 'MISMATCH ✗'}")
            else:
                print(f"  {name:38s} written (no existing file to compare)")
        else:
            print(f"  wrote {p}")
    if args.verify:
        print("\nALL EXACT" if ok_all else "\nSOME MISMATCH — investigate")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
