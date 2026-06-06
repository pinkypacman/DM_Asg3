"""Blend three probability sources and write the submission.

Probability sources (NOT three different "models" exactly):
    1. TCN-KNN  — TCN encoder → 128-d embedding → KNN-on-embedding (with k-sweep + TTA);
                  cached at cache/postprocess_tcn{TAG}.npz
    2. LightGBM — gradient boosting on the rich 116-d hand-crafted features;
                  cached at cache/gbdt{LGBM_TAG[i]}.npz
    3. XGBoost  — gradient boosting (different framework) on the same rich features;
                  cached at cache/xgb{XGB_TAG[i]}.npz  (optional)

Procedure:
    1. Load TCN-KNN OOF + test (`oof_raw`/`test_raw` — pre-bias — so we re-tune on the blend).
    2. Load + average LightGBM caches across --gbdt-tags.
    3. If --xgb-tags is non-empty: load + average XGBoost caches across --xgb-tags.
    4. Grid-search blend weights (α_tcn, α_gbdt, α_xgb) summing to 1 on OOF macroF1
         - 2-way grid (no XGB)         → 1-D sweep over α_tcn
         - 3-way grid (with XGB)       → 2-D sweep over (α_tcn, α_gbdt), γ = 1 − α − β
    5. Re-tune per-class additive biases on the blended OOF (Powell, 6-d).
    6. Apply (weights, biases) to test → argmax → submission_<NAME>.csv.

Why re-tune biases on the blend? The TCN-only biases that postprocess learned are
optimal for the TCN-only OOF; once we mix in LGBM (and XGB), the optimal shift changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from .data import CACHE_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
SAMPLE_PATH = PROJECT_ROOT / "sample_submission.csv"
NUM_CLASSES = 6

# Known-good calibrations from past submissions. Use --frozen-recipe to bypass
# the α-search and bias-tuning entirely and ship a submission with these exact
# values applied to the supplied OOF/test caches. Designed to test "did the new
# component help on the leaderboard?" without re-tuning the calibration.
FROZEN_RECIPES = {
    # v5 — TCN + LGBM blend that scored 0.8008 (our best leaderboard).
    "v5": {
        "alpha_tcn": 0.40,
        "alpha_gbdt": 0.60,
        "alpha_xgb": 0.00,
        "biases": [0.332, 0.06, 0.991, 0.289, -0.077, -0.066],
    },
}


def _macro_f1(y, p):
    return float(f1_score(y, p.argmax(1), average="macro", zero_division=0))


def _per_class_f1(y, p):
    return f1_score(
        y, p.argmax(1),
        labels=list(range(NUM_CLASSES)), average=None, zero_division=0,
    )


def _shifted_softmax(p_in, biases):
    """Apply additive logit biases in log-prob space, return normalized probs."""
    lp = np.log(np.clip(p_in, 1e-8, 1.0)) + biases[None, :]
    lp = lp - lp.max(axis=1, keepdims=True)
    ep = np.exp(lp)
    return (ep / ep.sum(axis=1, keepdims=True)).astype(np.float32)


def search_alpha(oof_tcn, oof_gbdt, y, step=0.05):
    """1-D blend sweep (no XGB). Returns best α (TCN weight) and full table."""
    best = {"alpha": 0.5, "macro": -1.0}
    alphas = np.arange(0.0, 1.0 + 1e-9, step)
    rows = []
    for a in alphas:
        blend = a * oof_tcn + (1.0 - a) * oof_gbdt
        m = _macro_f1(y, blend)
        rows.append((float(a), m))
        if m > best["macro"]:
            best = {"alpha": float(a), "macro": m}
    return best, rows


def search_alpha_3way(oof_tcn, oof_gbdt, oof_xgb, y, step=0.05):
    """2-D simplex sweep over (α_tcn, α_gbdt), γ = 1 − α − β. Returns best triple + table."""
    best = {"alpha_tcn": 1/3, "alpha_gbdt": 1/3, "alpha_xgb": 1/3, "macro": -1.0}
    rows = []
    alphas = np.arange(0.0, 1.0 + 1e-9, step)
    for a in alphas:
        for b in alphas:
            if a + b > 1.0 + 1e-9:
                continue
            c = max(0.0, 1.0 - a - b)
            blend = a * oof_tcn + b * oof_gbdt + c * oof_xgb
            m = _macro_f1(y, blend)
            rows.append((float(a), float(b), float(c), m))
            if m > best["macro"]:
                best = {"alpha_tcn": float(a), "alpha_gbdt": float(b),
                        "alpha_xgb": float(c), "macro": m}
    return best, rows


def tune_biases(oof, y):
    """Powell minimize over 6-d additive log-prob bias to maximize OOF macroF1."""
    oof_logp = np.log(np.clip(oof, 1e-8, 1.0))

    def neg_macro(b):
        return -_macro_f1(y, oof_logp + b[None, :])

    res = minimize(
        neg_macro, x0=np.zeros(NUM_CLASSES), method="Powell",
        options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 300},
    )
    return res.x, -res.fun


def _load_avg_cache(paths: list, name: str) -> tuple:
    """Load oof + test from a list of cache files, averaging both."""
    if not paths:
        return None, None, None, None
    oofs, tests, macros, per_classes = [], [], [], []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found.")
        f = np.load(p)
        oofs.append(f["oof"])
        tests.append(f["test"])
        macros.append(float(f["oof_macro"]))
        per_classes.append(f["oof_per_class"])
    oof = np.mean(np.stack(oofs, axis=0), axis=0).astype(np.float32)
    test = np.mean(np.stack(tests, axis=0), axis=0).astype(np.float32)
    print(f"  {name}: {len(paths)} model(s) averaged — per-model macroF1 {[round(m, 4) for m in macros]}")
    return oof, test, macros, np.stack(per_classes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcn-tag", default="_la05",
                        help="Tag of postprocess_tcn{TAG}.npz to consume (default: _la05).")
    parser.add_argument("--gbdt-tags", nargs="*", default=[""],
                        help="One or more tags for cache/gbdt{TAG}.npz files; default ['']. "
                             "Provide multiple to average across them.")
    parser.add_argument("--xgb-tags", nargs="*", default=[],
                        help="Zero or more tags for cache/xgb{TAG}.npz files. "
                             "Empty list → 2-way blend (TCN + GBDT only).")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--name", default="submission_tcn_v5.csv")
    args = parser.parse_args()

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load TCN side (pre-biases — we'll re-tune on the blend) ----
    pp_path = CACHE_DIR / f"postprocess_tcn{args.tcn_tag}.npz"
    if not pp_path.exists():
        raise FileNotFoundError(
            f"{pp_path} not found. Run postprocess first: "
            f"./run.sh STAGES='postprocess' SUBMISSION_NAME=tcn_vX LA_LOSS=1 TCN_TAG={args.tcn_tag}"
        )
    pp = np.load(pp_path, allow_pickle=True)
    oof_tcn = pp["oof_raw"]
    test_tcn = pp["test_raw"]
    y = pp["y_oof"]
    tcn_only_macro = float(pp["oof_macro_tuned"])
    tcn_sweep_macro = float(pp["oof_macro_sweep"])

    # ---- load GBDT (one or more) and XGB (zero or more) ----
    print("=" * 72)
    gbdt_paths = [CACHE_DIR / f"gbdt{t}.npz" for t in args.gbdt_tags]
    oof_gbdt, test_gbdt, gbdt_macros, gbdt_per_class = _load_avg_cache(gbdt_paths, "GBDT")
    if args.xgb_tags:
        xgb_paths = [CACHE_DIR / f"xgb{t}.npz" for t in args.xgb_tags]
        oof_xgb, test_xgb, xgb_macros, xgb_per_class = _load_avg_cache(xgb_paths, "XGB ")
    else:
        oof_xgb, test_xgb = None, None
        print(f"  XGB : (none — 2-way blend mode)")

    # ---- consistency checks ----
    file_id_test = np.load(gbdt_paths[0])["file_id_test"].astype(int)
    for oof_other in [oof_gbdt] + ([oof_xgb] if oof_xgb is not None else []):
        assert np.allclose(oof_other.sum(axis=1), 1.0, atol=1e-4), "OOF rows don't sum to 1"
    assert np.allclose(oof_tcn.sum(axis=1), 1.0, atol=1e-4), "TCN OOF rows don't sum to 1"

    print()
    print(f"TCN OOF (post-sweep, post-TTA, pre-biases) : macroF1 {tcn_sweep_macro:.4f}")
    print(f"TCN OOF (post-biases, single-model best)   : macroF1 {tcn_only_macro:.4f}")
    print(f"GBDT OOF (averaged across {len(gbdt_paths)} model(s)) : macroF1 {_macro_f1(y, oof_gbdt):.4f}")
    print(f"  GBDT per-class : {[round(float(v), 3) for v in _per_class_f1(y, oof_gbdt)]}")
    if oof_xgb is not None:
        print(f"XGB  OOF (averaged across {len(args.xgb_tags)} model(s)) : macroF1 {_macro_f1(y, oof_xgb):.4f}")
        print(f"  XGB  per-class : {[round(float(v), 3) for v in _per_class_f1(y, oof_xgb)]}")
    print(f"  TCN  per-class : {[round(float(v), 3) for v in _per_class_f1(y, oof_tcn)]}")
    print("=" * 72)

    # ---- (3) blend search ----
    if oof_xgb is None:
        print("\n--- α sweep (TCN weight, 2-way blend) ---")
        best, rows = search_alpha(oof_tcn, oof_gbdt, y, step=args.alpha_step)
        for a, m in rows:
            flag = "  ← best" if abs(a - best["alpha"]) < 1e-9 else ""
            print(f"  α_tcn={a:.2f}  α_gbdt={1-a:.2f}  macroF1={m:.4f}{flag}")
        a_t, a_g, a_x = best["alpha"], 1.0 - best["alpha"], 0.0
        blend_oof = a_t * oof_tcn + a_g * oof_gbdt
        blend_test = a_t * test_tcn + a_g * test_gbdt
    else:
        print("\n--- 2-D simplex sweep (TCN, GBDT, XGB weights) ---")
        best, rows = search_alpha_3way(oof_tcn, oof_gbdt, oof_xgb, y, step=args.alpha_step)
        # Print top 12 by macroF1, then the best
        rows_sorted = sorted(rows, key=lambda r: -r[3])
        for a, b, c, m in rows_sorted[:12]:
            print(f"  α_tcn={a:.2f}  α_gbdt={b:.2f}  α_xgb={c:.2f}  macroF1={m:.4f}")
        print(f"  ✔ best: α_tcn={best['alpha_tcn']:.2f}  α_gbdt={best['alpha_gbdt']:.2f}  "
              f"α_xgb={best['alpha_xgb']:.2f}  macroF1={best['macro']:.4f}")
        a_t, a_g, a_x = best["alpha_tcn"], best["alpha_gbdt"], best["alpha_xgb"]
        blend_oof = a_t * oof_tcn + a_g * oof_gbdt + a_x * oof_xgb
        blend_test = a_t * test_tcn + a_g * test_gbdt + a_x * test_xgb

    pre_bias_macro = _macro_f1(y, blend_oof)
    pre_bias_pc = _per_class_f1(y, blend_oof)
    print(f"\n  blend macroF1 (pre-bias) = {pre_bias_macro:.4f}")
    print(f"  per-class                = {[round(float(v), 3) for v in pre_bias_pc]}")

    # ---- (4) Re-tune threshold biases on the blended OOF ----
    print("\n--- threshold tuning on blended OOF ---")
    b_star, tuned_macro = tune_biases(blend_oof, y)
    tuned_oof = _shifted_softmax(blend_oof, b_star)
    tuned_pc = _per_class_f1(y, tuned_oof)
    print(f"  biases = {[round(float(v), 3) for v in b_star]}")
    print(f"  OOF macroF1 pre-bias  {pre_bias_macro:.4f}  →  tuned {tuned_macro:.4f}  (Δ {tuned_macro - pre_bias_macro:+.4f})")
    print(f"  tuned per-class       {[round(float(v), 3) for v in tuned_pc]}")

    # ---- (5) Apply (α, biases) to test ----
    test_final = _shifted_softmax(blend_test, b_star)
    pred = test_final.argmax(axis=1).astype(int)
    shares = np.bincount(pred, minlength=NUM_CLASSES) / len(pred) * 100

    if shares[0] + shares[1] < 80.0:
        print(f"  ⚠ c0+c1 share dropped to {(shares[0]+shares[1]):.1f}% (<80%) — reverting biases.")
        b_star = np.zeros_like(b_star)
        test_final = blend_test
        pred = test_final.argmax(axis=1).astype(int)
        shares = np.bincount(pred, minlength=NUM_CLASSES) / len(pred) * 100

    # ---- (6) write submission ----
    sample = pd.read_csv(SAMPLE_PATH)
    id_to_pred = dict(zip(file_id_test.tolist(), pred.tolist()))
    missing = [int(i) for i in sample["Id"] if int(i) not in id_to_pred]
    if missing:
        raise ValueError(f"{len(missing)} Ids missing from prediction (first 5): {missing[:5]}")
    out_path = SUBMISSION_DIR / args.name
    sample = sample.copy()
    sample["Label"] = sample["Id"].astype(int).map(id_to_pred).astype(int)
    sample.to_csv(out_path, index=False)

    print()
    print("=" * 72)
    print(f"Blend weights         : α_tcn={a_t:.2f}  α_gbdt={a_g:.2f}  α_xgb={a_x:.2f}")
    print(f"OOF macroF1 progression")
    print(f"  TCN only (post-biases): {tcn_only_macro:.4f}")
    print(f"  GBDT only             : {_macro_f1(y, oof_gbdt):.4f}")
    if oof_xgb is not None:
        print(f"  XGB  only             : {_macro_f1(y, oof_xgb):.4f}")
    print(f"  blend (pre-bias)      : {pre_bias_macro:.4f}")
    print(f"  blend (post-bias)     : {tuned_macro:.4f}")
    print(f"submission             : {out_path}")
    print(f"  test pred shares (%)  : "
          f"{ {c: round(float(shares[c]), 2) for c in range(NUM_CLASSES)} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
