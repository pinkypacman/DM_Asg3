"""v15 — fundamentals-first decision layer (research-backed, replaces OOF-tuned biases).

Pipeline (all peer-reviewed, see deep-research 2026-06-04):
  base   : simple average of proven strong models (TCN-KNN + LightGBM-richer + XGBoost).
           Equal weights (no OOF-overfit convex weights).
  cal    : Dirichlet calibration = multinomial logistic regression on log-probs
           (Kull et al. NeurIPS 2019). Fit on group-disjoint OOF; the precondition that
           makes prior-correction transfer across the 40 unseen users.
  decide : PRIOR-MATCH to the train prior. Adversarial validation (train-vs-test AUC=0.41)
           proved NO covariate shift ⇒ test prior ≈ train prior, and the LB confirmed the
           class-2 optimum sits at the train prior. So we set per-class additive log-prob
           biases that make the predicted class distribution equal the train prior — a
           principled, low-variance rule instead of the winner's-curse OOF threshold search.

Why this should beat v5 (0.8008): v5's predicted shares deviate from the train prior
(over-predicts c3 7.4% vs 6.0%, under-predicts c5 3.4% vs 4.8%); prior-match removes those.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from .data import CACHE_DIR

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"; SAMPLE = ROOT / "sample_submission.csv"; NC = 6


def macro(y, p): return float(f1_score(y, p, average="macro", labels=list(range(NC)), zero_division=0))
def pcls(y, p): return [round(x, 3) for x in f1_score(y, p, average=None, labels=list(range(NC)), zero_division=0)]


def dirichlet_calibrate(base_oof, y, fold, base_te):
    """Multinomial LR on log-probs. Returns (group-CV OOF cal, test cal fit on all OOF)."""
    logp = np.log(np.clip(base_oof, 1e-6, 1.0))
    cal_oof = np.zeros_like(base_oof)
    for f in np.unique(fold):
        tr, va = fold != f, fold == f
        m = LogisticRegression(C=1.0, max_iter=2000, multi_class="multinomial").fit(logp[tr], y[tr])
        cal_oof[va] = m.predict_proba(logp[va])
    m_all = LogisticRegression(C=1.0, max_iter=2000, multi_class="multinomial").fit(logp, y)
    cal_te = m_all.predict_proba(np.log(np.clip(base_te, 1e-6, 1.0)))
    return cal_oof.astype(np.float32), cal_te.astype(np.float32)


def prior_match_bias(P, target, lr=0.5, iters=400):
    """Additive log-prob bias so argmax(log P + b) has class distribution == target."""
    b = np.zeros(NC)
    for _ in range(iters):
        pred = (np.log(np.clip(P, 1e-8, 1.0)) + b).argmax(1)
        cur = np.bincount(pred, minlength=NC) / len(pred)
        b += lr * (np.log(target + 1e-6) - np.log(cur + 1e-6))
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="submission_tcn_v15.csv")
    ap.add_argument("--base", default="knn,lgb,xgb", help="comma list of base models to average")
    ap.add_argument("--no-cal", action="store_true", help="skip Dirichlet calibration (ablation)")
    args = ap.parse_args()
    SUB.mkdir(exist_ok=True)

    y = np.load(CACHE_DIR / "gbdt_richer.npz")["y_oof"]
    fold = np.load(CACHE_DIR / "folds.npz")["fold_of"]
    fid = np.load(CACHE_DIR / "windows.npz", allow_pickle=True)["file_id_test"].astype(int)
    prior = np.bincount(y, minlength=NC) / len(y)

    src = {
        "knn": ("postprocess_tcn_la05.npz", "oof_raw", "test_raw"),
        "lgb": ("gbdt_richer.npz", "oof", "test"),
        "xgb": ("xgb.npz", "oof", "test"),
        "cat": ("_catboost_oof.npz", "oof", "test"),
        "head": ("head_la05.npz", "oof", "test"),
    }
    keys = [k for k in args.base.split(",") if k]
    oofs, tes = [], []
    for k in keys:
        fn, ok, tk = src[k]; z = np.load(CACHE_DIR / fn, allow_pickle=True)
        oofs.append(z[ok]); tes.append(z[tk])
    base_oof = np.mean(oofs, 0).astype(np.float32); base_te = np.mean(tes, 0).astype(np.float32)
    print(f"base = mean({keys}): OOF argmax macro={macro(y, base_oof.argmax(1)):.4f}")

    if args.no_cal:
        cal_oof, cal_te = base_oof, base_te; print("calibration: SKIPPED (ablation)")
    else:
        cal_oof, cal_te = dirichlet_calibrate(base_oof, y, fold, base_te)
        print(f"Dirichlet-cal: OOF argmax macro={macro(y, cal_oof.argmax(1)):.4f}")

    # OOF report: prior-match per held-out fold (honest)
    pred_oof = np.zeros(len(y), dtype=int)
    for f in np.unique(fold):
        tr, va = fold != f, fold == f
        b = prior_match_bias(cal_oof[tr], prior)
        pred_oof[va] = (np.log(np.clip(cal_oof[va], 1e-8, 1.0)) + b).argmax(1)
    print(f"\ncal + prior-match (nested OOF): macro={f1_score(y, pred_oof, average='macro', zero_division=0):.4f}")
    print(f"  per-class F1 : {pcls(y, pred_oof)}")
    print(f"  OOF shares % : {(np.bincount(pred_oof, minlength=NC)/len(y)*100).round(1).tolist()}")

    # TEST: prior-match bias fit on the test calibrated probs to hit train prior exactly
    b_te = prior_match_bias(cal_te, prior)
    pred_te = (np.log(np.clip(cal_te, 1e-8, 1.0)) + b_te).argmax(1)
    sh = np.bincount(pred_te, minlength=NC) / len(pred_te) * 100
    print(f"\nTEST shares %  : {sh.round(1).tolist()}   (train prior: {(prior*100).round(1).tolist()})")

    sample = pd.read_csv(SAMPLE); m = dict(zip(fid.tolist(), pred_te.tolist()))
    miss = [int(i) for i in sample["Id"] if int(i) not in m]
    if miss: raise ValueError(f"{len(miss)} missing ids")
    sample["Label"] = sample["Id"].astype(int).map(m).astype(int)
    sample.to_csv(SUB / args.name, index=False)
    print(f"submission → {SUB/args.name}")


if __name__ == "__main__":
    sys.exit(main())
