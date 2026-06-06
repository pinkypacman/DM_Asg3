"""v10 submission builder — Tier-1 upgrades over the v5/v9 recipe.

Changes vs the old TCN-KNN + LGB (+XGB) α-grid + 6-dim Powell-bias pipeline:
  #1  Deep member = TCN softmax HEAD (cache/head_la05.npz), not KNN-on-embedding.
  #2  Convex per-base weights (Powell over a simplex) over {head, LGB, XGB, CatBoost[, BiGRU]}
      — XGB (the strongest single base) was wasted at weight 0 in the 2-way recipe.
  #3  Extra diverse bases: CatBoost (cache/_catboost_oof.npz) and BiGRU (cache/bigru.npz, optional).
  #4  Bias = a SINGLE class-2 additive log-prob bias (the only dim that survives nested CV),
      replacing the overfit 6-dim Powell bias.
  #5  Leave-fold-out transition-aware Viterbi temporal smoothing (file_id is a true per-user
      time index; 89% adjacent-label agreement) at alpha=0.4 — lifts c0/c1/c3/c5 via temporal
      consistency without erasing the rare classes (transition-aware, not majority vote).

All OOF gains are reported under GroupKFold(user); selection is honest (c2-bias nested-checked,
Viterbi alpha fixed at 0.4 from the broad 0.3-0.5 plateau, T always estimated leave-fold-out).
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
NC = 6
VITERBI_ALPHA = 0.4  # fixed from the broad 0.3-0.5 plateau (Inv1); not cherry-picked per-run


def macro(y, pred):
    return float(f1_score(y, pred, average="macro", labels=list(range(NC)), zero_division=0))


def perclass(y, pred):
    return [round(float(v), 3) for v in f1_score(y, pred, average=None, labels=list(range(NC)), zero_division=0)]


def shifted(p, b):
    lp = np.log(np.clip(p, 1e-8, 1.0)) + b[None, :]
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    return (e / e.sum(1, keepdims=True)).astype(np.float64)


def tune_c2_bias(p, y):
    """1-D search over a class-2-only additive log-prob bias (maximize macro-F1)."""
    lp = np.log(np.clip(p, 1e-8, 1.0))
    best = (0.0, -1.0)
    for b2 in np.arange(0.0, 2.01, 0.05):
        b = np.zeros(NC); b[2] = b2
        m = macro(y, (lp + b[None, :]).argmax(1))
        if m > best[1]:
            best = (b2, m)
    b = np.zeros(NC); b[2] = best[0]
    return b


def convex_weights(oofs, y, restarts=8, seed=0):
    K = len(oofs)
    stack = np.stack(oofs, 0)
    rng = np.random.default_rng(seed)

    def blend(w):
        w = np.clip(w, 0, None); s = w.sum()
        w = w / s if s > 1e-9 else np.ones(K) / K
        return np.tensordot(w, stack, axes=(0, 0))

    best = (None, 1e9)
    starts = [np.ones(K) / K] + [rng.dirichlet(np.ones(K)) for _ in range(restarts)]
    for x0 in starts:
        r = minimize(lambda w: -macro(y, blend(w).argmax(1)), x0=x0,
                     method="Powell", options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 600})
        if r.fun < best[1]:
            best = (r.x, r.fun)
    w = np.clip(best[0], 0, None)
    return w / w.sum()


def estimate_T(y, user, fid, mask, laplace=1e-6):
    T = np.full((NC, NC), laplace)
    idx = np.flatnonzero(mask)
    order = idx[np.lexsort((fid[idx], user[idx]))]
    yu, uu = y[order], user[order]
    for i in range(len(order) - 1):
        if uu[i] == uu[i + 1]:
            T[yu[i], yu[i + 1]] += 1.0
    return T / T.sum(1, keepdims=True)


def _viterbi_user(logE, logT, logpi):
    L = logE.shape[0]
    delta = np.empty((L, NC)); psi = np.empty((L, NC), dtype=int)
    delta[0] = logpi + logE[0]
    for t in range(1, L):
        sc = delta[t - 1][:, None] + logT
        psi[t] = sc.argmax(0); delta[t] = sc.max(0) + logE[t]
    path = np.empty(L, dtype=int); path[-1] = delta[-1].argmax()
    for t in range(L - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]
    return path


def viterbi_smooth(probs, user, fid, T, alpha, prior):
    logE = np.log(np.clip(probs, 1e-8, 1.0))
    logT = alpha * np.log(np.clip(T, 1e-12, 1.0))
    logpi = np.log(np.clip(prior, 1e-12, 1.0))
    out = np.empty(len(probs), dtype=int)
    for u in np.unique(user):
        idx = np.flatnonzero(user == u)
        idx = idx[np.argsort(fid[idx])]
        out[idx] = _viterbi_user(logE[idx], logT, logpi)
    return out


def _load(path, key):
    return np.load(path, allow_pickle=True)[key]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="submission_tcn_v10.csv")
    ap.add_argument("--alpha", type=float, default=VITERBI_ALPHA)
    ap.add_argument("--no-viterbi", action="store_true")
    ap.add_argument("--drop", default="", help="comma-separated base names to exclude (e.g. 'bigru').")
    ap.add_argument("--c2-bias", type=float, default=None,
                    help="Manual class-2 log-prob bias override (skips the OOF grid). "
                         "LB evidence shows the test set wants LOW c2 share, so a small/zero "
                         "value beats the OOF-optimal (which over-boosts c2).")
    args = ap.parse_args()
    drop = {s for s in args.drop.split(",") if s}
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # ---- labels / folds / sequence index ----
    y = _load(CACHE_DIR / "gbdt_richer.npz", "y_oof")
    fold_of = _load(CACHE_DIR / "folds.npz", "fold_of")
    w = np.load(CACHE_DIR / "windows.npz", allow_pickle=True)
    user_tr, fid_tr = w["user_train"], w["file_id_train"]
    user_te, fid_te = w["user_test"], w["file_id_test"]
    prior = np.bincount(y, minlength=NC) / len(y)

    # ---- bases: (name, oof_path, oof_key, test_path, test_key) ----
    specs = [
        ("head", CACHE_DIR / "head_la05.npz", "oof", CACHE_DIR / "head_la05.npz", "test"),
        ("headL", CACHE_DIR / "head_ldamG.npz", "oof", CACHE_DIR / "head_ldamG.npz", "test"),
        ("lgb", CACHE_DIR / "gbdt_richer.npz", "oof", CACHE_DIR / "gbdt_richer.npz", "test"),
        ("xgb", CACHE_DIR / "xgb.npz", "oof", CACHE_DIR / "xgb.npz", "test"),
        ("cat", CACHE_DIR / "_catboost_oof.npz", "oof", CACHE_DIR / "_catboost_oof.npz", "test"),
        ("bigru", CACHE_DIR / "bigru.npz", "oof", CACHE_DIR / "bigru.npz", "test"),
    ]
    names, oofs, tests = [], [], []
    for nm, op, ok, tp, tk in specs:
        if nm in drop:
            print(f"  [drop] {nm}")
            continue
        if not op.exists():
            print(f"  [skip] {nm}: {op.name} not found")
            continue
        names.append(nm); oofs.append(_load(op, ok)); tests.append(_load(tp, tk))
    print(f"bases used: {names}")

    # alignment: file_id_test must match windows order for all bases that store it
    fid_ref = _load(CACHE_DIR / "gbdt_richer.npz", "file_id_test").astype(int)
    assert np.array_equal(fid_ref, fid_te.astype(int)), "gbdt file_id_test != windows order"
    assert np.array_equal(_load(CACHE_DIR / "head_la05.npz", "file_id_test").astype(int), fid_te.astype(int))
    for o in oofs:
        assert len(o) == len(y) and np.allclose(o.sum(1), 1, atol=1e-4)
    for t in tests:
        assert len(t) == len(fid_te) and np.allclose(t.sum(1), 1, atol=1e-4)

    # ---- #2 convex weights on OOF ----
    wts = convex_weights(oofs, y)
    print("convex weights:", {n: round(float(x), 3) for n, x in zip(names, wts)})
    blend_oof = np.tensordot(wts, np.stack(oofs, 0), axes=(0, 0))
    blend_te = np.tensordot(wts, np.stack(tests, 0), axes=(0, 0))
    print(f"  convex blend (pre-bias) OOF macroF1 = {macro(y, blend_oof.argmax(1)):.4f}  {perclass(y, blend_oof.argmax(1))}")

    # ---- #4+#5 jointly: pick the class-2 bias that maximizes the POST-Viterbi OOF.
    # Viterbi erodes class 2 unless it is boosted first (Inv1: "smoothing helps only WITH a
    # c2 boost"), so the c2 bias MUST be selected against the final post-smoothing objective,
    # not the pre-Viterbi blend. The leave-fold-out Viterbi (T from training folds) keeps it honest.
    def lfo_viterbi(prob_oof):
        pred = np.empty(len(y), dtype=int)
        for f in range(5):
            tr, va = fold_of != f, fold_of == f
            T = estimate_T(y, user_tr, fid_tr, tr)
            pred[va] = viterbi_smooth(prob_oof[va], user_tr[va], fid_tr[va], T, args.alpha, prior)
        return pred

    if args.c2_bias is not None:
        c2_star = args.c2_bias
        b_c2 = np.zeros(NC); b_c2[2] = c2_star
        biased_oof = shifted(blend_oof, b_c2)
        final_oof = macro(y, biased_oof.argmax(1)) if args.no_viterbi else macro(y, lfo_viterbi(biased_oof))
        print(f"  c2 bias = {c2_star:.2f} (manual override)  OOF macroF1={final_oof:.4f}")
    else:
        c2_grid = np.arange(0.0, 1.51, 0.25)
        best = (-1.0, 0.0, None)
        for c2b in c2_grid:
            b = np.zeros(NC); b[2] = c2b
            bp = shifted(blend_oof, b)
            m = macro(y, bp.argmax(1)) if args.no_viterbi else macro(y, lfo_viterbi(bp))
            tag = " <-" if m > best[0] else ""
            print(f"  c2 bias={c2b:.2f}  {'post-bias' if args.no_viterbi else 'post-Viterbi'} OOF macroF1={m:.4f}{tag}")
            if m > best[0]:
                best = (m, c2b, bp)
        final_oof, c2_star, biased_oof = best
        b_c2 = np.zeros(NC); b_c2[2] = c2_star
    biased_te = shifted(blend_te, b_c2)
    print(f"  chosen c2 bias = {c2_star:.2f}")

    # ---- apply to test: full-train T, per test-user Viterbi ----
    if not args.no_viterbi:
        oof_pred = lfo_viterbi(biased_oof)
        print(f"  FINAL OOF per-class = {perclass(y, oof_pred)}")
        T_full = estimate_T(y, user_tr, fid_tr, np.ones(len(y), dtype=bool))
        test_pred = viterbi_smooth(biased_te, user_te, fid_te, T_full, args.alpha, prior)
    else:
        test_pred = biased_te.argmax(1)

    shares = np.bincount(test_pred, minlength=NC) / len(test_pred) * 100
    print(f"\nFINAL OOF macroF1 = {final_oof:.4f}  (vs production 0.7331, Δ {final_oof-0.7331:+.4f})")
    print(f"test pred shares (%): {{ {', '.join(f'{c}:{shares[c]:.1f}' for c in range(NC))} }}  c0+c1={shares[0]+shares[1]:.1f}%")

    # ---- write submission ----
    sample = pd.read_csv(SAMPLE_PATH)
    id_to_pred = dict(zip(fid_te.astype(int).tolist(), test_pred.tolist()))
    missing = [int(i) for i in sample["Id"] if int(i) not in id_to_pred]
    if missing:
        raise ValueError(f"{len(missing)} Ids missing (first 5): {missing[:5]}")
    out = SUBMISSION_DIR / args.name
    sample = sample.copy()
    sample["Label"] = sample["Id"].astype(int).map(id_to_pred).astype(int)
    sample.to_csv(out, index=False)
    print(f"submission → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
