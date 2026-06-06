"""Ablation studies for the report (Q2 preprocessing, Q4 core design choices).

Three studies, all under GroupKFold(user) OOF:

 A. FEATURE PROGRESSION (Q2): one fixed LightGBM config trained on each feature
    set (34 lean -> 116 rich -> 196 +jerk/RMS -> 201 +position). Isolates the
    macro-F1 / class-2 gain from each preprocessing/feature group.

 B. ENSEMBLE ABLATION (Q4): macro-F1 as we go from the best single model ->
    4 core models -> +5 diverse families -> +3 deep-sequence models (the final
    "gru-blend"), using the exact frozen weights from reproduce/build_finals.py.

 C. DECISION-LAYER ABLATION (Q4): on the final blended OOF, compare raw argmax
    vs v5 frozen biases vs the gru c2/c5 share-calibration vs prior-match.

Also writes analysis/figures/{feature_progression,confusion_matrix,ensemble_ablation}.png.

LightGBM config mirrors src/gbdt.py (num_leaves=63, lr=0.05, class_weight=balanced)
with a smaller n_estimators + early stopping so the ablation runs in a few minutes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NC = 6
B0 = np.array([0.332, 0.06, 0.991, 0.289, -0.077, -0.066])  # v5 frozen decision biases

# ---- the final blend definition (identical to reproduce/build_finals.py) ----
MEMBERS = {
    "knn": ("postprocess_tcn_la05.npz", "oof_raw", "test_raw"),
    "lgb": ("gbdt_richest.npz", "oof", "test"), "xgb": ("xgb_richest.npz", "oof", "test"),
    "cat": ("_catboost_oof.npz", "oof", "test"), "bagging": ("div_bagging.npz", "oof", "test"),
    "mlp": ("div_mlp.npz", "oof", "test"), "linear": ("div_linear.npz", "oof", "test"),
    "inception": ("div_inception.npz", "oof", "test"), "tabular": ("div_tabular.npz", "oof", "test"),
    "resnet": ("div_resnet.npz", "oof", "test"), "transformer": ("div_transformer.npz", "oof", "test"),
    "bigru": ("bigru.npz", "oof", "test"),
}
D = 0.45
CORE = {"knn": .30, "lgb": .23, "xgb": .24, "cat": .23}
DIVERSE = ["bagging", "mlp", "linear", "inception", "tabular"]
DEEP = {"resnet": .10, "transformer": .07, "bigru": .08}


def macro(y, p):
    return f1_score(y, p, average="macro", labels=list(range(NC)), zero_division=0)


def perclass(y, p):
    return [round(v, 3) for v in f1_score(y, p, average=None, labels=list(range(NC)), zero_division=0)]


def shifted(p, b):
    lp = np.log(np.clip(p, 1e-8, 1)) + b
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


# ============================ A. FEATURE PROGRESSION ============================
def lgb_oof(F, y, fold):
    """Fixed, fast LightGBM (early-stopped) — same config across feature sets so the
    macro-F1 deltas isolate the feature contribution. Faster than production gbdt.py
    (fewer trees), so absolute scores sit slightly below the cached richest model."""
    rng = np.random.default_rng(0)
    oof = np.zeros((len(y), NC), np.float32)
    for f in range(5):
        tr_idx = np.flatnonzero(fold != f)
        va = fold == f
        # carve 12% of training rows for early-stopping validation
        rng.shuffle(tr_idx)
        cut = int(0.12 * len(tr_idx))
        es_idx, fit_idx = tr_idx[:cut], tr_idx[cut:]
        m = lgb.LGBMClassifier(objective="multiclass", num_class=NC, learning_rate=0.08,
                               num_leaves=31, n_estimators=600, min_data_in_leaf=40,
                               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
                               class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
        m.fit(F[fit_idx], y[fit_idx], eval_set=[(F[es_idx], y[es_idx])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        oof[va] = m.predict_proba(F[va])
    return oof


def feature_progression(y, fold):
    print("\n=== A. FEATURE PROGRESSION (fixed LightGBM, GroupKFold OOF) ===")
    sets = [("lean (34)", "features.npz"), ("+spectral/temporal/stats = rich (116)", "features_rich.npz"),
            ("+jerk/RMS = richest (196)", "features_richest.npz"), ("+position (201)", "features_pos.npz")]
    rows = []
    for label, fn in sets:
        z = np.load(CACHE / fn, allow_pickle=True)
        F = z["F_train"].astype(np.float32)
        oof = lgb_oof(F, y, fold)
        mac = macro(y, oof.argmax(1)); c2 = f1_score(y, oof.argmax(1), labels=[2], average=None, zero_division=0)[0]
        rows.append((label, F.shape[1], mac, c2))
        print(f"  {label:42s} dim={F.shape[1]:3d}  macroF1={mac:.4f}  c2-F1={c2:.3f}")
    plot_feature_progression(rows)
    return rows


def plot_feature_progression(rows):
    """Two clean panels (no twin-axis overlap): overall macro-F1 and class-2 F1."""
    labels = ["lean", "rich", "richest", "+position"]
    macs = [r[2] for r in rows]
    c2s = [r[3] for r in rows]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, color, ttl, ylab in [
        (axL, macs, "#4C72B0", "Overall macro-F1 rises", "macro-F1"),
        (axR, c2s, "#C44E52", "Class-2 F1 stays flat (data ceiling)", "class-2 F1"),
    ]:
        ax.plot(labels, vals, "o-", color=color, lw=2, ms=7)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.55 + 0.006
        ax.set_ylim(lo - pad, hi + pad)
        for x, v in zip(labels, vals):
            ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color=color, fontweight="bold")
        ax.set_title(ttl, color=color, fontsize=11)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, axis="y")
        ax.margins(x=0.12)
    fig.suptitle("Feature progression (single fixed LightGBM, GroupKFold OOF)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "feature_progression.png", dpi=130)
    plt.close(fig)


# ============================ B. ENSEMBLE ABLATION ============================
def load_oof():
    O = {}
    for n, (fn, ok, _) in MEMBERS.items():
        o = np.load(CACHE / fn, allow_pickle=True)[ok].astype(float)
        O[n] = o / o.sum(1, keepdims=True)
    return O


def blend_with(O, deep):
    w = {k: v * (1 - D) for k, v in CORE.items()}
    for n in DIVERSE:
        w[n] = D / 5
    s = 1 - sum(deep.values())
    out = sum(w[n] * s * O[n] for n in w) + sum(deep[n] * O[n] for n in deep)
    return out / out.sum(1, keepdims=True)


def ensemble_ablation(y, O):
    print("\n=== B. ENSEMBLE ABLATION (raw OOF argmax macro-F1) ===")
    stages = []
    # best single
    best_n = max(CORE, key=lambda n: macro(y, O[n].argmax(1)))
    stages.append((f"best single ({best_n})", macro(y, O[best_n].argmax(1))))
    # core-4 only (diverse + deep weight removed): renormalize core weights
    cw = np.array(list(CORE.values())); cw = cw / cw.sum()
    core_blend = sum(w * O[n] for w, n in zip(cw, CORE))
    stages.append(("4 core (knn+lgb+xgb+cat)", macro(y, core_blend.argmax(1))))
    # +5 diverse (v21 base, no deep)
    stages.append(("+5 diverse families (v21)", macro(y, blend_with(O, {}).argmax(1))))
    # +deep (gru blend)
    gru = blend_with(O, DEEP)
    stages.append(("+3 deep-seq (final gru-blend)", macro(y, gru.argmax(1))))
    for nm, m in stages:
        print(f"  {nm:34s} macroF1={m:.4f}")
    # plot
    plt.figure(figsize=(6.6, 3.6))
    names = [s[0] for s in stages]; vals = [s[1] for s in stages]
    bars = plt.barh(range(len(names)), vals, color="#55A868")
    plt.yticks(range(len(names)), names, fontsize=8); plt.gca().invert_yaxis()
    plt.xlim(min(vals) - 0.01, max(vals) + 0.006)
    for b, v in zip(bars, vals):
        plt.text(v + 0.0005, b.get_y() + b.get_height()/2, f"{v:.4f}", va="center", fontsize=8)
    plt.xlabel("OOF macro-F1"); plt.title("Ensemble ablation: diversity stacks")
    plt.tight_layout(); plt.savefig(FIG / "ensemble_ablation.png", dpi=130); plt.close()
    return gru, stages


# ============================ C. DECISION LAYER ============================
def solve(prob, b, cls, tg):
    lo, hi = -3.0, 4.0
    for _ in range(40):
        m = (lo + hi) / 2; bb = b.copy(); bb[cls] = m
        if (shifted(prob, bb).argmax(1) == cls).mean() < tg:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def decision_ablation(y, fold, gru):
    print("\n=== C. DECISION-LAYER ABLATION (on final gru-blend OOF) ===")
    prior = np.bincount(y, minlength=NC) / len(y)
    # raw argmax
    print(f"  raw argmax                 macroF1={macro(y, gru.argmax(1)):.4f}  c2-F1={perclass(y, gru.argmax(1))[2]}")
    # v5 biases
    p_v5 = shifted(gru, B0).argmax(1)
    print(f"  + v5 frozen biases         macroF1={macro(y, p_v5):.4f}  c2-F1={perclass(y, p_v5)[2]}")
    # gru c2/c5 calibration (nested per fold: fit shares within each held-out fold)
    pred = np.zeros(len(y), int)
    for f in range(5):
        va = fold == f; b = B0.copy()
        for _ in range(15):
            b[2] = solve(gru[va], b, 2, .033); b[5] = solve(gru[va], b, 5, .036)
        pred[va] = shifted(gru[va], b).argmax(1)
    print(f"  + c2/c5 share-cal (gru)    macroF1={macro(y, pred):.4f}  c2-F1={perclass(y, pred)[2]}  shares={(np.bincount(pred,minlength=NC)/len(y)*100).round(1).tolist()}")
    # prior-match (robust_v5dist style, to train prior, nested per fold)
    predpm = np.zeros(len(y), int)
    for f in range(5):
        va = fold == f; b = np.zeros(NC)
        for _ in range(500):
            cur = np.bincount((np.log(np.clip(gru[va], 1e-8, 1)) + b).argmax(1), minlength=NC) / va.sum()
            b += 0.5 * (np.log(prior + 1e-6) - np.log(cur + 1e-6))
        predpm[va] = (np.log(np.clip(gru[va], 1e-8, 1)) + b).argmax(1)
    print(f"  + prior-match (robust)     macroF1={macro(y, predpm):.4f}  c2-F1={perclass(y, predpm)[2]}  shares={(np.bincount(predpm,minlength=NC)/len(y)*100).round(1).tolist()}")

    # confusion matrix of the v5-biased final blend (shows class-2 bottleneck)
    cm = confusion_matrix(y, p_v5, labels=list(range(NC)))
    cmn = cm / cm.sum(1, keepdims=True)
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for a in range(NC):
        for b in range(NC):
            plt.text(b, a, f"{cmn[a,b]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if cmn[a, b] > 0.5 else "black")
    plt.xticks(range(NC), [f"c{i}" for i in range(NC)]); plt.yticks(range(NC), [f"c{i}" for i in range(NC)])
    plt.xlabel("predicted"); plt.ylabel("true")
    plt.title("Final blend OOF confusion (row-normalized)\nclass-2 leaks into class-1")
    plt.colorbar(fraction=0.046); plt.tight_layout()
    plt.savefig(FIG / "confusion_matrix.png", dpi=130); plt.close()
    print(f"  [class-2 recall split] of true c2: {cmn[2,2]*100:.0f}% correct, {cmn[2,1]*100:.0f}% -> c1")


def main():
    y = np.load(CACHE / "gbdt_richest.npz")["y_oof"]
    fold = np.load(CACHE / "folds.npz")["fold_of"]
    # fast, cache-only studies first (no training) ----
    O = load_oof()
    print("single-model OOF macro-F1:")
    for n in MEMBERS:
        print(f"  {n:12s} {macro(y, O[n].argmax(1)):.4f}   per-class={perclass(y, O[n].argmax(1))}")
    gru, _ = ensemble_ablation(y, O)
    decision_ablation(y, fold, gru)
    # feature progression last (trains LightGBM) ----
    feature_progression(y, fold)
    print("\nfigures written to", FIG)


if __name__ == "__main__":
    sys.exit(main())
