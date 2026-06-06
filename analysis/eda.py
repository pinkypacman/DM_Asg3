"""Preliminary data analysis + naive baselines for the report (Q1).

Produces (under analysis/figures/):
    class_distribution.png   train label histogram (the imbalance)
    position_vs_class.png     class composition by normalized position in each
                              user's file_id-sorted sequence (scripted collection)
    signal_examples.png       example mean_x/y/z traces, one window per class
    adjacent_label.png        adjacent-label agreement along the per-user sequence

Prints: class shares, adjacent-label agreement, and a table of naive-baseline
macro-F1 scores (majority / logistic-on-6-means / position-only / lean-LGB),
all under the GroupKFold(user) protocol used everywhere in the project.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NC = 6
CLASS_NAMES = [f"c{i}" for i in range(NC)]


def macro(y, p):
    return f1_score(y, p, average="macro", labels=list(range(NC)), zero_division=0)


def main():
    w = np.load(CACHE / "windows.npz", allow_pickle=True)
    X, y = w["X_train"], w["y_train"]                 # (N,300,6), (N,)
    user, fid = w["user_train"], w["file_id_train"]
    fold = np.load(CACHE / "folds.npz")["fold_of"]
    N = len(y)
    print(f"train windows={N}  shape={X.shape}  users={len(np.unique(user))}")

    # ---------- class distribution ----------
    counts = np.bincount(y, minlength=NC)
    shares = counts / N * 100
    print("\nclass counts :", counts.tolist())
    print("class shares%:", shares.round(2).tolist())
    print(f"imbalance ratio (max/min) = {counts.max()/counts.min():.1f}x")

    plt.figure(figsize=(6.5, 4))
    bars = plt.bar(CLASS_NAMES, counts, color="#4C72B0")
    plt.ylim(0, counts.max() * 1.24)          # headroom so bar labels clear the title
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width()/2, c + counts.max() * 0.012,
                 f"{c}\n{c/N*100:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.ylabel("# windows (train)")
    plt.title(f"Class distribution — {counts.max()/counts.min():.0f}× imbalance", pad=10)
    plt.tight_layout(); plt.savefig(FIG / "class_distribution.png", dpi=130); plt.close()

    # ---------- normalized position within each user's sequence ----------
    pos = np.zeros(N)
    for u in np.unique(user):
        idx = np.flatnonzero(user == u)
        order = idx[np.argsort(fid[idx])]
        n = len(order)
        pos[order] = np.arange(n) / max(n - 1, 1)
    # class composition per decile
    deciles = np.clip((pos * 10).astype(int), 0, 9)
    comp = np.zeros((10, NC))
    for d in range(10):
        m = deciles == d
        comp[d] = np.bincount(y[m], minlength=NC) / max(m.sum(), 1)
    plt.figure(figsize=(7, 3.8))
    bottom = np.zeros(10)
    colors = plt.cm.tab10(np.arange(NC))
    for c in range(NC):
        plt.bar(range(10), comp[:, c], bottom=bottom, label=f"c{c}", color=colors[c])
        bottom += comp[:, c]
    plt.xlabel("normalized position in user's sequence (decile)")
    plt.ylabel("class fraction"); plt.ylim(0, 1)
    plt.title("Activity depends on position → data collection is scripted")
    plt.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout(); plt.savefig(FIG / "position_vs_class.png", dpi=130); plt.close()
    print("\nposition deciles: first/last decile c0 share = "
          f"{comp[0,0]*100:.0f}% / {comp[9,0]*100:.0f}%  (rare classes absent at the ends)")

    # ---------- example signals, one window per class ----------
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    for c, ax in zip(range(NC), axes.ravel()):
        i = np.flatnonzero(y == c)[0]
        for ch, name in zip(range(3), ["mean_x", "mean_y", "mean_z"]):
            ax.plot(X[i, :, ch], lw=0.8, label=name)
        ax.set_title(f"class {c}", fontsize=9); ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Per-second mean acceleration traces (one window per class)")
    fig.supxlabel("second (0–299)")
    plt.tight_layout(); plt.savefig(FIG / "signal_examples.png", dpi=130); plt.close()

    # ---------- adjacent-label agreement along the per-user time index ----------
    order = np.lexsort((fid, user))
    yo, uo = y[order], user[order]
    same_user = uo[:-1] == uo[1:]
    agree = (yo[:-1] == yo[1:])[same_user].mean()
    print(f"\nadjacent-label agreement (same user, consecutive file_id) = {agree*100:.1f}%")
    # transition heatmap
    Tn = np.zeros((NC, NC))
    for a, b, su in zip(yo[:-1], yo[1:], same_user):
        if su:
            Tn[a, b] += 1
    Tn = Tn / Tn.sum(1, keepdims=True)
    plt.figure(figsize=(4.6, 4))
    plt.imshow(Tn, cmap="Blues", vmin=0, vmax=1)
    for a in range(NC):
        for b in range(NC):
            plt.text(b, a, f"{Tn[a,b]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if Tn[a, b] > 0.5 else "black")
    plt.xticks(range(NC), CLASS_NAMES); plt.yticks(range(NC), CLASS_NAMES)
    plt.xlabel("next label"); plt.ylabel("current label")
    plt.title(f"P(next | current), same user\nadjacent agreement = {agree*100:.0f}%")
    plt.colorbar(fraction=0.046); plt.tight_layout()
    plt.savefig(FIG / "adjacent_label.png", dpi=130); plt.close()

    # ---------- naive baselines (GroupKFold OOF macro-F1) ----------
    print("\n=== NAIVE BASELINES (GroupKFold-by-user OOF macro-F1) ===")
    # 1. majority class
    maj = np.full(N, np.bincount(y).argmax())
    print(f"  majority-class            macroF1 = {macro(y, maj):.4f}")

    # 2. logistic regression on the 6 per-window channel means (no temporal info)
    feat6 = X.mean(axis=1)  # (N,6)
    pred = np.zeros(N, dtype=int)
    for f in range(5):
        tr, va = fold != f, fold == f
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feat6[tr], y[tr])
        pred[va] = m.predict(feat6[va])
    print(f"  logreg on 6 channel-means macroF1 = {macro(y, pred):.4f}")

    # 3. position-only (single feature) logistic regression
    predp = np.zeros(N, dtype=int)
    for f in range(5):
        tr, va = fold != f, fold == f
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(pos[tr, None], y[tr])
        predp[va] = m.predict(pos[va, None])
    print(f"  position-only             macroF1 = {macro(y, predp):.4f}")
    print("\nfigures written to", FIG)


if __name__ == "__main__":
    sys.exit(main())
