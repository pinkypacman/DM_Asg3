"""Deeper EDA + visual explanations for the report.

Produces (under analysis/figures/):
    motion_orientation.png   physical descriptors (motion intensity, orientation)
                             per class — WHY classes separate (or don't)
    embedding_tsne.png        t-SNE of the 196-d feature space: global structure +
                             a c1-vs-c2 overlay that shows the bottleneck visually
    c2_vs_c1_separability.png best single feature's c1/c2 histogram overlap + the
                             binary c2-vs-c1 AUC ceiling
    per_user_activities.png   how many distinct activities each user performs
    feature_importance.png    top features by mutual information with the label

Everything is computed from cache/windows.npz (+ cache/features_richest.npz) and
printed numbers back the explanations in REPORT.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NC = 6
COLORS = cm.tab10(np.arange(NC))
RNG = np.random.default_rng(0)


def physical_descriptors(X):
    """Interpretable per-window descriptors straight from the raw channels."""
    mean_mag = np.sqrt((X[..., 0:3] ** 2).sum(2))       # |mean accel| per second
    jitter = np.sqrt((X[..., 3:6] ** 2).sum(2))         # |std| per second = motion/jitter
    motion = jitter.mean(1)                              # avg motion intensity
    mean_mag_avg = mean_mag.mean(1)
    orient = X[..., 2].mean(1) / np.clip(mean_mag_avg, 1e-8, None)  # mean_z / |mean| (posture)
    return motion, orient


def fig_motion_orientation(X, y):
    motion, orient = physical_descriptors(X)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, title, ylab in [
        (axes[0], motion, "Motion intensity (mean |std| per window)", "motion intensity"),
        (axes[1], orient, "Orientation (mean_z / |mean accel|)", "orientation ratio"),
    ]:
        data = [vals[y == c] for c in range(NC)]
        bp = ax.boxplot(data, tick_labels=[f"c{c}" for c in range(NC)], showfliers=False,
                        patch_artist=True, medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], range(NC)):
            patch.set_facecolor(COLORS[c]); patch.set_alpha(0.7)
        ax.set_title(title); ax.set_ylabel(ylab); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Physical descriptors per class — c0/c1 are static & oriented; c2 sits between")
    plt.tight_layout(); plt.savefig(FIG / "motion_orientation.png", dpi=130); plt.close()
    # print class medians for the report
    print("class : median motion | median orientation")
    for c in range(NC):
        print(f"  c{c}  : {np.median(motion[y==c]):.4f}        | {np.median(orient[y==c]):+.3f}")
    return motion, orient


def fig_embedding(F, y):
    """PCA(50) -> t-SNE on a stratified subsample (rare classes kept in full)."""
    idx = []
    for c in range(NC):
        ci = np.flatnonzero(y == c)
        take = ci if len(ci) <= 1500 else RNG.choice(ci, 1500, replace=False)
        idx.append(take)
    idx = np.concatenate(idx); RNG.shuffle(idx)
    Fs = StandardScaler().fit_transform(F[idx])
    Z = PCA(n_components=50, random_state=0).fit_transform(Fs)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0,
               learning_rate="auto").fit_transform(Z)
    ys = y[idx]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    # (a) all classes
    for c in range(NC):
        m = ys == c
        axes[0].scatter(emb[m, 0], emb[m, 1], s=6, color=COLORS[c], label=f"c{c}", alpha=0.6)
    axes[0].legend(markerscale=2, fontsize=8); axes[0].set_title("t-SNE of 196-d features (all classes)")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    # (b) c1 vs c2 only — the bottleneck
    m1, m2 = ys == 1, ys == 2
    axes[1].scatter(emb[m1, 0], emb[m1, 1], s=6, color="#bbbbbb", label="c1", alpha=0.5)
    axes[1].scatter(emb[m2, 0], emb[m2, 1], s=14, color="#C44E52", label="c2", alpha=0.9,
                    edgecolors="k", linewidths=0.2)
    axes[1].legend(markerscale=2); axes[1].set_title("class-2 (red) is buried inside class-1 (grey)")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.tight_layout(); plt.savefig(FIG / "embedding_tsne.png", dpi=130); plt.close()


def fig_c2_vs_c1(F, y, names, user):
    """Most-discriminative single feature for c1-vs-c2 + the binary AUC ceiling."""
    m = (y == 1) | (y == 2)
    Fb, yb = F[m], (y[m] == 2).astype(int)
    mi = mutual_info_classif(StandardScaler().fit_transform(Fb), yb, random_state=0)
    j = int(mi.argmax()); fname = names[j]
    # binary AUC ceiling via 5-fold GroupKFold LightGBM
    ub = user[m]
    from sklearn.model_selection import GroupKFold
    aucs = []
    for tr, va in GroupKFold(5).split(Fb, yb, ub):
        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                 class_weight="balanced", verbose=-1, n_jobs=-1)
        clf.fit(Fb[tr], yb[tr])
        aucs.append(roc_auc_score(yb[va], clf.predict_proba(Fb[va])[:, 1]))
    auc = float(np.mean(aucs))
    plt.figure(figsize=(6.4, 4))
    v1, v2 = Fb[yb == 0, j], Fb[yb == 1, j]
    lo, hi = np.percentile(np.r_[v1, v2], [1, 99])
    bins = np.linspace(lo, hi, 40)
    plt.hist(v1, bins=bins, density=True, alpha=0.6, color="#bbbbbb", label="class 1")
    plt.hist(v2, bins=bins, density=True, alpha=0.7, color="#C44E52", label="class 2")
    plt.xlabel(f"most discriminative feature: {fname}"); plt.ylabel("density")
    plt.title(f"c1 vs c2 overlap — even the best feature can't separate them\n"
              f"binary c2-vs-c1 AUC ceiling = {auc:.3f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "c2_vs_c1_separability.png", dpi=130); plt.close()
    print(f"\nc2-vs-c1: best MI feature = {fname} (MI={mi[j]:.3f}); binary AUC = {auc:.3f}")
    return auc


def fig_per_user(y, user):
    counts = [len(np.unique(y[user == u])) for u in np.unique(user)]
    plt.figure(figsize=(6, 3.6))
    plt.hist(counts, bins=np.arange(0.5, 7.5, 1), color="#4C72B0", rwidth=0.85)
    plt.xlabel("# distinct activities performed by a user"); plt.ylabel("# users")
    plt.title(f"Activities per user (mean {np.mean(counts):.2f}) — "
              f"not all users do all 6")
    plt.tight_layout(); plt.savefig(FIG / "per_user_activities.png", dpi=130); plt.close()
    print(f"\nactivities per user: mean={np.mean(counts):.2f} min={min(counts)} max={max(counts)}")


def fig_importance(F, y, names):
    sub = RNG.choice(len(y), min(5000, len(y)), replace=False)
    mi = mutual_info_classif(StandardScaler().fit_transform(F[sub]), y[sub], random_state=0)
    order = np.argsort(mi)[::-1][:15][::-1]
    plt.figure(figsize=(7, 4.5))
    plt.barh(range(len(order)), mi[order], color="#55A868")
    plt.yticks(range(len(order)), [names[i] for i in order], fontsize=8)
    plt.xlabel("mutual information with label"); plt.title("Top-15 most informative features")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=130); plt.close()
    print("\ntop-8 features by MI:", [names[i] for i in order[::-1][:8]])


def main():
    w = np.load(CACHE / "windows.npz", allow_pickle=True)
    X, y, user = w["X_train"], w["y_train"], w["user_train"]
    z = np.load(CACHE / "features_richest.npz", allow_pickle=True)
    F = z["F_train"].astype(np.float32)
    names = [str(s) for s in z["feature_names"]]
    print(f"X={X.shape}  F={F.shape}")
    fig_motion_orientation(X, y)
    fig_per_user(y, user)
    fig_importance(F, y, names)
    fig_c2_vs_c1(F, y, names, user)
    fig_embedding(F, y)           # slowest (t-SNE) last
    print("\nfigures written to", FIG)


if __name__ == "__main__":
    sys.exit(main())
