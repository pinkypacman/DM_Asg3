# DM Assignment 3 — Human Activity Recognition

Classify each independent 5-minute accelerometer window into an activity label
(0–5) from 1 Hz-aggregated `mean_x/y/z`, `std_x/y/z`. Metric: **macro F1**.
Final Kaggle submissions: `submission/submission_tcn_gru.csv` (public 0.8160) and
`submission/submission_tcn_robust_v5dist.csv` (robust/private bet, public 0.8115).

---

# ▶ Execution commands

```bash
# 0. environment (Python 3.10)
pip install -r requirements.txt
```

## A) Reproduce the exact submissions  (no GPU, ~seconds, deterministic)

```bash
python reproduce/build_finals.py
# → submission/submission_tcn_gru.csv
# → submission/submission_tcn_robust_v5dist.csv
```

Regenerates **both final CSVs bit-for-bit** from the committed prediction caches in
`cache/*.npz` (verified: identical SHA-256 across repeated runs). This is the
authoritative path for the graded Kaggle results — needs only `numpy` + `pandas`.

Optionally reproduce the report's figures/numbers (needs the dataset + caches rebuilt,
see B; figures are also committed under `analysis/figures/`):

```bash
python analysis/eda.py          # Q1 class balance, naive baselines, signals
python analysis/eda_deep.py     # Q1 t-SNE, motion/orientation, class-2 bottleneck
python analysis/ablation.py     # Q2/Q4 feature, ensemble, decision-layer ablations
```

## B) Train everything from scratch  (rebuild all caches → finals)

> Needs the raw dataset in `train/train/User_*/*.csv` and `test/test/User_*/*.csv`
> (restore from Kaggle) and a **GPU** for the deep models. The deep models are
> nondeterministic, so a full retrain yields an *equivalent* — not bit-identical —
> model; for the exact graded submissions use path **A**.

```bash
# 1. data → windows, user-grouped folds, and the feature sets
python -m src.data                              # → cache/windows.npz
python -m src.cv                                # → cache/folds.npz
python -m src.features_richest                  # → cache/features_richer.npz (172) + features_richest.npz (196)
# (optional, only for the report's feature-ablation) python -m src.features --rich

# 2. tabular members (CPU)
python -m src.gbdt --features-set richest --tag _richest   # → cache/gbdt_richest.npz
python -m src.xgb  --features-set richest --tag _richest   # → cache/xgb_richest.npz
python -m src.catboost_model                               # → cache/_catboost_oof.npz  (uses features_richer)

# 3. diverse families (CPU: bagging, linear ; GPU: mlp, inception, tabular)
python -m src.div_bagging        # → cache/div_bagging.npz
python -m src.div_linear         # → cache/div_linear.npz
python -m src.div_mlp            # → cache/div_mlp.npz
python -m src.div_inception      # → cache/div_inception.npz
python -m src.div_tabular        # → cache/div_tabular.npz   (TabNet)

# 4. deep sequence models (GPU)
python -m src.div_resnet                          # → cache/div_resnet.npz
python -m src.div_transformer                     # → cache/div_transformer.npz
python -m src.bigru                               # → cache/bigru.npz

# 5. TCN encoder → KNN-on-embedding member (GPU). 3 seeds × 5 folds, logit-adjusted (tau=0.5):
for st in "42:_la05" "7:_la05_s7" "2024:_la05_s2024"; do
  seed=${st%%:*}; tag=${st##*:}
  for f in 0 1 2 3 4; do
    python -m src.tcn   --fold "$f" --loss logit-adjusted --la-tau 0.5 --seed "$seed" --tag "$tag"
    python -m src.embed --fold "$f" --tag "$tag"
  done
done
python -m src.postprocess --tag _la05 --seed-tags _la05_s7 _la05_s2024   # → cache/postprocess_tcn_la05.npz

# 6. assemble the two final submissions from the rebuilt caches
python reproduce/build_finals.py
```

The 12 caches produced by steps 2–5 are exactly the members consumed by
`reproduce/build_finals.py`:

```
postprocess_tcn_la05  gbdt_richest  xgb_richest  _catboost_oof
div_bagging  div_mlp  div_linear  div_inception  div_tabular
div_resnet  div_transformer  bigru
```

---

## The model, in one paragraph

One **12-member diverse ensemble** (TCN-KNN + LightGBM + XGBoost + CatBoost +
bagging/MLP/linear/Inception/TabNet + 1D-ResNet/Transformer/BiGRU) blended with
frozen weights, then a per-final **decision layer**: `gru` pins the class-2 / class-5
predicted shares (3.3% / 3.6%); `robust_v5dist` prior-matches to v5's distribution.
The tabular members use the **196-d "richest"** features (172-d `richer` + 24 jerk/RMS
motion features); CatBoost uses 172-d `richer`; the deep models read the raw 300×6
windows. Full analysis and ablations are in **[REPORT.md](REPORT.md)**.

| File | Strategy | Public LB |
|------|----------|-----------|
| `submission/submission_tcn_gru.csv` | public-tuned rare-class calibration | 0.8160 |
| `submission/submission_tcn_robust_v5dist.csv` | distribution-robust (private bet) | 0.8115 |

## What's in the repo (git)

`.gitignore` keeps the repo lean (~11 MB) while still reproducing the finals.
**Tracked:** `src/`, `reproduce/`, `analysis/*.py` + `analysis/figures/*.png`,
`README.md`, `requirements.txt`, `sample_submission.csv`, and the small prediction
caches in `cache/*.npz` that `reproduce/build_finals.py` consumes. **Ignored:** the
raw dataset (`train/`, `test/`, `Dataset/`), logs, generated submission CSVs,
`archive/`, and bulky/regenerable caches (`windows.npz`, `features*.npz`, `*.pt`).

## Directory layout

```
README.md / REPORT.md         this file / the written report (+ analysis/figures/)
requirements.txt              pinned dependencies
sample_submission.csv         submission template (Id, Label)
train/ test/ Dataset/         raw per-user accelerometer windows (git-ignored)
src/                          pipeline: data, cv, features(+richest), gbdt, xgb,
                              catboost_model, tcn, embed, postprocess, bigru, div_* models
cache/                        precomputed OOF/test probabilities (+ ignored big caches)
reproduce/build_finals.py     rebuilds the two finals from cache/ (verified bit-exact)
analysis/                     eda.py, eda_deep.py, ablation.py + figures/
submission/                   the two FINAL submissions (regenerated by build_finals)
archive/                      old experiments, submissions, logs, notebooks, run scripts
```
