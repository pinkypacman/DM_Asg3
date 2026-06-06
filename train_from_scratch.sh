#!/usr/bin/env bash
# =============================================================================
# train_from_scratch.sh — rebuild ALL model caches from the raw dataset and
# assemble the two final submissions.
#
#   Requires: the raw data in train/train/User_*/*.csv and test/test/User_*/*.csv
#             (restore from Kaggle) and a GPU for the deep models.
#   Note:     the deep models are nondeterministic, so a full retrain yields an
#             *equivalent* — not bit-identical — model. For the exact graded
#             submissions, use:  python reproduce/build_finals.py
#
#   Usage:    ./train_from_scratch.sh
#   Override: PYTHON=/path/to/python CUDA_VISIBLE_DEVICES=0 ./train_from_scratch.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"                      # run from the project root
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

step() { echo; echo "============================================================"; echo ">> $*"; echo "============================================================"; }

# ---- 1. data, user-grouped folds, feature sets ------------------------------
step "1/6  data + folds + features"
"$PYTHON" -m src.data                                      # -> cache/windows.npz
"$PYTHON" -m src.cv                                        # -> cache/folds.npz
"$PYTHON" -m src.features_richest                          # -> features_richer.npz (172) + features_richest.npz (196)

# ---- 2. tabular members (CPU) ----------------------------------------------
step "2/6  tabular members (LightGBM / XGBoost / CatBoost)"
"$PYTHON" -m src.gbdt --features-set richest --tag _richest   # -> cache/gbdt_richest.npz
"$PYTHON" -m src.xgb  --features-set richest --tag _richest   # -> cache/xgb_richest.npz
"$PYTHON" -m src.catboost_model                               # -> cache/_catboost_oof.npz

# ---- 3. diverse families (bagging/linear = CPU ; mlp/inception/tabular = GPU)-
step "3/6  diverse families"
"$PYTHON" -m src.div_bagging                              # -> cache/div_bagging.npz
"$PYTHON" -m src.div_linear                              # -> cache/div_linear.npz
"$PYTHON" -m src.div_mlp                                 # -> cache/div_mlp.npz
"$PYTHON" -m src.div_inception                           # -> cache/div_inception.npz
"$PYTHON" -m src.div_tabular                             # -> cache/div_tabular.npz   (TabNet)

# ---- 4. deep sequence models (GPU) -----------------------------------------
step "4/6  deep sequence models (ResNet / Transformer / BiGRU)"
"$PYTHON" -m src.div_resnet                              # -> cache/div_resnet.npz
"$PYTHON" -m src.div_transformer                         # -> cache/div_transformer.npz
"$PYTHON" -m src.bigru                                   # -> cache/bigru.npz

# ---- 5. TCN encoder -> KNN-on-embedding (GPU): 3 seeds x 5 folds, logit-adj --
step "5/6  TCN (3 seeds x 5 folds, logit-adjusted tau=0.5) + KNN postprocess"
for st in "42:_la05" "7:_la05_s7" "2024:_la05_s2024"; do
  seed="${st%%:*}"; tag="${st##*:}"
  for f in 0 1 2 3 4; do
    "$PYTHON" -m src.tcn   --fold "$f" --loss logit-adjusted --la-tau 0.5 --seed "$seed" --tag "$tag"
    "$PYTHON" -m src.embed --fold "$f" --tag "$tag"
  done
done
"$PYTHON" -m src.postprocess --tag _la05 --seed-tags _la05_s7 _la05_s2024   # -> cache/postprocess_tcn_la05.npz

# ---- 6. assemble the two final submissions ---------------------------------
step "6/6  build final submissions"
"$PYTHON" reproduce/build_finals.py

echo; echo "DONE — submissions written to submission/:"
echo "  submission/submission_tcn_gru.csv"
echo "  submission/submission_tcn_robust_v5dist.csv"
