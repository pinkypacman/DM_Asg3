"""CatBoost member of the final blend — on the 172-d `features_richer` set under
GroupKFold(user). Adds boosting-framework diversity vs LightGBM/XGBoost.

Output: cache/_catboost_oof.npz  {oof, test, y}

Run:  python -m src.catboost_model        (after src.features --richer)
"""
from __future__ import annotations

import sys
import time

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

from .data import CACHE_DIR
from .features import build_richer
from .cv import build as build_folds

NC = 6


def main() -> int:
    t0 = time.time()
    fr = build_richer(force=False)
    F = np.nan_to_num(fr["F_train"].astype(np.float32))
    Ft = np.nan_to_num(fr["F_test"].astype(np.float32))
    y = fr["y_train"]
    fold_of = build_folds(force=False)["fold_of"]

    cnt = np.bincount(y, minlength=NC)
    cw = (len(y) / (NC * cnt)).tolist()          # balanced-style class weights
    oof = np.zeros((len(y), NC), dtype=np.float32)
    test = np.zeros((len(Ft), NC), dtype=np.float64)
    for k in range(5):
        tr, va = fold_of != k, fold_of == k
        m = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            loss_function="MultiClass", class_weights=cw, random_seed=42,
            eval_metric="TotalF1", early_stopping_rounds=80, thread_count=-1,
            verbose=False,
        )
        m.fit(F[tr], y[tr], eval_set=(F[va], y[va]), use_best_model=True)
        oof[va] = m.predict_proba(F[va])
        test += m.predict_proba(Ft)
        print(f"  fold{k} best_iter={m.get_best_iteration()}")
    test /= 5

    macro = f1_score(y, oof.argmax(1), average="macro", zero_division=0)
    print(f"CatBoost-richer OOF macroF1={macro:.4f}  "
          f"per-class={np.round(f1_score(y, oof.argmax(1), labels=list(range(NC)), average=None, zero_division=0), 3).tolist()}")
    np.savez(CACHE_DIR / "_catboost_oof.npz", oof=oof, test=test.astype(np.float32), y=y)
    print(f"saved {CACHE_DIR/'_catboost_oof.npz'}   total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
