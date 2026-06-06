"""Generate a submission CSV via KNN-on-embedding, optionally ensembled across folds.

For each fold in --folds:
    - Load cache/tcn_embed_fold{F}{TAG}.npz
    - Concatenate fold-train + fold-val embeddings (all labeled training data)
    - Fit KNN(--k, --metric) on the standardized embedding
    - Predict per-class probability for each of the 6,849 test windows
Average probabilities across folds → argmax → label.

Writes submission/{NAME or auto-derived}.csv in sample_submission.csv row order.

Examples:
    python -m src.submit                            # single fold 0
    python -m src.submit --folds 0 1 2 3 4 --name submission_tcn_v2.csv
    python -m src.submit --folds 0 2 --k 25 --metric cosine
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
SAMPLE_PATH = PROJECT_ROOT / "sample_submission.csv"
NUM_CLASSES = 6


def _embed_path(fold: int, tag: str) -> Path:
    return CACHE_DIR / f"tcn_embed_fold{fold}{tag}.npz"


def _predict_proba_fold(fold: int, tag: str, k: int, metric: str, weights: str):
    path = _embed_path(fold, tag)
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {path}. "
            f"Run: python -m src.embed --fold {fold} --tag '{tag}'"
        )
    emb = np.load(path)
    Z_db = np.concatenate([emb["Z_train"], emb["Z_val"]], axis=0)
    y_db = np.concatenate([emb["y_train"], emb["y_val"]], axis=0)
    Z_test = emb["Z_test"]
    file_id_test = emb["file_id_test"]
    val_macro = float(emb["best_macro_f1"]) if "best_macro_f1" in emb.files else float("nan")

    scaler = StandardScaler().fit(Z_db)
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
    knn.fit(scaler.transform(Z_db), y_db)
    raw = knn.predict_proba(scaler.transform(Z_test))  # (n_test, n_classes_present)

    # Pad to a fixed (n_test, NUM_CLASSES) layout — KNN's classes_ may skip a missing class.
    proba = np.zeros((len(Z_test), NUM_CLASSES), dtype=np.float32)
    for i, c in enumerate(knn.classes_):
        proba[:, int(c)] = raw[:, i]

    return proba, file_id_test, val_macro, int(Z_db.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0],
                        help="Fold indices to ensemble (default: just 0).")
    parser.add_argument("--tag", default="", help="Optional tag suffix used at embed time.")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--weights", default="uniform", choices=["uniform", "distance"],
                        help="KNN voting weights.")
    parser.add_argument("--name", default="", help="Submission filename (default: auto-derived).")
    args = parser.parse_args()

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    proba_per_fold: list[np.ndarray] = []
    file_id_ref: np.ndarray | None = None
    macro_per_fold: list[float] = []
    db_sizes: list[int] = []

    for f in args.folds:
        proba, fid, val_macro, db_n = _predict_proba_fold(f, args.tag, args.k, args.metric, args.weights)
        proba_per_fold.append(proba)
        macro_per_fold.append(val_macro)
        db_sizes.append(db_n)
        if file_id_ref is None:
            file_id_ref = fid
        else:
            assert np.array_equal(file_id_ref, fid), f"file_id_test mismatch between folds (fold {f})"
        print(f"  fold {f}: db={db_n}  val-macroF1={val_macro:.4f}")

    avg_proba = np.mean(np.stack(proba_per_fold, axis=0), axis=0)
    pred = avg_proba.argmax(axis=1).astype(int)

    print()
    print(f"folds              : {args.folds}")
    print(f"per-fold val macroF1: {[round(v, 4) for v in macro_per_fold]}")
    print(f"mean val macroF1    : {np.mean(macro_per_fold):.4f}")
    print(f"KNN: k={args.k}, metric={args.metric}, weights={args.weights}")
    print(f"predicted class counts: {np.bincount(pred, minlength=NUM_CLASSES).tolist()}")

    sample = pd.read_csv(SAMPLE_PATH)
    id_to_pred = dict(zip(file_id_ref.astype(int).tolist(), pred.tolist()))
    missing = [int(i) for i in sample["Id"] if int(i) not in id_to_pred]
    if missing:
        raise ValueError(f"{len(missing)} sample Ids missing from prediction (first 5): {missing[:5]}")

    if args.name:
        out_name = args.name
    else:
        folds_label = "_".join(str(f) for f in args.folds)
        out_name = f"submission_tcn_folds{folds_label}{args.tag}.csv"
    out_path = SUBMISSION_DIR / out_name
    sample = sample.copy()
    sample["Label"] = sample["Id"].astype(int).map(id_to_pred).astype(int)
    sample.to_csv(out_path, index=False)

    written = pd.read_csv(out_path)
    assert len(written) == len(sample) == 6849
    assert set(written.columns) == {"Id", "Label"}
    assert set(written["Label"].unique()).issubset(set(range(NUM_CLASSES)))
    assert written["Id"].tolist() == sample["Id"].tolist()

    print()
    print(f"wrote → {out_path}")
    print(f"  rows               : {len(written)}")
    print(f"  label distribution : {written['Label'].value_counts().sort_index().to_dict()}")
    print(f"  share (%)          : "
          f"{(written['Label'].value_counts(normalize=True).sort_index() * 100).round(2).to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
