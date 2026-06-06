"""Extract TCN *softmax-head* OOF + test probabilities from trained checkpoints.

Tier-1 win: the KNN-on-embedding step discards signal that the softmax head keeps
(notably on the bottleneck classes 2/3/5). This module loads the per-fold, per-seed
checkpoints, runs the classification head (NOT the KNN), averages the seeds, and
saves probabilities in the same layout the ensemble consumes from gbdt/xgb caches.

OOF construction: for each fold F, the val windows are X_train[fold_of == F]; we
predict them with the fold-F checkpoint(s) (so every train window is predicted by a
model that never saw it). Test windows are predicted by every fold×seed checkpoint
and averaged.

Because the checkpoints were trained with logit-adjusted CE (logits shifted by
τ·log π at train time), the saved head logits are already prior-balanced, so a plain
softmax is the correct read-off (no extra adjustment).

Output: cache/head{TAG}.npz with keys oof, test, y_oof, file_id_test, oof_macro,
oof_per_class — mirrors cache/gbdt_richer.npz so src/ensemble.py can load it directly.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ._progress import tqdm
from .cv import build as build_folds
from .data import CACHE_DIR, build as build_windows
from .tcn import CHECKPOINT_TEMPLATE, DEVICE, HARDataset, TCN

NUM_CLASSES = 6


@torch.no_grad()
def _softmax_probs(model: TCN, X: np.ndarray, mean: np.ndarray, std: np.ndarray,
                   batch_size: int, num_workers: int, desc: str) -> np.ndarray:
    """Run the classification head on X; return (N, NUM_CLASSES) softmax probs."""
    ds = HARDataset(X, np.zeros(len(X), dtype=np.int64), mean, std, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    chunks: list[np.ndarray] = []
    for xb, _ in tqdm(loader, desc=desc, leave=False, unit="batch"):
        xb = xb.to(DEVICE, non_blocking=True)
        logits, _ = model(xb)
        chunks.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _load_model(fold: int, tag: str) -> tuple[TCN, np.ndarray, np.ndarray]:
    ckpt_path = CACHE_DIR / CHECKPOINT_TEMPLATE.format(fold=fold, tag=tag)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = TCN(channels=tuple(cfg["channels"]), dilations=tuple(cfg["dilations"]),
                embed_dim=cfg["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["mean"].astype(np.float32), ckpt["std"].astype(np.float32)


def build(tags: list[str], folds: list[int], batch_size: int = 256, num_workers: int = 2) -> dict:
    arrays = build_windows()
    fold_of = build_folds()["fold_of"]
    X_train = arrays["X_train"]
    y_train = arrays["y_train"]
    X_test = arrays["X_test"]

    oof = np.zeros((len(y_train), NUM_CLASSES), dtype=np.float64)
    test_acc = np.zeros((len(X_test), NUM_CLASSES), dtype=np.float64)
    n_test_models = 0

    for fold in folds:
        va_idx = np.flatnonzero(fold_of == fold)
        val_seed_probs = np.zeros((len(va_idx), NUM_CLASSES), dtype=np.float64)
        for tag in tags:
            model, mean, std = _load_model(fold, tag)
            val_seed_probs += _softmax_probs(model, X_train[va_idx], mean, std,
                                             batch_size, num_workers, f"f{fold}{tag} val")
            test_acc += _softmax_probs(model, X_test, mean, std,
                                       batch_size, num_workers, f"f{fold}{tag} test")
            n_test_models += 1
        oof[va_idx] = val_seed_probs / len(tags)  # average seeds for this fold's val

    test = (test_acc / n_test_models).astype(np.float32)
    oof = oof.astype(np.float32)

    macro = float(f1_score(y_train, oof.argmax(1), average="macro",
                           labels=list(range(NUM_CLASSES)), zero_division=0))
    per_class = f1_score(y_train, oof.argmax(1), average=None,
                         labels=list(range(NUM_CLASSES)), zero_division=0).astype(np.float32)
    return dict(oof=oof, test=test, y_oof=y_train.astype(np.int64),
                file_id_test=arrays["file_id_test"].astype(np.int64),
                oof_macro=np.float32(macro), oof_per_class=per_class)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["_la05", "_la05_s7", "_la05_s2024"],
                        help="Checkpoint seed tags to average (default: 3-seed la05 family).")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out-tag", default="_la05",
                        help="Suffix for cache/head{OUT_TAG}.npz (default: _la05).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    out = build(args.tags, args.folds, args.batch_size, args.num_workers)
    out_path = CACHE_DIR / f"head{args.out_tag}.npz"
    np.savez_compressed(out_path, **out)
    print(f"\nsaved → {out_path}")
    print(f"  seeds averaged   : {args.tags}")
    print(f"  OOF macroF1      : {float(out['oof_macro']):.4f}")
    print(f"  OOF per-class F1 : {[round(float(v), 3) for v in out['oof_per_class']]}")
    print(f"  oof {out['oof'].shape}  test {out['test'].shape}")
    print(f"  test pred shares : {np.bincount(out['test'].argmax(1), minlength=NUM_CLASSES) / len(out['test'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
