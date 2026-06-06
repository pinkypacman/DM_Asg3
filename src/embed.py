"""Extract 128-d embeddings from a trained TCN checkpoint.

Loads cache/tcn_fold{F}{TAG}.pt, runs the encoder over the train-fold-train,
train-fold-val, and full test sets (no augmentation), and saves the penultimate
embeddings (one row per window) to cache/tcn_embed{TAG}.npz alongside the
labels and file_ids needed for the embedding-inspection notebook.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from ._progress import tqdm
from .cv import build as build_folds
from .data import CACHE_DIR, build as build_windows
from .tcn import CHECKPOINT_TEMPLATE, DEVICE, HARDataset, TCN

EMBED_TEMPLATE = "tcn_embed_fold{fold}{tag}.npz"


@torch.no_grad()
def embed_array(model: TCN, X: np.ndarray, mean: np.ndarray, std: np.ndarray,
                batch_size: int, num_workers: int, desc: str) -> np.ndarray:
    """Run the encoder on X; return (N, embed_dim) float32."""
    ds = HARDataset(X, np.zeros(len(X), dtype=np.int64), mean, std, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    chunks: list[np.ndarray] = []
    for xb, _ in tqdm(loader, desc=desc, leave=False, unit="batch"):
        xb = xb.to(DEVICE, non_blocking=True)
        z = model.embed(xb)
        chunks.append(z.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--tag", type=str, default="", help="Suffix used at training time.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    ckpt_path = CACHE_DIR / CHECKPOINT_TEMPLATE.format(fold=args.fold, tag=args.tag)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first with: "
            f"python -m src.tcn --fold {args.fold} --tag '{args.tag}'"
        )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(
        f"loaded {ckpt_path.name} | best epoch {ckpt['epoch']} | "
        f"best val macroF1 {ckpt.get('best_macro_f1', float('nan')):.4f}"
    )

    cfg = ckpt["config"]
    model = TCN(
        channels=tuple(cfg["channels"]),
        dilations=tuple(cfg["dilations"]),
        embed_dim=cfg["embed_dim"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    mean = ckpt["mean"].astype(np.float32)
    std = ckpt["std"].astype(np.float32)

    arrays = build_windows()
    fold_of = build_folds()["fold_of"]
    fold = args.fold

    val_mask = fold_of == fold
    tr_idx = np.flatnonzero(~val_mask)
    va_idx = np.flatnonzero(val_mask)
    X_all = arrays["X_train"]
    y_all = arrays["y_train"]
    fid_all = arrays["file_id_train"]
    u_all = arrays["user_train"]

    Z_train = embed_array(model, X_all[tr_idx], mean, std, args.batch_size, args.num_workers, "train")
    Z_val   = embed_array(model, X_all[va_idx], mean, std, args.batch_size, args.num_workers, "val  ")
    Z_test  = embed_array(model, arrays["X_test"], mean, std, args.batch_size, args.num_workers, "test ")

    out_path = CACHE_DIR / EMBED_TEMPLATE.format(fold=fold, tag=args.tag)
    np.savez_compressed(
        out_path,
        Z_train=Z_train, Z_val=Z_val, Z_test=Z_test,
        y_train=y_all[tr_idx], y_val=y_all[va_idx],
        file_id_train=fid_all[tr_idx], file_id_val=fid_all[va_idx],
        user_train=u_all[tr_idx], user_val=u_all[va_idx],
        file_id_test=arrays["file_id_test"], user_test=arrays["user_test"],
        fold=np.int8(fold),
        ckpt_basename=np.array(ckpt_path.name),
        best_macro_f1=np.float32(ckpt.get("best_macro_f1", np.nan)),
    )

    # Sanity checks (matches plan §Verification step 7).
    with np.load(out_path) as f:
        for k in ("Z_train", "Z_val", "Z_test"):
            assert np.isfinite(f[k]).all(), f"non-finite values in {k}"
        assert f["Z_train"].shape == (len(tr_idx), cfg["embed_dim"])
        assert f["Z_val"].shape == (len(va_idx), cfg["embed_dim"])
        assert f["Z_test"].shape == (len(arrays["X_test"]), cfg["embed_dim"])
        assert set(f["file_id_train"].tolist()) | set(f["file_id_val"].tolist()) == set(fid_all.tolist())
        assert set(f["file_id_test"].tolist()) == set(arrays["file_id_test"].tolist())

    print(f"saved → {out_path}")
    print(f"  Z_train {Z_train.shape}  Z_val {Z_val.shape}  Z_test {Z_test.shape}")
    print(f"  embed_dim {cfg['embed_dim']}, all finite, file_ids aligned with cache/windows.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
