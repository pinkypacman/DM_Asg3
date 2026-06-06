"""Non-causal Temporal Convolutional Network — encoder + supervised training.

Trains on a single GroupKFold(user) fold (default fold 0) for embedding
inspection. Records per-epoch train/val loss, per-class F1, and val macro F1.
Saves best checkpoint to cache/tcn_fold{f}.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ._progress import tqdm
from .cv import build as build_folds
from .data import CACHE_DIR, build as build_windows

CHECKPOINT_TEMPLATE = "tcn_fold{fold}{tag}.pt"
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss (Cao et al. 2019, arXiv:1906.07413).

    Enforces a larger classification margin for rarer classes: margin_c ∝ 1/n_c^(1/4),
    rescaled so the largest margin equals `max_m`. The true-class logit is reduced by its
    margin before a temperature-scaled (`s`) softmax CE. `self.weight` is set externally
    per-epoch to implement DRW (Deferred Re-Weighting): None early, class-balanced later.
    Directly targets the minority→majority confusion (here c2/c3/c5 → c1).
    """

    def __init__(self, cls_num_list, max_m: float = 0.5, s: float = 30.0):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(np.asarray(cls_num_list, dtype=np.float64)))
        m = m * (max_m / m.max())
        self.register_buffer("m_list", torch.tensor(m, dtype=torch.float32))
        self.s = float(s)
        self.weight = None  # (C,) tensor when DRW is active, else None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros_like(logits, dtype=torch.bool)
        idx.scatter_(1, target.view(-1, 1), True)
        batch_m = self.m_list[target].view(-1, 1)
        logits_m = logits - idx.float() * batch_m
        return F.cross_entropy(self.s * logits_m, target, weight=self.weight)


def class_balanced_weights(cls_num_list, beta: float = 0.9999) -> torch.Tensor:
    """Class-balanced reweighting (Cui et al. 2019): w_c ∝ (1-β)/(1-β^{n_c}), mean-normalized."""
    n = np.asarray(cls_num_list, dtype=np.float64)
    eff = 1.0 - np.power(beta, n)
    w = (1.0 - beta) / np.maximum(eff, 1e-12)
    return torch.tensor(w / w.sum() * len(n), dtype=torch.float32)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = dilation * (kernel - 1) // 2  # non-causal symmetric padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, padding_mode="reflect")
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation, padding_mode="reflect")
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.dropout(self.act(self.bn1(self.conv1(x))))
        h = self.dropout(self.act(self.bn2(self.conv2(h))))
        return h + res


class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        channels: tuple[int, ...] = (32, 64, 64, 64, 64, 128),
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        embed_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        block_dropout: float = 0.2,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        assert len(channels) == len(dilations)
        blocks: list[nn.Module] = []
        prev = in_channels
        for ch, d in zip(channels, dilations):
            blocks.append(TemporalBlock(prev, ch, kernel=3, dilation=d, dropout=block_dropout))
            prev = ch
        self.tcn = nn.Sequential(*blocks)
        self.proj = nn.Sequential(
            nn.Linear(prev * 2, embed_dim),  # *2 for mean+max concat
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.tcn(x)
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)  # (B, 2*C_last)
        return self.proj(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embed(x)
        return self.head(z), z


class HARDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool = False,
        jitter_std: float = 0.01,
        max_shift: int = 15,
        scale_jitter: float = 0.0,
        rng_seed: int = 0,
    ):
        self.X = X
        self.y = y
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.augment = augment
        self.jitter_std = float(jitter_std)
        self.max_shift = int(max_shift)
        self.scale_jitter = float(scale_jitter)
        self._rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        x = self.X[i].copy()  # (300, 6)
        if self.augment:
            if self.max_shift > 0:
                shift = int(self._rng.integers(-self.max_shift, self.max_shift + 1))
                if shift != 0:
                    x = np.roll(x, shift, axis=0)
            if self.scale_jitter > 0:
                # Multiplicative ±scale_jitter, independently for the 3 mean and 3 std channels.
                s = self._rng.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter, size=x.shape[-1]).astype(np.float32)
                x = x * s
            if self.jitter_std > 0:
                x = x + self._rng.normal(0.0, self.jitter_std, x.shape).astype(np.float32)
        x = (x - self.mean) / self.std
        x = np.ascontiguousarray(x.T)  # (6, 300)
        x_t = torch.from_numpy(x)
        if self.y is None:
            return x_t, torch.tensor(0, dtype=torch.long)
        return x_t, torch.tensor(int(self.y[i]), dtype=torch.long)


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fold_of: np.ndarray,
    fold: int,
    batch_size: int,
    use_sampler: bool,
    sampler_power: float,
    augment: bool,
    jitter_std: float,
    max_shift: int,
    scale_jitter: float,
    num_workers: int,
    seed: int,
):
    val_mask = fold_of == fold
    tr_idx = np.flatnonzero(~val_mask)
    va_idx = np.flatnonzero(val_mask)

    # Per-channel mean/std from train side only.
    flat = X_train[tr_idx].reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)

    train_ds = HARDataset(
        X_train[tr_idx], y_train[tr_idx], mean, std,
        augment=augment, jitter_std=jitter_std, max_shift=max_shift,
        scale_jitter=scale_jitter, rng_seed=seed,
    )
    val_ds = HARDataset(X_train[va_idx], y_train[va_idx], mean, std, augment=False, rng_seed=seed + 1)

    if use_sampler:
        counts = np.bincount(y_train[tr_idx], minlength=NUM_CLASSES).astype(np.float64)
        # power 0.5 = sqrt-inverse (mild), 1.0 = pure inverse-frequency (equalize classes).
        class_w = 1.0 / np.power(np.maximum(counts, 1.0), sampler_power)
        sample_w = class_w[y_train[tr_idx]]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(tr_idx),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, tr_idx, va_idx, mean.astype(np.float32), std.astype(np.float32)


@torch.no_grad()
def evaluate(model: TCN, loader: DataLoader, desc: str = "val") -> dict:
    model.eval()
    logits_all, y_all = [], []
    total_loss, total_n = 0.0, 0
    for xb, yb in tqdm(loader, desc=desc, leave=False, unit="batch"):
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += float(loss.item()); total_n += yb.numel()
        logits_all.append(logits.detach().cpu())
        y_all.append(yb.detach().cpu())
    logits_all = torch.cat(logits_all).numpy()
    y_all = torch.cat(y_all).numpy()
    preds = logits_all.argmax(axis=1)
    macro = float(f1_score(y_all, preds, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0))
    per_class = f1_score(y_all, preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0).tolist()
    return {"loss": total_loss / max(total_n, 1), "macro_f1": macro, "per_class_f1": per_class}


def train_one_fold(
    fold: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    use_sampler: bool,
    sampler_power: float,
    augment: bool,
    jitter_std: float,
    max_shift: int,
    scale_jitter: float,
    loss_kind: str,
    la_tau: float,
    num_workers: int,
    seed: int,
    tag: str,
    ldam_max_m: float = 0.5,
    ldam_s: float = 30.0,
    drw_start_frac: float = 0.6,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    arrays = build_windows()
    folds = build_folds()
    X_train = arrays["X_train"]
    y_train = arrays["y_train"]
    fold_of = folds["fold_of"]

    train_loader, val_loader, tr_idx, va_idx, mean, std = make_loaders(
        X_train, y_train, fold_of, fold, batch_size, use_sampler, sampler_power,
        augment, jitter_std, max_shift, scale_jitter, num_workers, seed,
    )
    print(
        f"fold {fold}: train n={len(tr_idx)} val n={len(va_idx)} "
        f"users(val)={len(np.unique(arrays['user_train'][va_idx]))}"
    )
    counts_tr = np.bincount(y_train[tr_idx], minlength=NUM_CLASSES).tolist()
    counts_va = np.bincount(y_train[va_idx], minlength=NUM_CLASSES).tolist()
    print(f"  train class counts: {counts_tr}")
    print(f"  val   class counts: {counts_va}")
    sampler_label = f"inv-freq^{sampler_power:g}" if use_sampler else "OFF"
    aug_label = (f"on (σ={jitter_std:g}, shift=±{max_shift}, scale=±{scale_jitter:g})"
                 if augment else "off")
    if loss_kind == "logit-adjusted":
        loss_label = f"logit-adjusted (τ={la_tau:g})"
    else:
        loss_label = "ce"
    print(f"  sampler={sampler_label}  augment={aug_label}  loss={loss_label}  device={DEVICE}")

    # Precompute log-prior for logit-adjusted CE (from train-fold class counts).
    if loss_kind == "logit-adjusted":
        counts_tr_arr = np.array(counts_tr, dtype=np.float64)
        priors = counts_tr_arr / counts_tr_arr.sum()
        log_prior_tensor = torch.from_numpy(np.log(priors + 1e-12).astype(np.float32)).to(DEVICE)
    else:
        log_prior_tensor = None

    # LDAM (+ optional DRW) setup. DRW is disabled when drw_start_frac >= 1.0 (margin-only);
    # on this c0/c1-dominated dataset, class-balanced DRW over-weights minorities and crushes
    # majority F1, so margin-only (with the proven sqrt-inverse sampler) is the gentler default.
    ldam_criterion = None
    drw_start_epoch = None
    cb_w = None
    if loss_kind == "ldam":
        ldam_criterion = LDAMLoss(counts_tr, max_m=ldam_max_m, s=ldam_s).to(DEVICE)
        print(f"  [ldam] max_m={ldam_max_m} s={ldam_s}  "
              f"margins={np.round(ldam_criterion.m_list.cpu().numpy(), 3).tolist()}")
        if drw_start_frac < 1.0:
            cb_w = class_balanced_weights(counts_tr).to(DEVICE)
            drw_start_epoch = max(1, int(drw_start_frac * epochs))
            print(f"  [ldam] DRW class-balanced reweight from epoch {drw_start_epoch} "
                  f"(weights={np.round(cb_w.cpu().numpy(), 2).tolist()})")
        else:
            print(f"  [ldam] DRW disabled (margin-only); sampler={'on' if use_sampler else 'off'}")

    model = TCN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best = {"macro_f1": -1.0, "epoch": -1, "per_class_f1": None}
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        # DRW: turn on class-balanced reweighting only in the deferred (late) phase.
        if ldam_criterion is not None and drw_start_epoch is not None:
            ldam_criterion.weight = cb_w if epoch >= drw_start_epoch else None
            if epoch == drw_start_epoch:
                no_improve = 0  # fresh early-stop window once DRW kicks in
        running_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"ep {epoch:3d} train", leave=False, unit="batch")
        for xb, yb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            if ldam_criterion is not None:
                loss = ldam_criterion(logits, yb)
            elif log_prior_tensor is not None:
                # Logit-adjusted CE (Menon et al. 2020): training-time logit shift.
                loss = F.cross_entropy(logits + la_tau * log_prior_tensor, yb)
            else:
                loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            bs = yb.numel()
            running_loss += float(loss.item()) * bs; n += bs
            pbar.set_postfix(loss=f"{running_loss / max(n, 1):.4f}")
        scheduler.step()
        train_loss = running_loss / max(n, 1)
        val = evaluate(model, val_loader, desc=f"ep {epoch:3d} val")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
            "val_macro_f1": val["macro_f1"], "val_per_class_f1": val["per_class_f1"],
            "lr": scheduler.get_last_lr()[0],
        })
        per_class_str = " ".join(f"{v:.3f}" for v in val["per_class_f1"])
        print(
            f"ep {epoch:3d} | tr_loss {train_loss:.4f} | va_loss {val['loss']:.4f} "
            f"| macroF1 {val['macro_f1']:.4f} | perclass [{per_class_str}] "
            f"| {time.time() - t0:.1f}s"
        )

        if val["macro_f1"] > best["macro_f1"]:
            best = {
                "macro_f1": val["macro_f1"],
                "epoch": epoch,
                "per_class_f1": val["per_class_f1"],
            }
            no_improve = 0
            ckpt_path = CACHE_DIR / CHECKPOINT_TEMPLATE.format(fold=fold, tag=tag)
            torch.save({
                "model_state": model.state_dict(),
                "mean": mean, "std": std,
                "fold": fold, "epoch": epoch,
                "config": {
                    "channels": [32, 64, 64, 64, 64, 128],
                    "dilations": [1, 2, 4, 8, 16, 32],
                    "embed_dim": 128,
                },
                "use_sampler": use_sampler, "sampler_power": float(sampler_power),
                "augment": augment,
                "loss": loss_kind, "la_tau": float(la_tau),
                "best_macro_f1": val["macro_f1"],
                "best_per_class_f1": val["per_class_f1"],
            }, ckpt_path)
        else:
            no_improve += 1
            # For LDAM-DRW, never early-stop during the stage-1 (pre-DRW) imbalanced phase —
            # val macro-F1 is intentionally poor there and would trip patience prematurely.
            allow_stop = (drw_start_epoch is None) or (epoch > drw_start_epoch)
            if allow_stop and no_improve >= patience:
                print(f"  early stop @ ep{epoch}: no val macroF1 improvement in {patience} epochs")
                break

    print(
        f"\nbest val macroF1 {best['macro_f1']:.4f} @ epoch {best['epoch']} "
        f"(per-class {[round(x, 3) for x in best['per_class_f1']]})"
    )
    return {"best": best, "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-sampler", action="store_true", help="Disable sqrt-inverse sampler (ablation).")
    parser.add_argument("--sampler-power", type=float, default=0.5,
                        help="Sampler exponent: weight_c ∝ 1/count_c^p. "
                             "0.5 = sqrt-inverse (mild), 1.0 = pure inverse-freq (equalize).")
    parser.add_argument("--no-augment", action="store_true", help="Disable jitter/time-roll augmentation.")
    parser.add_argument("--jitter-std", type=float, default=0.01, help="Gaussian noise std (per channel).")
    parser.add_argument("--max-shift", type=int, default=15, help="Max time-roll offset in seconds.")
    parser.add_argument("--scale-jitter", type=float, default=0.0,
                        help="Per-channel multiplicative jitter, e.g. 0.05 = ±5%.")
    parser.add_argument("--loss", choices=["ce", "logit-adjusted", "ldam"], default="ce",
                        help="Training loss. 'logit-adjusted' shifts logits by τ·log(prior_c) "
                             "(Menon et al. 2020); 'ldam' = Label-Distribution-Aware Margin + "
                             "Deferred Re-Weighting (Cao et al. 2019), larger margin for rare classes.")
    parser.add_argument("--la-tau", type=float, default=1.0,
                        help="Temperature for logit-adjusted loss (1.0 = canonical).")
    parser.add_argument("--ldam-max-m", type=float, default=0.5,
                        help="LDAM: largest class margin (rarest class). Default 0.5.")
    parser.add_argument("--ldam-s", type=float, default=30.0,
                        help="LDAM: logit temperature/scale. Default 30.")
    parser.add_argument("--drw-start-frac", type=float, default=0.6,
                        help="LDAM-DRW: fraction of epochs after which class-balanced reweighting turns on.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="", help="Suffix added to checkpoint filename (e.g. '_nosampler').")
    args = parser.parse_args()

    out = train_one_fold(
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_sampler=not args.no_sampler,
        sampler_power=args.sampler_power,
        augment=not args.no_augment,
        jitter_std=args.jitter_std,
        max_shift=args.max_shift,
        scale_jitter=args.scale_jitter,
        loss_kind=args.loss,
        la_tau=args.la_tau,
        num_workers=args.num_workers,
        seed=args.seed,
        tag=args.tag,
        ldam_max_m=args.ldam_max_m,
        ldam_s=args.ldam_s,
        drw_start_frac=args.drw_start_frac,
    )
    # Persist a small history JSON next to the checkpoint for later plotting.
    hist_path = CACHE_DIR / f"tcn_fold{args.fold}{args.tag}_history.json"
    hist_path.write_text(json.dumps(out, indent=2))
    print(f"history → {hist_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
