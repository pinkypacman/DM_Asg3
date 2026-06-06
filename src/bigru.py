"""BiGRU + attention member of the final blend — OOF + TEST probabilities,
2-seed averaged, on the raw (300×6) windows under GroupKFold(user).

Output: cache/bigru.npz  {oof, test, y_oof, file_id_test, oof_macro, oof_per_class}

Run:  python -m src.bigru [epochs]      (GPU recommended; default 30 epochs)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .data import CACHE_DIR

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttnBiGRU(nn.Module):
    def __init__(self, in_ch=6, hid=96, layers=2, nc=6, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(in_ch, hid, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=drop if layers > 1 else 0)
        self.attn = nn.Linear(2 * hid, 1)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hid, nc))

    def forward(self, x):
        h, _ = self.gru(x)
        a = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        z = (h * a.unsqueeze(-1)).sum(1)
        return self.head(z)


@torch.no_grad()
def _proba(m, X, bs=256):
    parts = []
    for s in range(0, X.shape[0], bs):
        parts.append(torch.softmax(m(X[s:s + bs]), 1).cpu().numpy())
    return np.concatenate(parts, 0)


def train_fold(X, y, Xt, fold_of, fold, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = np.flatnonzero(fold_of != fold); va = np.flatnonzero(fold_of == fold)
    flat = X[tr].reshape(-1, 6); mu = flat.mean(0); sd = flat.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    Xtn = ((Xt - mu) / sd).astype(np.float32)
    Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(y[tr])
    Xva = torch.tensor(Xn[va]).to(DEV); Xte = torch.tensor(Xtn).to(DEV)
    cnt = np.bincount(y[tr], minlength=6).astype(np.float64)
    sw = (1.0 / np.sqrt(cnt))[y[tr]]
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sw), len(tr), replacement=True)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr),
                                     batch_size=128, sampler=sampler, drop_last=True)
    prior = torch.tensor(np.log(cnt / cnt.sum() + 1e-12), dtype=torch.float32).to(DEV)
    m = AttnBiGRU().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = -1; best_val = None; best_test = None
    for ep in range(1, epochs + 1):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); out = m(xb); loss = F.cross_entropy(out + 1.0 * prior, yb)
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        sch.step(); m.eval()
        vproba = _proba(m, Xva)
        mac = f1_score(y[va], vproba.argmax(1), average="macro", zero_division=0)
        if mac > best:
            best = mac; best_val = vproba; best_test = _proba(m, Xte)
    return va, best_val, best_test, best


def main(epochs=30, seeds=(42, 7)):
    w = np.load(CACHE_DIR / "windows.npz", allow_pickle=True)
    X, y, Xt = w["X_train"], w["y_train"], w["X_test"]
    fid_test = w["file_id_test"].astype(np.int64)
    fold_of = np.load(CACHE_DIR / "folds.npz")["fold_of"]

    oof_seeds, test_seeds = [], []
    t0 = time.time()
    for seed in seeds:
        oof = np.zeros((len(y), 6), dtype=np.float32)
        test_acc = np.zeros((len(Xt), 6), dtype=np.float64)
        for k in range(5):
            va, vp, tp, best = train_fold(X, y, Xt, fold_of, k, epochs, seed)
            oof[va] = vp; test_acc += tp
            print(f"  seed{seed} fold{k} best={best:.4f} {time.time()-t0:.0f}s", flush=True)
        test = (test_acc / 5).astype(np.float32)
        mac = f1_score(y, oof.argmax(1), average="macro", zero_division=0)
        print(f"  seed{seed} OOF macro={mac:.4f}", flush=True)
        oof_seeds.append(oof); test_seeds.append(test)

    oof = np.mean(oof_seeds, 0).astype(np.float32)
    test = np.mean(test_seeds, 0).astype(np.float32)
    macro = float(f1_score(y, oof.argmax(1), average="macro", zero_division=0))
    per = f1_score(y, oof.argmax(1), labels=list(range(6)), average=None, zero_division=0).astype(np.float32)
    np.savez_compressed(CACHE_DIR / "bigru.npz", oof=oof, test=test,
                        y_oof=y.astype(np.int64), file_id_test=fid_test,
                        oof_macro=np.float32(macro), oof_per_class=per)
    print(f"\nsaved {CACHE_DIR/'bigru.npz'}  ({len(seeds)}-seed avg)")
    print(f"  OOF macroF1 {macro:.4f}  per-class {np.round(per,3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
