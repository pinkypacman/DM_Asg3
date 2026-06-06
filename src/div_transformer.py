import os, time, math
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score

C = "cache/"
NC = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 7]

w = np.load(C + "windows.npz")
X_train = w["X_train"].astype(np.float32)      # (11020,300,6)
y_train = w["y_train"].astype(np.int64)
X_test = w["X_test"].astype(np.float32)         # (6849,300,6)
file_id_test = w["file_id_test"].astype(np.int64)
fold_of = np.load(C + "folds.npz")["fold_of"]

N, T, Cin = X_train.shape
NTEST = X_test.shape[0]
print(f"X_train {X_train.shape} X_test {X_test.shape} device {DEVICE}")

# logit-adjusted CE base: log prior
cls_counts = np.bincount(y_train, minlength=NC).astype(np.float64)
log_prior = torch.tensor(np.log(cls_counts / cls_counts.sum()), dtype=torch.float32, device=DEVICE)


class TransEnc(nn.Module):
    def __init__(self, cin=6, d_model=64, nhead=4, dff=128, nlayers=3, dropout=0.2, seqlen=150):
        super().__init__()
        # stride-2 conv stem: downsample 300 -> 150, project channels to d_model
        self.stem = nn.Conv1d(cin, d_model, kernel_size=5, stride=2, padding=2)
        self.pos = nn.Parameter(torch.zeros(1, seqlen, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        # attention pooling
        self.attn = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, NC)

    def forward(self, x):
        # x: (B,T,Cin)
        x = x.transpose(1, 2)            # (B,Cin,T)
        x = self.stem(x)                 # (B,d_model,T/2)
        x = x.transpose(1, 2)            # (B,T/2,d_model)
        x = x + self.pos[:, :x.size(1), :]
        x = self.enc(x)
        x = self.norm(x)
        a = torch.softmax(self.attn(x), dim=1)   # (B,L,1)
        pooled = (x * a).sum(1)                    # (B,d_model)
        return self.head(self.drop(pooled))


def make_loader(Xt, yt, bs=64, train=True):
    ds = TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
    if train:
        cc = np.bincount(yt, minlength=NC).astype(np.float64)
        cw = 1.0 / np.maximum(cc, 1)
        sw = cw[yt]
        sampler = WeightedRandomSampler(torch.from_numpy(sw).double(), num_samples=len(yt), replacement=True)
        return DataLoader(ds, batch_size=bs, sampler=sampler, drop_last=True, num_workers=0)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)


def train_one(Xtr, ytr, Xva, yva, seqlen, seed, epochs=50):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TransEnc(seqlen=seqlen).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    tr_loader = make_loader(Xtr, ytr, bs=64, train=True)
    steps_per = len(tr_loader)
    total_steps = steps_per * epochs
    warmup = steps_per * 3
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    va_loader = make_loader(Xva, yva, train=False)

    best_f1, best_state, best_ep = -1, None, -1
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb) + log_prior  # logit-adjusted CE
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        # val every few epochs (and last 10)
        if ep >= 10 and (ep % 2 == 0 or ep >= epochs - 6):
            model.eval(); preds = []
            with torch.no_grad():
                for xb, _ in va_loader:
                    logits = model(xb.to(DEVICE))
                    preds.append(logits.argmax(1).cpu().numpy())
            f1 = f1_score(yva, np.concatenate(preds), average="macro", labels=range(NC), zero_division=0)
            if f1 > best_f1:
                best_f1, best_ep = f1, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1, best_ep


def predict(model, X, seqlen):
    model.eval()
    out = np.zeros((X.shape[0], NC), dtype=np.float64)
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=256, shuffle=False)
    i = 0
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(DEVICE))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            out[i:i+len(p)] = p
            i += len(p)
    return out


SEQLEN = (T + 1) // 2  # stem stride2: ceil(300/2)=150
oof = np.zeros((N, NC), dtype=np.float64)
test = np.zeros((NTEST, NC), dtype=np.float64)
t0 = time.time()

for f in range(5):
    tr = fold_of != f
    va = fold_of == f
    Xtr_raw, ytr = X_train[tr], y_train[tr]
    Xva_raw, yva = X_train[va], y_train[va]
    # per-channel z-norm using train-fold stats only
    mu = Xtr_raw.reshape(-1, Cin).mean(0)
    sd = Xtr_raw.reshape(-1, Cin).std(0) + 1e-6
    def nz(a): return ((a - mu) / sd).astype(np.float32)
    Xtr, Xva, Xte = nz(Xtr_raw), nz(Xva_raw), nz(X_test)

    fold_va = np.zeros((va.sum(), NC), dtype=np.float64)
    fold_te = np.zeros((NTEST, NC), dtype=np.float64)
    for s in SEEDS:
        model, bf1, bep = train_one(Xtr, ytr, Xva, yva, SEQLEN, s, epochs=50)
        fold_va += predict(model, Xva, SEQLEN) / len(SEEDS)
        fold_te += predict(model, Xte, SEQLEN) / len(SEEDS)
        print(f"  fold{f} seed{s} best_va_f1={bf1:.4f} @ep{bep}  t={time.time()-t0:.0f}s")
        del model; torch.cuda.empty_cache()
    oof[va] = fold_va
    test += fold_te / 5.0
    fpred = oof[va].argmax(1)
    print(f"fold{f} ensembled val macroF1={f1_score(yva, fpred, average='macro', labels=range(NC), zero_division=0):.4f}")

# normalize to sum 1
oof /= oof.sum(1, keepdims=True)
test /= test.sum(1, keepdims=True)

np.savez_compressed(C + "div_transformer.npz",
                    oof=oof.astype("float32"), test=test.astype("float32"),
                    y_oof=y_train.astype("int64"), file_id_test=file_id_test.astype("int64"))
raw = f1_score(y_train, oof.argmax(1), average="macro", labels=range(NC), zero_division=0)
print(f"RAW OOF macroF1 = {raw:.4f}  total t={time.time()-t0:.0f}s")
print("per-class OOF argmax counts:", np.bincount(oof.argmax(1), minlength=NC))
