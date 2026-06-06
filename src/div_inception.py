import os, time, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "6")
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score

torch.set_num_threads(6)
C = "cache/"
NC = 6
SEED = 1234
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev)

# ---- data ----
w = np.load(C + "windows.npz", allow_pickle=True)
X_train = w["X_train"].astype(np.float32)          # (N,300,6)
X_test = w["X_test"].astype(np.float32)            # (M,300,6)
y_train = w["y_train"].astype(np.int64)
file_id_test = w["file_id_test"].astype(np.int64)
fold_of = np.load(C + "folds.npz", allow_pickle=True)["fold_of"]
N, T, Cin = X_train.shape
M = X_test.shape[0]
print("X_train", X_train.shape, "X_test", X_test.shape)


# ---- model: compact InceptionTime ----
class InceptionModule(nn.Module):
    def __init__(self, in_ch, bottleneck=32, nf=32, ks=(9, 19, 39)):
        super().__init__()
        self.use_bottleneck = in_ch > 1
        b_ch = bottleneck if self.use_bottleneck else in_ch
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(b_ch, nf, kernel_size=k, padding=k // 2, bias=False) for k in ks
        ])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_ch, nf, 1, bias=False)
        out_ch = nf * (len(ks) + 1)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        inp = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        outs = [c(x) for c in self.convs]
        outs.append(self.conv_pool(self.maxpool(inp)))
        z = torch.cat(outs, dim=1)
        return F.relu(self.bn(z))


class InceptionTime(nn.Module):
    def __init__(self, in_ch=6, nf=32, depth=6, n_classes=NC, ks=(9, 19, 39)):
        super().__init__()
        out_ch = nf * (len(ks) + 1)   # = 128 with nf=32
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        self.use_res = []
        prev = in_ch
        res_in = in_ch
        for d in range(depth):
            self.blocks.append(InceptionModule(prev, bottleneck=32, nf=nf, ks=ks))
            prev = out_ch
            if (d + 1) % 3 == 0:   # residual every 3
                self.use_res.append(True)
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(res_in, out_ch, 1, bias=False),
                    nn.BatchNorm1d(out_ch),
                ))
                res_in = out_ch
            else:
                self.use_res.append(False)
                self.shortcuts.append(None)
        self.head = nn.Linear(out_ch, n_classes)

    def forward(self, x):
        # x: (B,T,Cin) -> (B,Cin,T)
        x = x.transpose(1, 2)
        res = x
        si = 0
        for d, blk in enumerate(self.blocks):
            x = blk(x)
            if self.use_res[d]:
                x = F.relu(x + self.shortcuts[d](res))
                res = x
        x = x.mean(dim=2)            # GlobalAveragePool
        return self.head(x)


# logit-adjusted CE for imbalance (tau-scaled log prior)
def make_logit_adjust(y, tau=1.0):
    cnt = np.bincount(y, minlength=NC).astype(np.float64)
    prior = cnt / cnt.sum()
    return torch.tensor(tau * np.log(prior + 1e-12), dtype=torch.float32, device=dev)


def batched_logits(model, Xt, bs=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], bs):
            xb = torch.from_numpy(Xt[i:i + bs]).to(dev)
            outs.append(model(xb).softmax(1).cpu().numpy())
    return np.concatenate(outs, 0)


def train_fold(Xtr, ytr, Xva, la_logit, epochs=40, bs=64, lr=1e-3):
    model = InceptionTime().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # class-balanced sampler
    cnt = np.bincount(ytr, minlength=NC).astype(np.float64)
    sw = (1.0 / (cnt[ytr] + 1e-9))
    sw = sw / sw.sum()
    n_tr = len(ytr)
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    for ep in range(epochs):
        model.train()
        # weighted sampling with replacement, one epoch ~ n_tr samples
        idx = np.random.choice(n_tr, size=n_tr, replace=True, p=sw)
        for i in range(0, n_tr, bs):
            bidx = idx[i:i + bs]
            xb = Xtr_t[bidx].to(dev)
            yb = ytr_t[bidx].to(dev)
            logits = model(xb) + la_logit  # logit adjustment
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def znorm_fit(X):
    # per-channel z-norm stats from train (over N and T)
    mu = X.mean(axis=(0, 1), keepdims=True)
    sd = X.std(axis=(0, 1), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


oof = np.zeros((N, NC), np.float32)
test_acc = np.zeros((M, NC), np.float32)
la_logit = make_logit_adjust(y_train, tau=1.0)

t0 = time.time()
for f in range(5):
    tr = fold_of != f
    va = fold_of == f
    mu, sd = znorm_fit(X_train[tr])
    Xtr = (X_train[tr] - mu) / sd
    Xva = (X_train[va] - mu) / sd
    Xte = (X_test - mu) / sd
    model = train_fold(Xtr, y_train[tr], Xva, la_logit)
    oof[va] = batched_logits(model, Xva)
    test_acc += batched_logits(model, Xte) / 5.0
    vf1 = f1_score(y_train[va], oof[va].argmax(1), average="macro",
                   labels=range(NC), zero_division=0)
    print(f"fold {f} done  va_macroF1={vf1:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

# renormalize (already softmax, but be safe)
oof = oof / oof.sum(1, keepdims=True)
test_acc = test_acc / test_acc.sum(1, keepdims=True)

np.savez_compressed(C + "div_inception.npz",
                    oof=oof.astype("float32"),
                    test=test_acc.astype("float32"),
                    y_oof=y_train.astype("int64"),
                    file_id_test=file_id_test.astype("int64"))
print("saved cache/div_inception.npz  total_time=%.0fs" % (time.time() - t0))

# ---- evaluation ----
B = np.array([0.332, 0.06, 0.991, 0.289, -0.077, -0.066])
y = y_train
def Lf(fn, k): return np.load(C + fn, allow_pickle=True)[k].astype(float)
v18 = (0.30 * Lf("postprocess_tcn_la05.npz", "oof_raw")
       + 0.23 * Lf("gbdt_richest.npz", "oof")
       + 0.24 * Lf("xgb_richest.npz", "oof")
       + 0.23 * Lf("_catboost_oof.npz", "oof"))
v18 /= v18.sum(1, keepdims=True)
def postbias_macro(p):
    lp = np.log(np.clip(p, 1e-8, 1)) + B
    return f1_score(y, lp.argmax(1), average="macro", labels=range(NC), zero_division=0)
V18_BASE = postbias_macro(v18)
my_oof = oof
standalone = postbias_macro(my_oof)
raw = f1_score(y, my_oof.argmax(1), average="macro", labels=range(NC), zero_division=0)
disagreement = float((v18.argmax(1) != my_oof.argmax(1)).mean() * 100)
mixed = 0.85 * v18 + 0.15 * (my_oof / my_oof.sum(1, keepdims=True))
mixed /= mixed.sum(1, keepdims=True)
blend_lift = postbias_macro(mixed) - V18_BASE

from sklearn.metrics import f1_score as f1s
per_class = f1s(y, my_oof.argmax(1), average=None, labels=range(NC), zero_division=0)
v18_pc = f1s(y, v18.argmax(1), average=None, labels=range(NC), zero_division=0)
mixed_pc = f1s(y, (np.log(np.clip(mixed,1e-8,1))+B).argmax(1), average=None, labels=range(NC), zero_division=0)

print("==RESULTS==")
print("V18_BASE", round(V18_BASE, 4))
print("raw_oof_macro", round(raw, 4))
print("standalone_oof_macro", round(standalone, 4))
print("disagreement_pct", round(disagreement, 2))
print("blend_lift", round(blend_lift, 4))
print("per_class_standalone", np.round(per_class, 3).tolist())
print("v18_per_class", np.round(v18_pc, 3).tolist())
print("mixed_per_class", np.round(mixed_pc, 3).tolist())
