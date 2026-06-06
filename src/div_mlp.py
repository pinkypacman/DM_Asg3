import os, time
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

C = "cache/"; NC = 6
torch.manual_seed(0); np.random.seed(0)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev)

f = np.load(C + "features_richest.npz", allow_pickle=True)
F_train = f["F_train"].astype("float32")
F_test = f["F_test"].astype("float32")
y_train = f["y_train"].astype("int64")
file_id_test = f["file_id_test"].astype("int64")
fold_of = np.load(C + "folds.npz")["fold_of"]
N, D = F_train.shape
print("N,D", N, D, "test", F_test.shape)

# v5 frozen biases for reporting
B = np.array([0.332, 0.06, 0.991, 0.289, -0.077, -0.066])


class MLP(nn.Module):
    def __init__(self, d_in, p=0.3):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Linear(i, o), nn.BatchNorm1d(o), nn.GELU(), nn.Dropout(p))
        self.net = nn.Sequential(
            blk(d_in, 512), blk(512, 256), blk(256, 128), nn.Linear(128, NC)
        )

    def forward(self, x):
        return self.net(x)


def train_one(Xtr, ytr, Xva, scaler, epochs=70, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_s = scaler.transform(Xtr).astype("float32")
    Xva_s = scaler.transform(Xva).astype("float32")
    # class-balanced sampler
    cnt = np.bincount(ytr, minlength=NC).astype("float64")
    cw = 1.0 / np.clip(cnt, 1, None)
    samp_w = cw[ytr]
    sampler = WeightedRandomSampler(torch.as_tensor(samp_w), num_samples=len(ytr), replacement=True)
    ds = TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=256, sampler=sampler, drop_last=False)
    # mild class-weighted CE too (sqrt to not over-correct on top of sampler)
    ce_w = torch.as_tensor((cw / cw.mean()) ** 0.5, dtype=torch.float32, device=dev)
    model = MLP(Xtr_s.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=ce_w, label_smoothing=0.05)
    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb = xb.to(dev); yb = yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        va = torch.softmax(model(torch.from_numpy(Xva_s).to(dev)), 1).cpu().numpy()
        Xte_s = scaler.transform(F_test).astype("float32")
        te = torch.softmax(model(torch.from_numpy(Xte_s).to(dev)), 1).cpu().numpy()
    return va, te


oof = np.zeros((N, NC), "float64")
test = np.zeros((F_test.shape[0], NC), "float64")
SEEDS = [0, 1, 2]  # seed-averaging for stability/diversity
t0 = time.time()
for fcur in range(5):
    tr = fold_of != fcur
    va = fold_of == fcur
    scaler = StandardScaler().fit(F_train[tr])
    va_acc = np.zeros((va.sum(), NC), "float64")
    te_acc = np.zeros((F_test.shape[0], NC), "float64")
    for s in SEEDS:
        va_p, te_p = train_one(F_train[tr], y_train[tr], F_train[va], scaler, epochs=70, seed=s)
        va_acc += va_p; te_acc += te_p
    va_acc /= len(SEEDS); te_acc /= len(SEEDS)
    oof[va] = va_acc
    test += te_acc / 5.0
    fm = f1_score(y_train[va], va_acc.argmax(1), average="macro", labels=range(NC), zero_division=0)
    print(f"fold {fcur} macro {fm:.4f}  elapsed {time.time()-t0:.0f}s")

oof = oof / oof.sum(1, keepdims=True)
test = test / test.sum(1, keepdims=True)

# raw macro
raw_macro = f1_score(y_train, oof.argmax(1), average="macro", labels=range(NC), zero_division=0)
print("raw_oof_macro", raw_macro)

np.savez_compressed(C + "div_mlp.npz",
                    oof=oof.astype("float32"), test=test.astype("float32"),
                    y_oof=y_train.astype("int64"), file_id_test=file_id_test.astype("int64"))
print("saved cache/div_mlp.npz")
print("total time", time.time() - t0)
