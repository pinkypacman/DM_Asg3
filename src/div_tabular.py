import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

C = "cache/"
NC = 6
SEED = 1234
np.random.seed(SEED); torch.manual_seed(SEED)

# ---- load data ----
d = np.load(C + "features_richest.npz", allow_pickle=True)
F_train = d["F_train"].astype("float32")
F_test = d["F_test"].astype("float32")
y_train = d["y_train"].astype("int64")
file_id_test = d["file_id_test"].astype("int64")
fold_of = np.load(C + "folds.npz", allow_pickle=True)["fold_of"].astype("int64")

N, P = F_train.shape
print(f"F_train {F_train.shape} F_test {F_test.shape} folds {np.bincount(fold_of)}", flush=True)

counts = np.bincount(y_train, minlength=NC).astype("float64")
# class weights (inverse-freq, smoothed) for loss
cw = (counts.sum() / (NC * counts))
cw = cw / cw.mean()
print("class counts", counts.astype(int), "class weights", np.round(cw, 3), flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, flush=True)

from pytorch_tabnet.tab_model import TabNetClassifier

oof = np.zeros((N, NC), dtype="float64")
test = np.zeros((F_test.shape[0], NC), dtype="float64")

weights_t = torch.tensor(cw, dtype=torch.float32, device=device)
def weighted_ce(y_pred, y_true):
    return torch.nn.functional.cross_entropy(y_pred, y_true, weight=weights_t)

t0 = time.time()
for f in range(5):
    tr = fold_of != f
    va = fold_of == f
    sc = StandardScaler().fit(F_train[tr])
    Xtr = sc.transform(F_train[tr]).astype("float32")
    Xva = sc.transform(F_train[va]).astype("float32")
    Xte = sc.transform(F_test).astype("float32")
    ytr = y_train[tr]; yva = y_train[va]

    clf = TabNetClassifier(
        n_d=32, n_a=32, n_steps=4, gamma=1.5,
        n_independent=2, n_shared=2, lambda_sparse=1e-4,
        momentum=0.3, clip_value=2.0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
        scheduler_params=dict(max_lr=2e-2, steps_per_epoch=int(np.ceil(tr.sum()/1024)), epochs=120),
        mask_type="entmax",
        seed=SEED + f, verbose=0, device_name=device,
    )
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)], eval_name=["val"], eval_metric=["balanced_accuracy"],
        loss_fn=weighted_ce,
        max_epochs=120, patience=25, batch_size=1024, virtual_batch_size=256,
        num_workers=0, drop_last=False,
    )
    pv = clf.predict_proba(Xva)
    pt = clf.predict_proba(Xte)
    oof[va] = pv
    test += pt / 5.0
    fm = f1_score(yva, pv.argmax(1), average="macro", labels=range(NC), zero_division=0)
    print(f"fold {f} done best_epoch={clf.best_epoch} val_macroF1(raw)={fm:.4f} elapsed={time.time()-t0:.0f}s", flush=True)

oof = oof / oof.sum(1, keepdims=True)
test = test / test.sum(1, keepdims=True)

np.savez_compressed(C + "div_tabular.npz",
                    oof=oof.astype("float32"), test=test.astype("float32"),
                    y_oof=y_train.astype("int64"), file_id_test=file_id_test.astype("int64"))
print("SAVED cache/div_tabular.npz", flush=True)

# ---- evaluation ----
B = np.array([0.332,0.06,0.991,0.289,-0.077,-0.066])
def Lc(fn,k): return np.load(C+fn,allow_pickle=True)[k].astype(float)
v18=0.30*Lc("postprocess_tcn_la05.npz","oof_raw")+0.23*Lc("gbdt_richest.npz","oof")+0.24*Lc("xgb_richest.npz","oof")+0.23*Lc("_catboost_oof.npz","oof")
v18/=v18.sum(1,keepdims=True)
y = y_train
def postbias_macro(p):
    lp=np.log(np.clip(p,1e-8,1))+B; return f1_score(y,lp.argmax(1),average="macro",labels=range(NC),zero_division=0)
def raw_macro(p):
    return f1_score(y,p.argmax(1),average="macro",labels=range(NC),zero_division=0)
V18_BASE=postbias_macro(v18)
my_oof=oof
standalone=postbias_macro(my_oof)
raw=raw_macro(my_oof)
disagreement=float((v18.argmax(1)!=my_oof.argmax(1)).mean()*100)
mixed=0.85*v18+0.15*(my_oof/my_oof.sum(1,keepdims=True)); mixed/=mixed.sum(1,keepdims=True)
blend_lift=postbias_macro(mixed)-V18_BASE
# per-class postbias F1 for the standalone model
lp=np.log(np.clip(my_oof,1e-8,1))+B
pc=f1_score(y,lp.argmax(1),average=None,labels=range(NC),zero_division=0)
print("==RESULTS==", flush=True)
print(f"V18_BASE={V18_BASE:.4f}", flush=True)
print(f"raw_oof_macro={raw:.4f}", flush=True)
print(f"standalone_postbias_macro={standalone:.4f}", flush=True)
print(f"disagreement_pct={disagreement:.2f}", flush=True)
print(f"blend_lift={blend_lift:.4f}  (mixed={V18_BASE+blend_lift:.4f})", flush=True)
print(f"per_class_postbias_F1={np.round(pc,3).tolist()}", flush=True)
