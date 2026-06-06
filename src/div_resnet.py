import os, time, numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES","2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS","6")
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score

C="cache/"; NC=6
dev="cuda" if torch.cuda.is_available() else "cpu"
w=np.load(C+"windows.npz")
X=w["X_train"].astype(np.float32)          # (N,300,6)
y=w["y_train"].astype(np.int64)
Xte=w["X_test"].astype(np.float32)
file_id_test=w["file_id_test"].astype(np.int64)
fold_of=np.load(C+"folds.npz")["fold_of"]
N,T,Cin=X.shape
print("data",X.shape,Xte.shape,"dev",dev)

class ResBlock(nn.Module):
    def __init__(s,ci,co,k,stride=1):
        super().__init__()
        p=k//2
        s.c1=nn.Conv1d(ci,co,k,stride=stride,padding=p,bias=False); s.b1=nn.BatchNorm1d(co)
        s.c2=nn.Conv1d(co,co,k,stride=1,padding=p,bias=False); s.b2=nn.BatchNorm1d(co)
        if ci!=co or stride!=1:
            s.sc=nn.Sequential(nn.Conv1d(ci,co,1,stride=stride,bias=False),nn.BatchNorm1d(co))
        else:
            s.sc=nn.Identity()
        s.act=nn.ReLU(inplace=True)
    def forward(s,x):
        r=s.sc(x)
        x=s.act(s.b1(s.c1(x))); x=s.b2(s.c2(x))
        return s.act(x+r)

class ResNet1D(nn.Module):
    def __init__(s,cin=6,nc=6,p=0.3):
        super().__init__()
        s.stage1=ResBlock(cin,64,7,stride=1)
        s.stage2=ResBlock(64,128,5,stride=2)
        s.stage3=ResBlock(128,256,3,stride=2)
        s.gap=nn.AdaptiveAvgPool1d(1)
        s.drop=nn.Dropout(p)
        s.fc=nn.Linear(256,nc)
    def forward(s,x):                # x: (B,6,300)
        x=s.stage1(x); x=s.stage2(x); x=s.stage3(x)
        x=s.gap(x).squeeze(-1)
        return s.fc(s.drop(x))

def znorm(tr,*arrs):
    m=tr.mean((0,1),keepdims=True); sd=tr.std((0,1),keepdims=True)+1e-6
    return [( (a-m)/sd ) for a in arrs]

def run_fold(f,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr=fold_of!=f; va=fold_of==f
    Xtr,ytr=X[tr],y[tr]; Xva,yva=X[va],y[va]
    Xtr_n,Xva_n,Xte_n=znorm(Xtr,Xtr,Xva,Xte)
    # to (B,6,300)
    def mk(a): return torch.tensor(a.transpose(0,2,1),dtype=torch.float32)
    Xtr_t,Xva_t,Xte_t=mk(Xtr_n),mk(Xva_n),mk(Xte_n)
    ytr_t=torch.tensor(ytr)
    cnt=np.bincount(ytr,minlength=NC).astype(np.float64)
    # balanced sampler
    sw=(1.0/cnt)[ytr]; sampler=WeightedRandomSampler(torch.tensor(sw,dtype=torch.double),len(sw),replacement=True)
    dl=DataLoader(TensorDataset(Xtr_t,ytr_t),batch_size=64,sampler=sampler,drop_last=True)
    # mild class-weighted CE on top of sampler
    cw=torch.tensor((cnt.sum()/(NC*cnt))**0.5,dtype=torch.float32).to(dev)
    net=ResNet1D().to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    EP=50
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP)
    lossf=nn.CrossEntropyLoss(weight=cw,label_smoothing=0.05)
    best=-1; best_va=None; best_te=None
    for ep in range(EP):
        net.train()
        for xb,yb in dl:
            xb,yb=xb.to(dev),yb.to(dev)
            opt.zero_grad(); out=net(xb); loss=lossf(out,yb); loss.backward(); opt.step()
        sch.step()
        if ep>=15 and (ep%2==0 or ep==EP-1):
            net.eval()
            with torch.no_grad():
                pv=[]
                for i in range(0,len(Xva_t),256):
                    pv.append(torch.softmax(net(Xva_t[i:i+256].to(dev)),1).cpu().numpy())
                pv=np.concatenate(pv)
            mf=f1_score(yva,pv.argmax(1),average="macro",labels=range(NC),zero_division=0)
            if mf>best:
                best=mf; best_va=pv
                with torch.no_grad():
                    pt=[]
                    for i in range(0,len(Xte_t),256):
                        pt.append(torch.softmax(net(Xte_t[i:i+256].to(dev)),1).cpu().numpy())
                    best_te=np.concatenate(pt)
    return best_va,best_te,best

SEEDS=[1,7]
oof=np.zeros((N,NC),np.float64); test=np.zeros((len(Xte),NC),np.float64)
t0=time.time()
for f in range(5):
    va_acc=np.zeros((np.sum(fold_of==f),NC)); te_acc=np.zeros((len(Xte),NC)); bests=[]
    for sd in SEEDS:
        bva,bte,bf=run_fold(f,sd); va_acc+=bva; te_acc+=bte; bests.append(bf)
    va_acc/=len(SEEDS); te_acc/=len(SEEDS)
    oof[fold_of==f]=va_acc; test+=te_acc/5
    print(f"fold {f} best-val mF1 {np.mean(bests):.4f}  elapsed {time.time()-t0:.0f}s")

oof=oof/oof.sum(1,keepdims=True); test=test/test.sum(1,keepdims=True)
np.savez_compressed(C+"div_resnet.npz",oof=oof.astype("float32"),test=test.astype("float32"),
                    y_oof=y.astype("int64"),file_id_test=file_id_test.astype("int64"))
print("saved cache/div_resnet.npz  total",time.time()-t0,"s")
