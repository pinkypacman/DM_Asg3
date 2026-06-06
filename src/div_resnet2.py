import os, time, numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS","6")
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score
from tqdm import tqdm

C="cache/"; NC=6
dev="cuda" if torch.cuda.is_available() else "cpu"
w=np.load(C+"windows.npz")
X=w["X_train"].astype(np.float32); y=w["y_train"].astype(np.int64)
Xte=w["X_test"].astype(np.float32); file_id_test=w["file_id_test"].astype(np.int64)
fold_of=np.load(C+"folds.npz")["fold_of"]
N,T,Cin=X.shape
print("STRONGER resnet; data",X.shape,Xte.shape,"dev",dev,flush=True)

class SE(nn.Module):
    def __init__(s,c,r=8):
        super().__init__(); s.fc=nn.Sequential(nn.Linear(c,max(c//r,4)),nn.ReLU(inplace=True),nn.Linear(max(c//r,4),c),nn.Sigmoid())
    def forward(s,x): w=x.mean(-1); return x*s.fc(w).unsqueeze(-1)

class ResBlock(nn.Module):
    def __init__(s,ci,co,k,stride=1,p=0.1):
        super().__init__(); pad=k//2
        s.c1=nn.Conv1d(ci,co,k,stride=stride,padding=pad,bias=False); s.b1=nn.BatchNorm1d(co)
        s.c2=nn.Conv1d(co,co,k,stride=1,padding=pad,bias=False); s.b2=nn.BatchNorm1d(co)
        s.se=SE(co); s.drop=nn.Dropout(p)
        s.sc=nn.Sequential(nn.Conv1d(ci,co,1,stride=stride,bias=False),nn.BatchNorm1d(co)) if (ci!=co or stride!=1) else nn.Identity()
        s.act=nn.ReLU(inplace=True)
    def forward(s,x):
        r=s.sc(x); x=s.act(s.b1(s.c1(x))); x=s.drop(x); x=s.se(s.b2(s.c2(x)))
        return s.act(x+r)

class ResNet1D(nn.Module):
    def __init__(s,cin=6,nc=6,p=0.4):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv1d(cin,48,7,padding=3,bias=False),nn.BatchNorm1d(48),nn.ReLU(inplace=True))
        s.s1=nn.Sequential(ResBlock(48,64,7),ResBlock(64,64,7))
        s.s2=nn.Sequential(ResBlock(64,128,5,stride=2),ResBlock(128,128,5))
        s.s3=nn.Sequential(ResBlock(128,256,3,stride=2),ResBlock(256,256,3))
        s.gap=nn.AdaptiveAvgPool1d(1); s.drop=nn.Dropout(p); s.fc=nn.Linear(256,nc)
    def forward(s,x):
        x=s.stem(x); x=s.s1(x); x=s.s2(x); x=s.s3(x)
        return s.fc(s.drop(s.gap(x).squeeze(-1)))

def znorm(tr,*arrs):
    m=tr.mean((0,1),keepdims=True); sd=tr.std((0,1),keepdims=True)+1e-6
    return [((a-m)/sd) for a in arrs]

def augment(xb):                       # xb: (B,6,300) on device — train only
    xb=torch.roll(xb,shifts=int(torch.randint(-20,21,(1,)).item()),dims=2)   # time-shift
    xb=xb*(0.9+0.2*torch.rand(xb.size(0),1,1,device=xb.device))              # magnitude scale
    return xb+0.01*torch.randn_like(xb)                                       # jitter

EP=65; EVAL_EVERY=2; EVAL_START=10; PATIENCE=10   # early-stop after PATIENCE evals (=20 ep) w/o val-mF1 improvement

def run_fold(f,seed,desc=""):
    torch.manual_seed(seed); np.random.seed(seed)
    tr=fold_of!=f; va=fold_of==f
    Xtr,ytr=X[tr],y[tr]; Xva,yva=X[va],y[va]
    Xtr_n,Xva_n,Xte_n=znorm(Xtr,Xtr,Xva,Xte)
    def mk(a): return torch.tensor(a.transpose(0,2,1),dtype=torch.float32)
    Xtr_t,Xva_t,Xte_t=mk(Xtr_n),mk(Xva_n),mk(Xte_n); ytr_t=torch.tensor(ytr)
    cnt=np.bincount(ytr,minlength=NC).astype(np.float64)
    sw=(1.0/cnt)[ytr]; sampler=WeightedRandomSampler(torch.tensor(sw,dtype=torch.double),len(sw),replacement=True)
    dl=DataLoader(TensorDataset(Xtr_t,ytr_t),batch_size=64,sampler=sampler,drop_last=True)
    cw=torch.tensor((cnt.sum()/(NC*cnt))**0.5,dtype=torch.float32).to(dev)
    net=ResNet1D().to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=1.2e-3,weight_decay=2e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP)
    lossf=nn.CrossEntropyLoss(weight=cw,label_smoothing=0.05)
    # SAVE-BEST: keep the predictions AND the model weights at the best val-macroF1 epoch (never the final/overfit epoch)
    best=-1; best_ep=-1; best_va=None; best_te=None; best_state=None; no_improve=0
    pbar=tqdm(range(EP), desc=desc, leave=False, dynamic_ncols=True, mininterval=3.0)
    for ep in pbar:
        net.train(); run_loss=0.0; nb=0
        for xb,yb in dl:
            xb,yb=xb.to(dev),yb.to(dev); xb=augment(xb)
            opt.zero_grad(); loss=lossf(net(xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
            run_loss+=float(loss.item()); nb+=1
        sch.step()
        do_eval = ep>=EVAL_START and (ep%EVAL_EVERY==0 or ep==EP-1)
        if do_eval:
            net.eval()
            with torch.no_grad():
                pv=np.concatenate([torch.softmax(net(Xva_t[i:i+256].to(dev)),1).cpu().numpy() for i in range(0,len(Xva_t),256)])
            mf=f1_score(yva,pv.argmax(1),average="macro",labels=range(NC),zero_division=0)
            if mf>best+1e-5:
                best=mf; best_ep=ep; best_va=pv; no_improve=0
                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}  # save-best model
                with torch.no_grad():
                    best_te=np.concatenate([torch.softmax(net(Xte_t[i:i+256].to(dev)),1).cpu().numpy() for i in range(0,len(Xte_t),256)])
            else:
                no_improve+=1
            pbar.set_postfix(loss=f"{run_loss/max(nb,1):.3f}", val_mF1=f"{mf:.4f}", best=f"{best:.4f}@{best_ep}", stop=f"{no_improve}/{PATIENCE}")
            if no_improve>=PATIENCE:                # EARLY STOPPING (best preds already saved)
                pbar.close(); break
    pbar.close()
    return best_va,best_te,best,best_ep

SEEDS=[1,7,13]
oof=np.zeros((N,NC),np.float64); test=np.zeros((len(Xte),NC),np.float64)
t0=time.time(); ntrain=5*len(SEEDS); done=0
for f in range(5):
    va_acc=np.zeros((np.sum(fold_of==f),NC)); te_acc=np.zeros((len(Xte),NC)); bests=[]
    for sd in SEEDS:
        bva,bte,bf,bep=run_fold(f,sd,desc=f"fold{f} seed{sd}")
        va_acc+=bva; te_acc+=bte; bests.append(bf); done+=1
        el=time.time()-t0; eta=el/done*(ntrain-done)
        print(f"  [{done:2d}/{ntrain}] fold{f} seed{sd}: best-val mF1 {bf:.4f} @ep{bep}  "
              f"| elapsed {el/60:.1f}m  ETA {eta/60:.1f}m",flush=True)
    va_acc/=len(SEEDS); te_acc/=len(SEEDS)
    oof[fold_of==f]=va_acc; test+=te_acc/5
    print(f"fold {f} DONE: mean best-val mF1 {np.mean(bests):.4f}  elapsed {(time.time()-t0)/60:.1f}m",flush=True)
oof/=oof.sum(1,keepdims=True); test/=test.sum(1,keepdims=True)
np.savez_compressed(C+"div_resnet2.npz",oof=oof.astype("float32"),test=test.astype("float32"),
                    y_oof=y.astype("int64"),file_id_test=file_id_test.astype("int64"))
print("saved cache/div_resnet2.npz  raw OOF mF1 %.4f  total %.0fs"%(
    f1_score(y,oof.argmax(1),average='macro',zero_division=0),time.time()-t0),flush=True)
print("DONE_RESNET2",flush=True)
