"""Pure-PyTorch bidirectional diagonal SSM (S4D / LRU-style) classifier — a NEW
inductive-bias family (structured linear recurrence) for ensemble diversity.
Diagonal complex state-space, stable LRU parameterization, FFT convolution mode
(no mamba-ssm / causal-conv1d kernels needed). Same harness as div_resnet2:
GroupKFold OOF, 3 seeds, save-best, early-stopping, tqdm. No augmentation.
"""
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
N_,T,Cin=X.shape
print("SSM (S4D/LRU bidir); data",X.shape,Xte.shape,"dev",dev,flush=True)

class S4D(nn.Module):
    """Bidirectional diagonal SSM. Real input/output via FFT depthwise conv."""
    def __init__(s,H,Nstate=16,bidir=True):
        super().__init__(); s.H=H; s.N=Nstate; s.bidir=bidir; nd=2 if bidir else 1
        s.lognu=nn.Parameter(torch.log(0.5*torch.rand(nd,H,Nstate)+0.01))   # |lambda|=exp(-exp(lognu))<1
        s.theta=nn.Parameter(6.283*torch.rand(nd,H,Nstate))
        s.B=nn.Parameter(torch.randn(nd,H,Nstate)/Nstate**0.5)
        s.Cr=nn.Parameter(torch.randn(nd,H,Nstate)/Nstate**0.5)
        s.Ci=nn.Parameter(torch.randn(nd,H,Nstate)/Nstate**0.5)
        s.D=nn.Parameter(torch.randn(H))
    def kernel(s,L,d):
        loglam=(-torch.exp(s.lognu[d])+1j*s.theta[d])                # (H,N) complex
        lam=torch.exp(loglam)
        gamma=torch.sqrt(torch.clamp(1-(lam.abs()**2),min=1e-6))     # (H,N)
        l=torch.arange(L,device=s.lognu.device).view(L,1,1)
        lam_pows=torch.exp(l*loglam.unsqueeze(0))                    # (L,H,N) complex
        Cc=(s.Cr[d]+1j*s.Ci[d]).unsqueeze(0)
        Bc=(s.B[d]*gamma).to(torch.complex64).unsqueeze(0)
        return (Cc*lam_pows*Bc).sum(-1).real                        # (L,H) real kernel
    def _conv(s,u,K):                                                # u (B,L,H), K (L,H) causal
        L=u.shape[1]; n=2*L
        Uf=torch.fft.rfft(u,n=n,dim=1); Kf=torch.fft.rfft(K,n=n,dim=0).unsqueeze(0)
        return torch.fft.irfft(Uf*Kf,n=n,dim=1)[:,:L,:]
    def forward(s,u):                                               # (B,L,H)
        L=u.shape[1]; yf=s._conv(u,s.kernel(L,0))
        if s.bidir:
            yb=torch.flip(s._conv(torch.flip(u,[1]),s.kernel(L,1)),[1]); yf=yf+yb
        return yf + s.D.view(1,1,-1)*u

class SSMBlock(nn.Module):
    def __init__(s,H,Nstate,drop):
        super().__init__()
        s.n1=nn.LayerNorm(H); s.ssm=S4D(H,Nstate); s.glu=nn.Linear(H,2*H)
        s.n2=nn.LayerNorm(H); s.ffn=nn.Sequential(nn.Linear(H,2*H),nn.GELU(),nn.Dropout(drop),nn.Linear(2*H,H))
        s.drop=nn.Dropout(drop)
    def forward(s,x):
        z=s.ssm(s.n1(x)); a,b=s.glu(z).chunk(2,-1); x=x+s.drop(a*torch.sigmoid(b))
        return x+s.drop(s.ffn(s.n2(x)))

class SSMNet(nn.Module):
    def __init__(s,cin=6,H=96,Nstate=12,layers=3,nc=6,drop=0.2):
        super().__init__()
        s.stem=nn.Linear(cin,H); s.blocks=nn.ModuleList([SSMBlock(H,Nstate,drop) for _ in range(layers)])
        s.norm=nn.LayerNorm(H); s.head=nn.Sequential(nn.Dropout(drop),nn.Linear(H,nc))
    def forward(s,x):                                              # x (B,L,6)
        x=s.stem(x)
        for b in s.blocks: x=b(x)
        return s.head(s.norm(x).mean(1))

def znorm(tr,*arrs):
    m=tr.mean((0,1),keepdims=True); sd=tr.std((0,1),keepdims=True)+1e-6
    return [((a-m)/sd) for a in arrs]

EP=60; EVAL_EVERY=2; EVAL_START=8; PATIENCE=10

def run_fold(f,seed,desc=""):
    torch.manual_seed(seed); np.random.seed(seed)
    tr=fold_of!=f; va=fold_of==f
    Xtr,ytr=X[tr],y[tr]; Xva,yva=X[va],y[va]
    Xtr_n,Xva_n,Xte_n=znorm(Xtr,Xtr,Xva,Xte)
    mk=lambda a: torch.tensor(a,dtype=torch.float32)              # keep (B,300,6)
    Xtr_t,Xva_t,Xte_t=mk(Xtr_n),mk(Xva_n),mk(Xte_n); ytr_t=torch.tensor(ytr)
    cnt=np.bincount(ytr,minlength=NC).astype(np.float64)
    sw=(1.0/cnt)[ytr]; sampler=WeightedRandomSampler(torch.tensor(sw,dtype=torch.double),len(sw),replacement=True)
    dl=DataLoader(TensorDataset(Xtr_t,ytr_t),batch_size=64,sampler=sampler,drop_last=True)
    cw=torch.tensor((cnt.sum()/(NC*cnt))**0.5,dtype=torch.float32).to(dev)
    net=SSMNet().to(dev)
    # no weight-decay on SSM recurrence params (standard SSM practice)
    nodecay={n for n,_ in net.named_parameters() if any(k in n for k in ["lognu","theta",".B",".Cr",".Ci",".D"])}
    g1=[p for n,p in net.named_parameters() if n in nodecay]; g2=[p for n,p in net.named_parameters() if n not in nodecay]
    opt=torch.optim.AdamW([{"params":g2,"weight_decay":5e-2},{"params":g1,"weight_decay":0.0}],lr=2e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP)
    lossf=nn.CrossEntropyLoss(weight=cw,label_smoothing=0.05)
    best=-1; best_ep=-1; best_va=None; best_te=None; best_state=None; no_improve=0
    pbar=tqdm(range(EP),desc=desc,leave=False,dynamic_ncols=True,mininterval=3.0)
    for ep in pbar:
        net.train(); rl=0.0; nb=0
        for xb,yb in dl:
            xb,yb=xb.to(dev),yb.to(dev)
            opt.zero_grad(); loss=lossf(net(xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step(); rl+=float(loss.item()); nb+=1
        sch.step()
        if ep>=EVAL_START and (ep%EVAL_EVERY==0 or ep==EP-1):
            net.eval()
            with torch.no_grad():
                pv=np.concatenate([torch.softmax(net(Xva_t[i:i+256].to(dev)),1).cpu().numpy() for i in range(0,len(Xva_t),256)])
            mf=f1_score(yva,pv.argmax(1),average="macro",labels=range(NC),zero_division=0)
            if mf>best+1e-5:
                best=mf; best_ep=ep; best_va=pv; no_improve=0
                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                with torch.no_grad():
                    best_te=np.concatenate([torch.softmax(net(Xte_t[i:i+256].to(dev)),1).cpu().numpy() for i in range(0,len(Xte_t),256)])
            else: no_improve+=1
            pbar.set_postfix(loss=f"{rl/max(nb,1):.3f}",val_mF1=f"{mf:.4f}",best=f"{best:.4f}@{best_ep}",stop=f"{no_improve}/{PATIENCE}")
            if no_improve>=PATIENCE: pbar.close(); break
    pbar.close()
    return best_va,best_te,best,best_ep

SEEDS=[1,7,13]
oof=np.zeros((N_,NC),np.float64); test=np.zeros((len(Xte),NC),np.float64)
t0=time.time(); ntr=5*len(SEEDS); done=0
for f in range(5):
    va_acc=np.zeros((np.sum(fold_of==f),NC)); te_acc=np.zeros((len(Xte),NC)); bests=[]
    for sd in SEEDS:
        bva,bte,bf,bep=run_fold(f,sd,desc=f"fold{f} seed{sd}")
        va_acc+=bva; te_acc+=bte; bests.append(bf); done+=1
        el=time.time()-t0; eta=el/done*(ntr-done)
        print(f"  [{done:2d}/{ntr}] fold{f} seed{sd}: best-val mF1 {bf:.4f} @ep{bep} | elapsed {el/60:.1f}m ETA {eta/60:.1f}m",flush=True)
    va_acc/=len(SEEDS); te_acc/=len(SEEDS); oof[fold_of==f]=va_acc; test+=te_acc/5
    print(f"fold {f} DONE: mean best-val mF1 {np.mean(bests):.4f}  elapsed {(time.time()-t0)/60:.1f}m",flush=True)
oof/=oof.sum(1,keepdims=True); test/=test.sum(1,keepdims=True)
np.savez_compressed(C+"div_ssm.npz",oof=oof.astype("float32"),test=test.astype("float32"),
                    y_oof=y.astype("int64"),file_id_test=file_id_test.astype("int64"))
print("saved cache/div_ssm.npz  raw OOF mF1 %.4f  total %.0fs"%(
    f1_score(y,oof.argmax(1),average='macro',zero_division=0),time.time()-t0),flush=True)
print("DONE_SSM",flush=True)
