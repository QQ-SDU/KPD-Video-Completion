#!/usr/bin/env python
from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from torch import nn, optim
import torch.nn.functional as F

class PermuteChange(nn.Module):
    def __init__(self,*dims):
        super().__init__(); self.dims=tuple(dims)
    def forward(self,x):
        return x.permute(*self.dims)

class DiagonalS(nn.Module):
    def __init__(self,r,t):
        super().__init__(); self.r=r; self.t=t
    def forward(self,s):
        # s: [T,R]
        return torch.diag_embed(s)

class OrthLinear(nn.Module):
    """Householder-product orthogonal transform following the official OTLRM video notebook."""
    def __init__(self,n_dim):
        super().__init__()
        self.n_dim=int(n_dim)
        self.u=nn.Parameter(torch.randn(self.n_dim,self.n_dim))
        self.I=nn.Parameter(torch.eye(self.n_dim).unsqueeze(0),requires_grad=False)
    def get_weight(self):
        u=F.normalize(self.u,dim=1)
        hs=self.I - 2*u.unsqueeze(-1)@u.unsqueeze(1)
        mats=[x.squeeze(0) for x in hs.chunk(self.n_dim,dim=0)]
        # official notebook uses torch.chain_matmul
        return torch.chain_matmul(*mats)
    def forward(self,x):
        return F.linear(x,self.get_weight())

class SVDNet(nn.Module):
    def __init__(self,n1,n2,n3,shrink):
        super().__init__()
        r=max(1,n2//int(shrink))
        self.rank=r
        self.A=nn.Parameter(torch.empty(n1,r,n3))
        self.s=nn.Parameter(torch.empty(r,n3))
        self.B=nn.Parameter(torch.empty(n2,r,n3))
        self.net_t=nn.Sequential(
            PermuteChange(1,2,0),
            OrthLinear(n3),
            nn.Linear(n3,n3,bias=False),
        )
        self.netA=nn.Sequential(OrthLinear(n3),PermuteChange(2,0,1))
        self.netB=nn.Sequential(OrthLinear(n3),PermuteChange(2,0,1))
        self.diag=DiagonalS(r,n3)
        self.adaptive=nn.Sequential(
            nn.Linear(r,r,bias=False), nn.LeakyReLU(),
            nn.Linear(r,r,bias=False)
        )
        self.reset_parameters()
    def reset_parameters(self):
        stdv=1.0/np.sqrt(max(1,self.A.size(2)))
        self.A.data.uniform_(-stdv,stdv)
        self.B.data.uniform_(-stdv,stdv)
        self.s.data.uniform_(-stdv,stdv)
    def forward(self):
        Ahat=self.netA(self.A)               # [T,H,R]
        BhatT=self.netB(self.B).permute(0,2,1) # [T,R,W]
        D=self.diag(self.adaptive(self.s.t())) # [T,R,R]
        x=torch.matmul(torch.matmul(Ahat,D),BhatT) # [T,H,W]
        out=self.net_t(x)                    # [H,W,T]
        return out,Ahat,BhatT


def _unicode_safe_torch_save(obj, path):
    path=Path(path) if 'Path' in globals() else _Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f:
        torch.save(obj,f)
        f.flush()
    tmp.replace(path)

def _unicode_safe_torch_load(path, map_location):
    path=Path(path) if 'Path' in globals() else _Path(path)
    with path.open("rb") as f:
        return torch.load(f,map_location=map_location)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--case",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--max-iter",type=int,default=6000)
    ap.add_argument("--lr",type=float,default=0.002)
    ap.add_argument("--shrink",type=int,default=6)
    ap.add_argument("--gamma",type=float,default=2e-8)
    ap.add_argument("--weight-decay",type=float,default=1e-7)
    ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--checkpoint-every",type=int,default=500)
    ap.add_argument("--no-resume",action="store_true")
    ap.add_argument("--device",default="cuda")
    a=ap.parse_args()

    # Deliberately do NOT load GT.
    D=sio.loadmat(a.case,variable_names=["observed","mask"])
    obs=np.asarray(D["observed"],dtype=np.float32)
    mask=np.asarray(D["mask"]).astype(np.float32)

    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("OTLRM formal runner requires CUDA.")
    dev=torch.device(a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if dev.type=="cuda":
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.benchmark=True

    X=torch.tensor(obs,device=dev)
    M=torch.tensor(mask,device=dev)
    model=SVDNet(obs.shape[0],obs.shape[1],obs.shape[2],a.shrink).to(dev)
    params=list(model.parameters())
    optimizer=optim.Adam(params,lr=a.lr,weight_decay=a.weight_decay)
    mse=nn.MSELoss()

    # Official notebook's H_k is params[5], which is net_t Linear weight
    # under its parameter ordering. We address it explicitly.
    H_k=model.net_t[2].weight

    out_path=Path(a.out)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    ckpt=out_path.with_suffix(".checkpoint.pt")
    start=0
    if ckpt.exists() and not a.no_resume:
        c=_unicode_safe_torch_load(ckpt,map_location=dev)
        model.load_state_dict(c["model"]); optimizer.load_state_dict(c["optimizer"])
        start=int(c["iter"])
        print(f"[OTLRM] RESUME {start}/{a.max_iter}")

    if dev.type=="cuda": torch.cuda.synchronize()
    t0=time.perf_counter()

    for it in range(start,a.max_iter):
        pred,Ahat,BhatT=model()
        loss_f=mse(pred*M,X*M)
        loss_A=a.gamma*torch.norm(Ahat[:,1:,:]-Ahat[:,:-1,:],1)
        loss_B=a.gamma*torch.norm(BhatT[:,:,1:]-BhatT[:,:,:-1],1)
        loss_H=a.gamma*torch.norm(H_k[1:,:]-H_k[:-1,:],1)
        loss=loss_f+loss_A+loss_B+loss_H
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        step=it+1
        if step%500==0 or step==a.max_iter:
            print(f"[OTLRM] iter={step} observed-loss={float(loss.detach().cpu()):.6g}")
        if a.checkpoint_every>0 and (step%a.checkpoint_every==0 or step==a.max_iter):
            _unicode_safe_torch_save({"iter":step,"model":model.state_dict(),
                        "optimizer":optimizer.state_dict()},ckpt)

    with torch.no_grad():
        pred,_,_=model()
        recon=pred*(1-M)+X*M
    if dev.type=="cuda": torch.cuda.synchronize()
    runtime=time.perf_counter()-t0

    sio.savemat(out_path,{
        "recon":recon.detach().cpu().numpy().astype(np.float32),
        "runtime_seconds":float(runtime),
        "iterations":int(a.max_iter),
        "shrink":int(a.shrink),
        "gamma":float(a.gamma),
        "rank_internal":int(model.rank),
    },do_compression=True)
    if ckpt.exists():
        try: ckpt.unlink()
        except Exception: pass
    print(f"OTLRM done: {runtime:.3f} s; internal rank={model.rank}")

if __name__=="__main__":
    main()
