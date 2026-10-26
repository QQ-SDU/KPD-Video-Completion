from __future__ import annotations
import argparse, math, random, time
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from torch import nn, optim

class SineLayer(nn.Module):
    def __init__(self,in_features,out_features,omega_0=2.0,is_first=False):
        super().__init__(); self.omega_0=float(omega_0); self.is_first=is_first
        self.in_features=int(in_features); self.linear=nn.Linear(in_features,out_features)
        self.init_weights()
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1/self.in_features,1/self.in_features)
            else:
                b=np.sqrt(6/self.in_features)/self.omega_0
                self.linear.weight.uniform_(-b,b)
    def forward(self,x): return torch.sin(self.omega_0*self.linear(x))

class Network(nn.Module):
    def __init__(self,r1,r2,r3,mid_channel,omega):
        super().__init__()
        def net(r):
            return nn.Sequential(SineLayer(1,mid_channel,omega,is_first=True),
                                 SineLayer(mid_channel,mid_channel,omega,is_first=True),
                                 nn.Linear(mid_channel,r))
        self.U_net=net(r1); self.V_net=net(r2); self.W_net=net(r3)
    def forward(self,centre,U_input,V_input,W_input):
        U=self.U_net(U_input); V=self.V_net(V_input); W=self.W_net(W_input)
        c=centre.permute(1,2,0); c=c@U.t()
        c=c.permute(2,1,0); c=c@V.t()
        c=c.permute(0,2,1); c=c@W.t()
        return c


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
    ap.add_argument("--case",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--max-iter",type=int,default=3001)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--weight-decay",type=float,default=3.0)
    ap.add_argument("--omega",type=float,default=2.0)
    ap.add_argument("--down",default="2,2,1")
    ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--checkpoint-every",type=int,default=500)
    ap.add_argument("--no-resume",action="store_true")
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("LRTFR formal runner requires CUDA.")
    D=sio.loadmat(a.case,variable_names=["observed","mask"])
    obs=np.asarray(D["observed"],dtype=np.float32)
    mask=np.asarray(D["mask"],dtype=np.float32)
    n1,n2,n3=obs.shape
    down=[int(x) for x in a.down.split(",")]
    r1=max(1,n1//down[0]); r2=max(1,n2//down[1]); r3=max(1,n3//down[2])

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev=torch.device("cuda")
    X=torch.tensor(obs,device=dev); M=torch.tensor(mask,device=dev)
    mid=int(n2)
    centre=torch.empty(r1,r2,r3,device=dev); centre.uniform_(-1/math.sqrt(r1),1/math.sqrt(r1))
    centre.requires_grad_(True)
    U_input=torch.arange(1,n1+1,device=dev,dtype=torch.float32).reshape(n1,1)
    V_input=torch.arange(1,n2+1,device=dev,dtype=torch.float32).reshape(n2,1)
    W_input=torch.arange(1,n3+1,device=dev,dtype=torch.float32).reshape(n3,1)
    model=Network(r1,r2,r3,mid,a.omega).to(dev)
    params=list(model.parameters())+[centre]
    optimizer=optim.Adam(params,lr=a.lr,weight_decay=a.weight_decay)

    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    ckpt=out.with_suffix(".checkpoint.pt"); start=0; elapsed0=0.0
    if ckpt.exists() and not a.no_resume:
        c=_unicode_safe_torch_load(ckpt,map_location=dev)
        model.load_state_dict(c["model"]); optimizer.load_state_dict(c["optimizer"])
        centre.data.copy_(c["centre"].to(dev)); start=int(c["iter"]); elapsed0=float(c.get("elapsed",0))
        print(f"[LRTFR] RESUME {start}/{a.max_iter}")

    torch.cuda.synchronize(); t0=time.perf_counter()
    for it in range(start,a.max_iter):
        pred=model(centre,U_input,V_input,W_input)
        loss=torch.norm(pred*M-X*M,2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        step=it+1
        if step%500==0 or step==a.max_iter:
            print(f"[LRTFR] iter={step} observed-loss={float(loss.detach().cpu()):.6g}")
        if a.checkpoint_every>0 and (step%a.checkpoint_every==0 or step==a.max_iter):
            torch.cuda.synchronize(); elapsed=elapsed0+time.perf_counter()-t0
            _unicode_safe_torch_save({"iter":step,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                        "centre":centre.detach().cpu(),"elapsed":elapsed},ckpt)
    with torch.no_grad():
        pred=model(centre,U_input,V_input,W_input)
        recon=torch.clamp(pred,0,1)*(1-M)+X*M
    torch.cuda.synchronize(); runtime=elapsed0+time.perf_counter()-t0
    sio.savemat(out,{"recon":recon.detach().cpu().numpy().astype(np.float32),
                     "runtime_seconds":float(runtime),"iterations":int(a.max_iter),
                     "down":np.array(down),"omega":float(a.omega)},do_compression=True)
    if ckpt.exists(): ckpt.unlink()
    print(f"LRTFR done: {runtime:.3f} s")

if __name__=="__main__": main()
