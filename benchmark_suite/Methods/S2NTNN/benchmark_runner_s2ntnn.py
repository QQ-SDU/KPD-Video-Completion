# import torchpwl (package of piecewise linear unit)
import torch
from torch import nn, optim 
from torch.autograd import Variable 
from torch.utils.data import DataLoader 
import os 
from utils import * 
# notebook magic removed
import numpy as np 
import scipy.io as scio
import matplotlib.pyplot as plt 
import scipy.io

dtype = torch.cuda.FloatTensor


def _unicode_safe_torch_save(obj, path):
    path=Path(path) if 'Path' in globals() else _Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f:
        _torch.save(obj,f)
        f.flush()
    tmp.replace(path)

def _unicode_safe_torch_load(path, map_location):
    path=Path(path) if 'Path' in globals() else _Path(path)
    with path.open("rb") as f:
        return _torch.load(f,map_location=map_location)

class NoFC_3(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(NoFC_3, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_channel,out_channel, bias = False),
                                   nn.LeakyReLU())
        
    def forward(self,x):
        return self.layer(x)

class SSNT(nn.Module): 
    def __init__(self,n_4,n_3):
        super(SSNT, self).__init__()
        self.f = nn.Sequential(  NoFC_3(n_3,n_4),
                                NoFC_3(n_4,n_4))
                                    
        self.g = nn.Sequential(NoFC_3(n_4,n_4),
                               NoFC_3(n_4,n_3))
                                    
    def forward(self, x):
        f_x = self.f(x)
        g_x = self.g(f_x)
        return f_x, g_x

# ===== SameMask benchmark runner appended by build_s2ntnn_runner.py =====
import argparse as _argparse
import scipy.io as _sio
import numpy as _np
import torch as _torch
import time as _time
from pathlib import Path as _Path

def _benchmark_main():
    p=_argparse.ArgumentParser()
    p.add_argument("--case",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--max-iter",type=int,default=7000)
    p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--lambda-nuc",type=float,default=2e-7)
    p.add_argument("--n4-mult",type=float,default=2.0)
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--checkpoint-every",type=int,default=500)
    p.add_argument("--no-resume",action="store_true")
    a=p.parse_args()

    D=_sio.loadmat(a.case, variable_names=["observed","mask","init_linear"])
    obs=_np.asarray(D["observed"],dtype=_np.float32)
    m=_np.asarray(D["mask"],dtype=_np.float32)
    xinit=_np.asarray(D["init_linear"],dtype=_np.float32)

    dev=_torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    if dev.type!="cuda":
        raise RuntimeError("S2NTNN benchmark runner expects CUDA.")
    _torch.manual_seed(a.seed); _torch.cuda.manual_seed_all(a.seed)

    X=_torch.tensor(obs,device=dev)
    M=_torch.tensor(m,device=dev)
    Xin=_torch.tensor(xinit,device=dev)  # fixed for video, as noted by official demo

    _torch.cuda.synchronize(); t0=_time.perf_counter()
    n4=max(1,int(round(a.n4_mult*obs.shape[2])))
    model=SSNT(n4,obs.shape[2]).to(dev)
    opt=_torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=0)
    mse=_torch.nn.MSELoss()

    ckpt=_Path(a.out).with_suffix(".checkpoint.pt")
    start=0
    if ckpt.exists() and not a.no_resume:
        c=_unicode_safe_torch_load(ckpt,map_location=dev)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"])
        start=int(c["iter"])
        print(f"[S2NTNN] RESUME {start}/{a.max_iter}")
    for it in range(start,a.max_iter):
        xlr,out=model(Xin)
        sl=xlr[:,:,int(it%n4)]
        loss=a.lambda_nuc*_torch.norm(sl,"nuc") + mse(out*M,X*M)
        opt.zero_grad(); loss.backward(); opt.step()
        step=it+1
        if step%500==0 or step==a.max_iter:
            print(f"[S2NTNN] iter={step} observed-loss={float(loss.detach().cpu()):.6g}")
        if a.checkpoint_every>0 and (step%a.checkpoint_every==0 or step==a.max_iter):
            _Path(a.out).parent.mkdir(parents=True,exist_ok=True)
            _unicode_safe_torch_save({"iter":step,"model":model.state_dict(),"optimizer":opt.state_dict()},ckpt)
    with _torch.no_grad():
        _,out=model(Xin)
        recon=out*(1-M)+X*M
    _torch.cuda.synchronize(); runtime=_time.perf_counter()-t0
    _Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    _sio.savemat(a.out,{"recon":recon.cpu().numpy().astype(_np.float32),
                        "runtime_seconds":runtime,"iterations":a.max_iter},
                 do_compression=True)
    if ckpt.exists():
        try: ckpt.unlink()
        except Exception: pass
    print(f"S2NTNN done: {runtime:.3f} s")

if __name__=="__main__":
    _benchmark_main()
