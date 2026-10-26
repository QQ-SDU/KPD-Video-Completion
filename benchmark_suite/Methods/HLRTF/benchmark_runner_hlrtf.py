import torch
from torch import nn, optim 
from torch.autograd import Variable 
import os 
from utils import * 
# notebook magic removed
import numpy as np 
import scipy.io as scio
import matplotlib.pyplot as plt 
import scipy.io
import math
import time

expand_x = 1; n_3 = 500; 

#for three channel RGB images, set "expand" as a big value (e.g., expand=10)
expand = 1

shrink = 5

#gamma is an hyperparameter

gamma = 0.000000001
data_all = ["data/Pavia"]
c_all = ["SR1"]

max_iter = 4000
lr_real = 0.003


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

class Y_net(nn.Module): 
    def __init__(self,n_1,n_2,n_3):
        super(Y_net, self).__init__()
        self.A_hat = nn.Parameter(torch.Tensor(n_3*expand,n_1,n_2//shrink))
        self.B_hat = nn.Parameter(torch.Tensor(n_3*expand,n_2//shrink,n_2))
        
        self.net = nn.Sequential(permute_change(1,2,0),
                                 nn.Linear(int(n_3*expand),int(n_3*expand),bias = False),
                                 nn.LeakyReLU(),
                                 nn.Linear(int(n_3*expand),n_3,bias = False))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.A_hat.size(0))
        self.A_hat.data.uniform_(-stdv, stdv)
        self.B_hat.data.uniform_(-stdv, stdv)
                                   
    def forward(self):
        x = torch.matmul(self.A_hat,self.B_hat)
        return self.net(x)

# ===== SameMask benchmark runner appended by build_hlrtf_runner.py =====
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
    p.add_argument("--max-iter",type=int,default=4000)
    p.add_argument("--lr",type=float,default=0.003)
    p.add_argument("--gamma",type=float,default=1e-9)
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--shrink",type=int,default=5)
    p.add_argument("--checkpoint-every",type=int,default=500)
    p.add_argument("--no-resume",action="store_true")
    a=p.parse_args()

    D=_sio.loadmat(a.case, variable_names=["observed","mask"])
    obs=_np.asarray(D["observed"],dtype=_np.float32)
    m=_np.asarray(D["mask"],dtype=_np.float32)
    dev=_torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    if dev.type!="cuda":
        raise RuntimeError("HLRTF benchmark runner expects CUDA.")

    _torch.manual_seed(a.seed); _torch.cuda.manual_seed_all(a.seed)
    X=_torch.tensor(obs,device=dev)
    M=_torch.tensor(m,device=dev)

    _torch.cuda.synchronize(); t0=_time.perf_counter()
    global shrink
    shrink=int(a.shrink)
    model=Y_net(X.shape[0],X.shape[1],X.shape[2]).to(dev)
    params=list(model.parameters())
    opt=_torch.optim.Adam(params,lr=a.lr,weight_decay=1e-7)
    mse=_torch.nn.MSELoss()

    ckpt=_Path(a.out).with_suffix(".checkpoint.pt")
    start=0
    if ckpt.exists() and not a.no_resume:
        c=_unicode_safe_torch_load(ckpt,map_location=dev)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"])
        start=int(c["iter"])
        print(f"[HLRTF] RESUME {start}/{a.max_iter}")
    for it in range(start,a.max_iter):
        out=model()
        loss=mse(out*M,X*M)
        for idx,q in enumerate(params, start=1):
            if idx==1:
                loss=loss+a.gamma*_torch.norm(q[:,1:,:]-q[:,:-1,:],1)
            elif idx==4:
                loss=loss+a.gamma*_torch.norm(q[1:,:]-q[:-1,:],1)
            elif idx==2:
                loss=loss+a.gamma*_torch.norm(q[:,:,1:]-q[:,:,:-1],1)
        opt.zero_grad(); loss.backward(); opt.step()
        step=it+1
        if step%500==0 or step==a.max_iter:
            print(f"[HLRTF] iter={step} observed-loss={float(loss.detach().cpu()):.6g}")
        if a.checkpoint_every>0 and (step%a.checkpoint_every==0 or step==a.max_iter):
            _Path(a.out).parent.mkdir(parents=True,exist_ok=True)
            _unicode_safe_torch_save({"iter":step,"model":model.state_dict(),"optimizer":opt.state_dict()},ckpt)
    with _torch.no_grad():
        out=model()
        recon=out*(1-M)+X*M
    _torch.cuda.synchronize(); runtime=_time.perf_counter()-t0
    _Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    _sio.savemat(a.out,{"recon":recon.cpu().numpy().astype(_np.float32),
                        "runtime_seconds":runtime,"iterations":a.max_iter},
                 do_compression=True)
    if ckpt.exists():
        try: ckpt.unlink()
        except Exception: pass
    print(f"HLRTF done: {runtime:.3f} s")

if __name__=="__main__":
    _benchmark_main()
