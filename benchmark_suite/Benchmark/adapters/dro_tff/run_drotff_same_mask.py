from __future__ import annotations
import argparse, sys, time, random
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch


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
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True)
    p.add_argument("--case",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--max-iter",type=int,default=4000)
    p.add_argument("--lr",type=float,default=0.002)
    p.add_argument("--beta",type=float,default=0.7)
    p.add_argument("--omega",type=int,default=2)
    p.add_argument("--layers",type=int,default=3)
    p.add_argument("--gamma",type=float,default=0.0) # official comment: zero for video
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--checkpoint-every",type=int,default=500)
    p.add_argument("--no-resume",action="store_true")
    args=p.parse_args()

    repo=Path(args.repo).resolve()
    sys.path.insert(0,str(repo))
    from model.drotff import DRO_TFF_video

    D=sio.loadmat(args.case, variable_names=["observed","mask"])
    obs=np.asarray(D["observed"],dtype=np.float32)
    mask=np.asarray(D["mask"],dtype=np.float32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda":
        raise RuntimeError("DRO-TFF official model is intended for CUDA; GPU not detected.")

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

    X=torch.tensor(obs,device=device)
    M=torch.tensor(mask,device=device)

    # Time model construction + optimization + final reconstruction; exclude file I/O.
    torch.cuda.synchronize()
    t0=time.perf_counter()

    r=int(X.shape[0]*args.beta)
    model=DRO_TFF_video(X.shape[0],X.shape[1],X.shape[2],
                        r,args.omega,args.layers).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-7)

    ckpt=Path(args.out).with_suffix(".checkpoint.pt")
    start=0
    if ckpt.exists() and not args.no_resume:
        c=_unicode_safe_torch_load(ckpt,map_location=device)
        model.load_state_dict(c["model"]); optimizer.load_state_dict(c["optimizer"])
        start=int(c["iter"])
        print(f"[DRO-TFF] RESUME {start}/{args.max_iter}")
    for it in range(start,args.max_iter):
        out=model()
        loss=torch.norm(out*M-X*M,2)
        # Match official video setting: gamma=0 by default.
        if args.gamma != 0:
            loss=loss + args.gamma*torch.norm(out[1:]-out[:-1],1)
            loss=loss + args.gamma*torch.norm(out[:,1:]-out[:,:-1],1)
            dz=out[:,:,1:]-out[:,:,:-1]
            loss=loss + 10*args.gamma*torch.norm(dz[1:,:,1:]-dz[:-1,:,1:],1)
            loss=loss + 10*args.gamma*torch.norm(dz[:,1:,1:]-dz[:,:-1,1:],1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step=it+1
        if step % 500 == 0 or step==args.max_iter:
            print(f"[DRO-TFF] iter={step} observed-loss={float(loss.detach().cpu()):.6g}")
        if args.checkpoint_every>0 and (step%args.checkpoint_every==0 or step==args.max_iter):
            Path(args.out).parent.mkdir(parents=True,exist_ok=True)
            _unicode_safe_torch_save({"iter":step,"model":model.state_dict(),
                        "optimizer":optimizer.state_dict()},ckpt)

    with torch.no_grad():
        out=model()
        recon=out*(1-M)+X*M

    torch.cuda.synchronize()
    runtime=time.perf_counter()-t0

    recon_np=recon.detach().cpu().numpy().astype(np.float32)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    sio.savemat(args.out,{"recon":recon_np,"runtime_seconds":runtime,
                          "iterations":args.max_iter},do_compression=True)
    if ckpt.exists():
        try: ckpt.unlink()
        except Exception: pass
    print(f"DRO-TFF done: {runtime:.3f} s")

if __name__=="__main__":
    main()
