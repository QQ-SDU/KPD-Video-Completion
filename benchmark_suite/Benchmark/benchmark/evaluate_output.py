from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import scipy.io as sio
from .metrics import psnr, framewise_ssim, temporal_gradient_mae

def load_case(path):
    p = Path(path)
    if p.suffix.lower() == ".npz":
        d = np.load(p)
        return d["gt"].astype(np.float32), d["observed"].astype(np.float32), d["mask"].astype(bool)
    d = sio.loadmat(p)
    return d["gt"].astype(np.float32), d["observed"].astype(np.float32), d["mask"].astype(bool)

def _decode_hdf5_mat(path, mat_key=None):
    import h5py
    with h5py.File(path, "r") as f:
        keys = [mat_key] if mat_key else ["recon","Xhat","x_recon","X_save","X_Out","result"]
        key = next((k for k in keys if k and k in f), None)
        if key is None:
            cand=[]
            for k,obj in f.items():
                if isinstance(obj,h5py.Dataset) and len(obj.shape)==3:
                    cand.append(k)
            if not cand:
                raise KeyError(f"No 3-D reconstruction dataset in {path}. Keys={list(f.keys())}")
            key=cand[0]

        raw=f[key][()]
        # MATLAB v7.3 arrays are stored with reversed dimension order.
        axes=tuple(reversed(range(raw.ndim)))

        if raw.dtype.fields and "real" in raw.dtype.fields:
            x=raw["real"] + 1j*raw["imag"]
            x=np.transpose(x,axes)
        else:
            x=np.transpose(raw,axes)

        runtime=None
        for rk in ["runtime_seconds","runtime","Time","time_cost"]:
            if rk in f:
                try:
                    runtime=float(np.asarray(f[rk][()]).squeeze())
                    break
                except Exception:
                    pass
    return x, runtime

def load_recon(path, mat_key=None):
    p=Path(path); ext=p.suffix.lower()
    runtime=None
    if ext==".npy":
        x=np.load(p)
    elif ext==".npz":
        d=np.load(p)
        key=mat_key or ("recon" if "recon" in d else d.files[0])
        x=d[key]
        if "runtime_seconds" in d:
            runtime=float(np.asarray(d["runtime_seconds"]).squeeze())
    elif ext==".mat":
        try:
            d=sio.loadmat(p)
            keys=[mat_key] if mat_key else ["recon","Xhat","x_recon","X_save","X_Out","result"]
            key=next((k for k in keys if k and k in d),None)
            if key is None:
                cand=[k for k,v in d.items()
                      if not k.startswith("__") and isinstance(v,np.ndarray) and v.ndim==3]
                if not cand:
                    raise KeyError(f"No 3-D reconstruction array in {p}. Keys={list(d)}")
                key=cand[0]
            x=d[key]
            for rk in ["runtime_seconds","runtime","Time","time_cost"]:
                if rk in d:
                    try:
                        runtime=float(np.asarray(d[rk]).squeeze())
                        break
                    except Exception:
                        pass
        except NotImplementedError:
            x,runtime=_decode_hdf5_mat(p,mat_key)
    else:
        raise ValueError("recon must be .npy/.npz/.mat")

    x=np.asarray(x)
    # Official SALTS demo evaluates abs(YY); applying abs to a complex reconstruction
    # is also safer than silently dropping its imaginary component.
    if np.iscomplexobj(x):
        x=np.abs(x)
    return x.astype(np.float32), runtime

def align_shape(x, target):
    if x.shape==target:
        return x
    if x.ndim==3 and (x.shape[2],x.shape[0],x.shape[1])==target:
        return np.transpose(x,(1,2,0))
    raise ValueError(f"Recon shape {x.shape} does not match target {target}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",required=True)
    p.add_argument("--recon",required=True)
    p.add_argument("--method",required=True)
    p.add_argument("--mat-key",default=None)
    p.add_argument("--force-dc",action="store_true")
    p.add_argument("--no-clip",action="store_true")
    p.add_argument("--results-dir",default="results")
    args=p.parse_args()

    gt,obs,mask=load_case(args.case)
    rec,runtime=load_recon(args.recon,args.mat_key)
    rec=align_shape(rec,gt.shape)
    if not args.no_clip:
        rec=np.clip(rec,0,1)
    raw_dc_mae=float(np.abs(rec[mask]-obs[mask]).mean()) if np.any(mask) else 0.0
    raw_dc_max=float(np.abs(rec[mask]-obs[mask]).max()) if np.any(mask) else 0.0
    if args.force_dc:
        rec=rec.copy()
        rec[mask]=obs[mask]

    miss=~mask
    ssim_mean,ssim_std=framewise_ssim(gt,rec,1.0)
    row={
        "method":args.method,
        "case":Path(args.case).stem,
        "shape":list(gt.shape),
        "psnr_full":psnr(gt,rec,1.0),
        "ssim_frame_mean":ssim_mean,
        "ssim_frame_std":ssim_std,
        "psnr_missing":psnr(gt,rec,1.0,mask=miss),
        "mae_missing":float(np.abs(gt[miss]-rec[miss]).mean()),
        "temporal_gradient_mae_missing":temporal_gradient_mae(gt,rec,mask=miss),
        "observed_dc_mae_before_optional_force":raw_dc_mae,
        "observed_dc_max_before_optional_force":raw_dc_max,
        "runtime_seconds":runtime,
        "force_dc_in_evaluator":bool(args.force_dc),
    }
    outdir=Path(args.results_dir); outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/f"metrics_{args.method}_{Path(args.case).stem}.json"
    out.write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(row,indent=2,ensure_ascii=False))
    print(f"Saved: {out}")

if __name__=="__main__":
    main()
