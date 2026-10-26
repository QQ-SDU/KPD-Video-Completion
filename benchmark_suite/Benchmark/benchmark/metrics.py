from __future__ import annotations
import numpy as np

def psnr(a, b, data_range=1.0, mask=None):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    d=(a-b)**2
    if mask is not None:
        m=np.asarray(mask,dtype=bool)
        if not np.any(m):
            return float("nan")
        mse=float(d[m].mean())
    else:
        mse=float(d.mean())
    if mse <= 0:
        return float("inf")
    return 10.0*np.log10((data_range**2)/mse)

def framewise_ssim(gt, rec, data_range=1.0):
    from skimage.metrics import structural_similarity
    vals=[]
    for k in range(gt.shape[2]):
        vals.append(structural_similarity(gt[:,:,k], rec[:,:,k], data_range=data_range))
    return float(np.mean(vals)), float(np.std(vals))

def temporal_gradient_mae(gt, rec, mask=None):
    dg=np.diff(gt,axis=2); dr=np.diff(rec,axis=2)
    e=np.abs(dg-dr)
    if mask is None:
        return float(e.mean())
    # Tube-wise mask is constant in t; evaluate missing tube temporal gradients.
    m=np.asarray(mask[:,:,:-1],dtype=bool)
    return float(e[m].mean()) if np.any(m) else float("nan")
