#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None
try:
    from scipy.ndimage import uniform_filter
except Exception:
    uniform_filter = None
try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

EPS = 1e-12

@dataclass
class Rank1Result:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    error: float
    sweeps: int

@dataclass
class KPDModel:
    components: List[Rank1Result]
    reconstruction: np.ndarray
    residual: np.ndarray
    residual_norms: List[float]


def outer3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.einsum('i,j,k->ijk', a, b, c, optimize=True)


def clsa_rank1(tensor: np.ndarray, max_sweeps: int=50, tol: float=1e-6,
               restarts: int=2, seed: int=0) -> Rank1Result:
    """CLSA-style d=3 nearest Kronecker product solver."""
    x=np.asarray(tensor,dtype=np.float64)
    H,W,T=x.shape
    rng=np.random.default_rng(seed)
    best=None
    for _ in range(max(1,restarts)):
        a=rng.normal(size=H)
        b=rng.normal(size=W)
        c=rng.normal(size=T)
        b/=np.linalg.norm(b)+EPS
        c/=np.linalg.norm(c)+EPS
        prev=outer3(a,b,c)
        sweeps_done=0
        for sweep in range(1,max_sweeps+1):
            sweeps_done=sweep
            a=np.einsum('ijk,j,k->i',x,b,c,optimize=True)/((b@b)*(c@c)+EPS)
            b=np.einsum('ijk,i,k->j',x,a,c,optimize=True)/((a@a)*(c@c)+EPS)
            c=np.einsum('ijk,i,j->k',x,a,b,optimize=True)/((a@a)*(b@b)+EPS)
            nb=np.linalg.norm(b)+EPS
            nc=np.linalg.norm(c)+EPS
            b/=nb; c/=nc; a*=nb*nc
            cur=outer3(a,b,c)
            change=np.linalg.norm(cur-prev)/(np.linalg.norm(prev)+EPS)
            prev=cur
            if change < tol:
                break
        err=float(np.linalg.norm(x-prev))
        rr=Rank1Result(a.copy(),b.copy(),c.copy(),err,sweeps_done)
        if best is None or rr.error < best.error:
            best=rr
    return best


def finite_sum_clsa(tensor: np.ndarray, rank: int, max_sweeps: int=50,
                    tol: float=1e-6, restarts: int=2, seed: int=0,
                    verbose: bool=True) -> KPDModel:
    x=np.asarray(tensor,dtype=np.float64)
    residual=x.copy(); recon=np.zeros_like(x); comps=[]
    residual_norms=[float(np.linalg.norm(residual))]
    xnorm=np.linalg.norm(x)+EPS
    for k in range(rank):
        comp=clsa_rank1(residual,max_sweeps,tol,restarts,seed+1009*k)
        term=outer3(comp.a,comp.b,comp.c)
        recon += term
        residual -= term
        comps.append(comp)
        residual_norms.append(float(np.linalg.norm(residual)))
        if verbose:
            print(f'    KPD component {k+1:02d}/{rank}: sweeps={comp.sweeps:02d}, '
                  f'relative residual={residual_norms[-1]/xnorm:.6f}')
    return KPDModel(comps,recon,residual,residual_norms)


def parse_resize(text: str) -> Optional[Tuple[int,int]]:
    if text.lower() in {'none','native','0'}:
        return None
    p=text.lower().replace('x',',').split(',')
    if len(p)!=2: raise ValueError('--resize must look like 72x88 or none')
    return int(p[0]),int(p[1])


def _resize_gray(frame: np.ndarray, size_hw: Optional[Tuple[int,int]]) -> np.ndarray:
    if size_hw is None:
        return frame

    H, W = size_hw

    # Prefer OpenCV when available.
    if cv2 is not None:
        return cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

    # Pillow fallback so Y4M experiments do not require OpenCV.
    if Image is not None:
        arr = np.asarray(frame, dtype=np.float64)
        arr8 = np.uint8(np.clip(arr, 0.0, 1.0) * 255.0)
        img = Image.fromarray(arr8)
        img = img.resize((W, H), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.float64) / 255.0

    raise RuntimeError(
        "Resizing requires either opencv-python or Pillow. "
        "Install one with: pip install opencv-python  OR  pip install Pillow"
    )


def load_raw_yuv420(path: Path, width: int, height: int, max_frames: int,
                    start_frame: int=0, resize_hw: Optional[Tuple[int,int]]=None):
    """Read Y plane from planar YUV420p. Returns H x W x T grayscale tensor."""
    if width%2 or height%2:
        raise ValueError('YUV420 requires even width and height')
    ysize=width*height
    frame_bytes=ysize + 2*(width//2)*(height//2)
    frames=[]
    with path.open('rb') as f:
        f.seek(start_frame*frame_bytes)
        for _ in range(max_frames):
            y=f.read(ysize)
            if len(y)<ysize: break
            frame=np.frombuffer(y,dtype=np.uint8).reshape(height,width).astype(np.float64)/255.0
            frame=_resize_gray(frame,resize_hw)
            frames.append(frame)
            skip=frame_bytes-ysize
            f.seek(skip,1)
    if not frames: raise RuntimeError(f'No YUV frames loaded from {path}')
    return np.stack(frames,axis=2),25.0



def load_y4m(path: Path, max_frames: int, resize_hw: Optional[Tuple[int,int]]=None,
             start_frame: int=0):
    """
    Read the luma (Y) channel from a YUV4MPEG2 (.y4m) file.

    Y4M stores width, height, frame rate, and chroma metadata in the header,
    so unlike raw .yuv input, no manual dimensions are required.

    Returns H x W x T grayscale tensor in [0,1] and fps.
    """
    frames = []

    with path.open("rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2"):
            raise ValueError(f"{path} is not a valid Y4M file")

        tokens = header.decode("ascii", errors="ignore").strip().split()[1:]
        meta = {}
        for tok in tokens:
            if tok:
                meta[tok[0]] = tok[1:]

        if "W" not in meta or "H" not in meta:
            raise ValueError("Y4M header does not contain W/H dimensions")

        width = int(meta["W"])
        height = int(meta["H"])

        fps = 25.0
        if "F" in meta and ":" in meta["F"]:
            num, den = meta["F"].split(":", 1)
            try:
                den_v = float(den)
                if den_v != 0:
                    fps = float(num) / den_v
            except Exception:
                pass

        chroma = meta.get("C", "420").lower()
        ysize = width * height

        if chroma.startswith("420"):
            chroma_bytes = 2 * ((width + 1)//2) * ((height + 1)//2)
        elif chroma.startswith("422"):
            chroma_bytes = 2 * ((width + 1)//2) * height
        elif chroma.startswith("444"):
            chroma_bytes = 2 * width * height
        elif chroma.startswith("mono"):
            chroma_bytes = 0
        else:
            raise ValueError(
                f"Unsupported Y4M chroma format '{chroma}'. "
                "Supported in this prototype: 420/422/444/mono 8-bit."
            )

        frame_index = 0
        while len(frames) < max_frames:
            frame_header = f.readline()
            if not frame_header:
                break
            if not frame_header.startswith(b"FRAME"):
                raise ValueError(
                    f"Unexpected Y4M frame header near frame {frame_index}: "
                    f"{frame_header[:40]!r}"
                )

            y = f.read(ysize)
            if len(y) < ysize:
                break

            if frame_index >= start_frame:
                frame = (
                    np.frombuffer(y, dtype=np.uint8)
                    .reshape(height, width)
                    .astype(np.float64) / 255.0
                )
                frame = _resize_gray(frame, resize_hw)
                frames.append(frame)

            if chroma_bytes:
                uv = f.read(chroma_bytes)
                if len(uv) < chroma_bytes:
                    break

            frame_index += 1

    if not frames:
        raise RuntimeError(f"No Y4M frames loaded from {path}")

    return np.stack(frames, axis=2), float(fps)


def load_video_file(path: Path, max_frames: int, resize_hw: Optional[Tuple[int,int]], start_frame: int=0):
    if cv2 is None: raise RuntimeError('opencv-python is required for video input')
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open video: {path}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0: fps=25.0
    if start_frame>0: cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    frames=[]
    while len(frames)<max_frames:
        ok,frame=cap.read()
        if not ok: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0
        gray=_resize_gray(gray,resize_hw)
        frames.append(gray)
    cap.release()
    if not frames: raise RuntimeError('No frames loaded')
    return np.stack(frames,axis=2),float(fps)


def load_npy(path: Path, max_frames: int, resize_hw: Optional[Tuple[int,int]], layout: str='HWT'):
    x=np.load(path)
    if x.ndim==4 and x.shape[-1] in (3,4): x=x[...,:3].mean(axis=-1)
    if x.ndim!=3: raise ValueError(f'Expected 3D npy, got {x.shape}')
    if layout.upper()=='THW': x=np.transpose(x,(1,2,0))
    x=x.astype(np.float64)
    if x.max()>1.5: x/=255.0
    x=x[:,:,:max_frames]
    if resize_hw is not None:
        x=np.stack([_resize_gray(x[:,:,t],resize_hw) for t in range(x.shape[2])],axis=2)
    return np.clip(x,0,1),25.0


def load_input(path: Path, max_frames: int=30, resize_hw: Optional[Tuple[int,int]]=(72,88),
               start_frame: int=0, npy_layout: str='HWT', yuv_width: int=176, yuv_height: int=144):
    s=path.suffix.lower()
    if s=='.npy': return load_npy(path,max_frames,resize_hw,npy_layout)
    if s=='.yuv': return load_raw_yuv420(path,yuv_width,yuv_height,max_frames,start_frame,resize_hw)
    if s=='.y4m': return load_y4m(path,max_frames,resize_hw,start_frame)
    return load_video_file(path,max_frames,resize_hw,start_frame)


def make_mask(shape, sampling_rate: float, mask_type: str, seed: int):
    rng=np.random.default_rng(seed)
    H,W,T=shape
    if mask_type=='random':
        return rng.random(shape)<sampling_rate
    if mask_type=='tube':
        m2=rng.random((H,W))<sampling_rate
        return np.repeat(m2[:,:,None],T,axis=2)
    raise ValueError(mask_type)


def mean_fill(observed: np.ndarray, mask: np.ndarray):
    out=observed.copy().astype(np.float64)
    mu=float(out[mask].mean()) if np.any(mask) else 0.5
    out[~mask]=mu
    return out


def smooth_fill(observed: np.ndarray, mask: np.ndarray, iterations: int=12, size=(5,5,3)):
    out=mean_fill(observed,mask)
    if uniform_filter is None: return out
    for _ in range(iterations):
        sm=uniform_filter(out,size=size,mode='nearest')
        out[~mask]=sm[~mask]
        out[mask]=observed[mask]
    return np.clip(out,0,1)


def psnr(gt, rec, data_range=1.0):
    mse=float(np.mean((gt-rec)**2))
    return float('inf') if mse<=1e-15 else 10*math.log10(data_range**2/mse)


def masked_psnr(gt, rec, mask_missing, data_range=1.0):
    if not np.any(mask_missing): return float('inf')
    mse=float(np.mean((gt[mask_missing]-rec[mask_missing])**2))
    return float('inf') if mse<=1e-15 else 10*math.log10(data_range**2/mse)


def mean_ssim(gt, rec):
    if skimage_ssim is None: return float('nan')
    vals=[skimage_ssim(gt[:,:,t],rec[:,:,t],data_range=1.0) for t in range(gt.shape[2])]
    return float(np.mean(vals))


def frame_metrics(gt, rec, observed_mask):
    rows=[]
    for t in range(gt.shape[2]):
        miss=~observed_mask[:,:,t]
        rows.append({
            'frame':t,
            'psnr_all':psnr(gt[:,:,t],rec[:,:,t]),
            'psnr_missing':masked_psnr(gt[:,:,t],rec[:,:,t],miss),
            'ssim_all': mean_ssim(gt[:,:,t:t+1],rec[:,:,t:t+1]),
        })
    return rows


def run_projection_completion(gt, sampling_rate: float, mask_type: str='random', rank: int=5,
                              outer_iters: int=10, max_sweeps: int=40, tol: float=1e-5,
                              restarts: int=2, seed: int=1234, init_method: str='mean', verbose: bool=True):
    mask=make_mask(gt.shape,sampling_rate,mask_type,seed)
    observed=np.zeros_like(gt); observed[mask]=gt[mask]
    mean0=mean_fill(observed,mask)
    smooth0=smooth_fill(observed,mask)
    if init_method=='mean': x=mean0.copy()
    elif init_method=='smooth': x=smooth0.copy()
    else: raise ValueError(init_method)
    init=x.copy()
    history=[]
    miss=~mask
    if verbose:
        print(f'Video shape: {gt.shape}')
        print(f'Observed fraction: {mask.mean():.4f}')
        print(f'Initialization: PSNR(all)={psnr(gt,x):.3f} dB, '
              f'PSNR(missing)={masked_psnr(gt,x,miss):.3f} dB, SSIM={mean_ssim(gt,x):.4f}')
    for it in range(1,outer_iters+1):
        import time
        tic=time.time()
        if verbose: print(f'\n[Outer iteration {it}/{outer_iters}]')
        model=finite_sum_clsa(x,rank,max_sweeps,tol,restarts,seed+100000*it,verbose)
        low=np.clip(model.reconstruction,0,1)
        xnew=low.copy(); xnew[mask]=observed[mask]
        delta=float(np.linalg.norm(xnew-x)/(np.linalg.norm(x)+EPS))
        x=xnew
        row={
            'outer_iter':it,
            'delta':delta,
            'psnr_all':psnr(gt,x),
            'psnr_missing':masked_psnr(gt,x,miss),
            'ssim_all':mean_ssim(gt,x),
            'seconds':time.time()-tic,
            'kpd_relative_residual':float(np.linalg.norm(model.residual)/(np.linalg.norm(x)+EPS)),
        }
        history.append(row)
        if verbose:
            print(f"  delta={row['delta']:.6e}, PSNR(all)={row['psnr_all']:.3f}, "
                  f"PSNR(missing)={row['psnr_missing']:.3f}, SSIM={row['ssim_all']:.4f}, "
                  f"time={row['seconds']:.2f}s")
        if delta<tol: break
    return {
        'recovered':x,'mask':mask,'observed':observed,'initialization':init,
        'mean_fill':mean0,'smooth_fill':smooth0,'history':history,
        'metrics':{
            'sampling_rate_actual':float(mask.mean()),
            'init_psnr_all':psnr(gt,init),'init_psnr_missing':masked_psnr(gt,init,miss),'init_ssim':mean_ssim(gt,init),
            'mean_psnr_all':psnr(gt,mean0),'mean_psnr_missing':masked_psnr(gt,mean0,miss),'mean_ssim':mean_ssim(gt,mean0),
            'smooth_psnr_all':psnr(gt,smooth0),'smooth_psnr_missing':masked_psnr(gt,smooth0,miss),'smooth_ssim':mean_ssim(gt,smooth0),
            'final_psnr_all':psnr(gt,x),'final_psnr_missing':masked_psnr(gt,x,miss),'final_ssim':mean_ssim(gt,x),
        }
    }


def save_video(path: Path, video: np.ndarray, fps: float=25.0, labels: Optional[List[str]]=None):
    if cv2 is None: return False
    # video HxWxT or list-like concatenated externally
    H,W,T=video.shape
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H),True)
    if not writer.isOpened(): return False
    for t in range(T):
        g=np.uint8(np.clip(video[:,:,t],0,1)*255)
        writer.write(cv2.cvtColor(g,cv2.COLOR_GRAY2BGR))
    writer.release(); return True


def save_comparison_video(path: Path, named_videos: List[Tuple[str,np.ndarray]], fps: float=25.0):
    if cv2 is None: return False
    H,W,T=named_videos[0][1].shape
    canvas_w=W*len(named_videos)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(canvas_w,H),True)
    if not writer.isOpened(): return False
    for t in range(T):
        panels=[]
        for name,v in named_videos:
            g=np.uint8(np.clip(v[:,:,t],0,1)*255)
            bgr=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
            cv2.rectangle(bgr,(0,0),(W,20),(0,0,0),-1)
            cv2.putText(bgr,name,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
            panels.append(bgr)
        writer.write(np.concatenate(panels,axis=1))
    writer.release(); return True
