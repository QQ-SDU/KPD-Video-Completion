from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import scipy.io as sio

def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Input contains NaN/Inf.")
    mn, mx = float(x.min()), float(x.max())
    if mn >= 0 and mx <= 1.00001:
        return np.clip(x, 0, 1).astype(np.float32)
    if mn >= 0 and mx <= 255.0001:
        return np.clip(x / 255.0, 0, 1).astype(np.float32)
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def read_y4m_luma(path: str | Path, max_frames: int | None = None) -> np.ndarray:
    path = Path(path)
    frames = []
    with path.open("rb") as f:
        header = f.readline().decode("ascii", errors="replace").strip()
        if not header.startswith("YUV4MPEG2"):
            raise ValueError("Not a Y4M file.")
        toks = header.split()
        w = h = None
        chroma = "420"
        for tok in toks[1:]:
            if tok.startswith("W"):
                w = int(tok[1:])
            elif tok.startswith("H"):
                h = int(tok[1:])
            elif tok.startswith("C"):
                chroma = tok[1:].lower()
        if not w or not h:
            raise ValueError("Cannot parse Y4M width/height.")

        def chroma_bytes(c):
            if c.startswith("mono"):
                return 0
            if c.startswith("420"):
                return 2 * ((w + 1)//2) * ((h + 1)//2)
            if c.startswith("422"):
                return 2 * ((w + 1)//2) * h
            if c.startswith("444"):
                return 2 * w * h
            # Most common Y4M video material used in the classical benchmark is 4:2:0.
            raise ValueError(f"Unsupported Y4M chroma mode: {c}")

        cb = chroma_bytes(chroma)
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            tag = f.readline()
            if not tag:
                break
            if not tag.startswith(b"FRAME"):
                raise ValueError(f"Unexpected Y4M frame marker: {tag[:32]!r}")
            y = f.read(w*h)
            if len(y) != w*h:
                break
            arr = np.frombuffer(y, dtype=np.uint8).reshape(h, w)
            frames.append(arr)
            if cb:
                skipped = f.read(cb)
                if len(skipped) != cb:
                    raise ValueError("Truncated Y4M chroma plane.")
    if not frames:
        raise ValueError("No frames found.")
    return np.stack(frames, axis=2).astype(np.float32) / 255.0

def _guess_mat_key(d):
    preferred = ["gt", "Ohsi", "Omsi", "video", "X", "data"]
    for k in preferred:
        if k in d and isinstance(d[k], np.ndarray) and d[k].ndim == 3:
            return k
    for k, v in d.items():
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3:
            return k
    raise KeyError("No 3-D numeric tensor found in MAT file.")

def load_video(path: str | Path, key: str | None = None, layout: str = "HWT",
               max_frames: int | None = None) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".y4m":
        x = read_y4m_luma(path, max_frames=max_frames)
    elif ext == ".npy":
        x = np.load(path)
    elif ext == ".npz":
        d = np.load(path)
        k = key or ("gt" if "gt" in d else d.files[0])
        x = d[k]
    elif ext == ".mat":
        d = sio.loadmat(path)
        k = key or _guess_mat_key(d)
        x = d[k]
    else:
        raise ValueError("Supported inputs: .y4m, .npy, .npz, .mat")

    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3-D video tensor, got {x.shape}.")
    if layout.upper() == "THW":
        x = np.transpose(x, (1, 2, 0))
    elif layout.upper() != "HWT":
        raise ValueError("layout must be HWT or THW")
    if max_frames is not None and x.shape[2] > max_frames:
        x = x[:, :, :max_frames]
    return normalize01(x)

def resize_hwt(x: np.ndarray, height: int | None, width: int | None) -> np.ndarray:
    if height is None and width is None:
        return x.astype(np.float32)
    from PIL import Image
    h0, w0, t = x.shape
    height = int(height or h0)
    width = int(width or w0)
    out = np.empty((height, width, t), dtype=np.float32)
    for k in range(t):
        im = Image.fromarray(np.clip(x[:,:,k]*255.0,0,255).astype(np.uint8), mode="L")
        im = im.resize((width, height), Image.Resampling.BICUBIC)
        out[:,:,k] = np.asarray(im, dtype=np.float32)/255.0
    return out

def make_tube_mask(h: int, w: int, t: int, sr: float, seed: int) -> np.ndarray:
    if not (0 < sr <= 1):
        raise ValueError("sr must be in (0,1].")
    rng = np.random.default_rng(seed)
    n = h*w
    nobs = int(round(sr*n))
    ids = rng.choice(n, size=nobs, replace=False)
    m2 = np.zeros(n, dtype=bool)
    m2[ids] = True
    m2 = m2.reshape(h, w)
    return np.repeat(m2[:,:,None], t, axis=2)

def linear_spatial_init(gt_observed: np.ndarray, mask3d: np.ndarray) -> np.ndarray:
    """Framewise linear interpolation, sharing spatial Delaunay geometry across frames.

    Equivalent to scipy.interpolate.griddata(..., method="linear") with
    nearest fill outside the convex hull. The geometry (and mask) do not
    change between video frames, so solve the triangulation once, then apply
    barycentric weights to all frames in chunks to control peak memory.
    """
    from scipy.spatial import Delaunay, cKDTree, QhullError
    h, w, t = gt_observed.shape
    m2 = np.asarray(mask3d[:, :, 0], dtype=bool)
    pts = np.column_stack(np.nonzero(m2))
    if len(pts) < 3:
        fill = float(np.mean(gt_observed[m2])) if np.any(m2) else 0.0
        out = np.full_like(gt_observed, fill, dtype=np.float32)
        out[mask3d] = gt_observed[mask3d]
        return out
    values = np.asarray(gt_observed[m2], dtype=np.float32)  # (num_observed_tubes, time)
    output = np.empty((h * w, t), dtype=np.float32)
    try:
        triangles = Delaunay(pts)
    except QhullError:
        triangles = None
    nearest = cKDTree(pts)
    chunk = 32768
    for start in range(0, h*w, chunk):
        stop=min(start+chunk,h*w)
        indices=np.arange(start,stop)
        query=np.column_stack((indices//w,indices%w))
        if triangles is None:
            _, nearest_ids=nearest.query(query)
            output[start:stop,:]=values[nearest_ids]
            continue
        simplex=triangles.find_simplex(query)
        inside=simplex>=0
        rows=output[start:stop,:]
        if np.any(inside):
            sidx=simplex[inside]
            shift=query[inside]-triangles.transform[sidx,2]
            weights2=np.einsum('nij,nj->ni',triangles.transform[sidx,:2,:],shift)
            weights=np.column_stack((weights2,1.-weights2.sum(axis=1)))
            ids=triangles.simplices[sidx]
            rows[inside,:]=np.einsum('ni,nit->nt',weights,values[ids],optimize=True)
        if np.any(~inside):
            _,nearest_ids=nearest.query(query[~inside])
            rows[~inside,:]=values[nearest_ids]
    out=output.reshape(h,w,t)
    out[mask3d]=gt_observed[mask3d]
    return np.clip(out,0.,1.).astype(np.float32)

def save_case(out_base: str | Path, gt, observed, mask, init_linear, meta):
    """Atomically save NPZ/MAT/JSON case files.

    If preparation is interrupted, a half-written final case will not be left
    behind. Temporary files are replaced only after each write completes.
    """
    import os, uuid
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    token=uuid.uuid4().hex[:10]
    final_npz=Path(str(out_base)+".npz")
    final_mat=Path(str(out_base)+".mat")
    final_json=Path(str(out_base)+".json")
    tmp_npz=out_base.parent/(out_base.name+f".{token}.tmp.npz")
    tmp_mat=out_base.parent/(out_base.name+f".{token}.tmp.mat")
    tmp_json=out_base.parent/(out_base.name+f".{token}.tmp.json")
    try:
        with tmp_npz.open("wb") as fh:
            np.savez_compressed(fh,gt=gt.astype(np.float32),observed=observed.astype(np.float32),
                                mask=mask.astype(np.uint8),init_linear=init_linear.astype(np.float32))
        omega0=np.flatnonzero(mask.ravel(order="F")); omega1=(omega0+1).astype(np.int64)
        sio.savemat(tmp_mat,{"gt":gt.astype(np.float32),"observed":observed.astype(np.float32),
                            "mask":mask.astype(np.uint8),"Omega":mask.astype(np.uint8),
                            "omega":omega1.reshape(-1,1),"Nhsi":observed.astype(np.float32),
                            "Ohsi":gt.astype(np.float32),"init_linear":init_linear.astype(np.float32)},
                    do_compression=True)
        tmp_json.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
        os.replace(tmp_npz,final_npz); os.replace(tmp_mat,final_mat); os.replace(tmp_json,final_json)
    finally:
        for p in (tmp_npz,tmp_mat,tmp_json):
            try:
                if p.exists(): p.unlink()
            except Exception: pass
