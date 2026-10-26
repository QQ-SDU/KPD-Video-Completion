# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np

def _resize_nearest(frame, size_hw):
    if size_hw is None:
        return frame
    h, w = size_hw
    src_h, src_w = frame.shape
    yi = np.linspace(0, src_h - 1, h).round().astype(int)
    xi = np.linspace(0, src_w - 1, w).round().astype(int)
    return frame[np.ix_(yi, xi)]

def parse_resize(text):
    if text is None or str(text).lower() in {"none", "native", "0"}:
        return None
    a, b = str(text).lower().replace("x", ",").split(",")
    return int(a), int(b)

def load_y4m(path, max_frames=100, resize_hw=None, start_frame=0):
    path = Path(path)
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
        W = int(meta["W"]); H = int(meta["H"])
        chroma = meta.get("C", "420").lower()
        ysize = W * H
        if chroma.startswith("420"):
            chroma_bytes = 2 * ((W + 1)//2) * ((H + 1)//2)
        elif chroma.startswith("422"):
            chroma_bytes = 2 * ((W + 1)//2) * H
        elif chroma.startswith("444"):
            chroma_bytes = 2 * W * H
        elif chroma.startswith("mono"):
            chroma_bytes = 0
        else:
            raise ValueError(f"Unsupported Y4M chroma: {chroma}")

        frame_idx = 0
        while len(frames) < max_frames:
            fh = f.readline()
            if not fh:
                break
            if not fh.startswith(b"FRAME"):
                raise ValueError(f"Unexpected Y4M frame header near {frame_idx}")
            y = f.read(ysize)
            if len(y) < ysize:
                break
            frame = np.frombuffer(y, dtype=np.uint8).reshape(H, W).astype(np.float64) / 255.0
            if frame_idx >= start_frame:
                frames.append(_resize_nearest(frame, resize_hw))
            if chroma_bytes:
                uv = f.read(chroma_bytes)
                if len(uv) < chroma_bytes:
                    break
            frame_idx += 1

    if not frames:
        raise RuntimeError(f"No frames loaded from {path}")
    return np.stack(frames, axis=2)

def load_npy(path, max_frames=100, resize_hw=None):
    x = np.load(path)
    if x.ndim != 3:
        raise ValueError(f"Expected a 3-D .npy tensor, got {x.shape}")
    x = x.astype(np.float64)
    if x.max() > 1.5:
        x /= 255.0
    x = x[:, :, :max_frames]
    if resize_hw is not None:
        x = np.stack([_resize_nearest(x[:, :, t], resize_hw) for t in range(x.shape[2])], axis=2)
    return np.clip(x, 0.0, 1.0)

def load_input(path, max_frames=100, resize_hw=None):
    path = Path(path)
    if path.suffix.lower() == ".y4m":
        return load_y4m(path, max_frames=max_frames, resize_hw=resize_hw)
    if path.suffix.lower() == ".npy":
        return load_npy(path, max_frames=max_frames, resize_hw=resize_hw)
    raise ValueError("This reproducible package supports .y4m and .npy inputs.")
