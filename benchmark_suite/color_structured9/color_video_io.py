from __future__ import annotations
from pathlib import Path
import csv
import json
import numpy as np


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise ValueError('Empty array.')
    if not np.isfinite(x).all():
        raise ValueError('NaN/Inf found.')
    mn, mx = float(x.min()), float(x.max())
    if mn >= 0.0 and mx <= 1.00001:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    if mn >= 0.0 and mx <= 255.0001:
        return np.clip(x / 255.0, 0.0, 1.0).astype(np.float32)
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def ycbcr_to_rgb_uint8(y, cb, cr):
    y = y.astype(np.float32)
    cb = cb.astype(np.float32) - 128.0
    cr = cr.astype(np.float32) - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=2)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def upsample_420(ch, h, w):
    return np.repeat(np.repeat(ch, 2, axis=0), 2, axis=1)[:h, :w]


def read_yuv420p_rgb(path: str | Path, width: int, height: int, max_frames: int | None = None) -> np.ndarray:
    path = Path(path)
    frame_bytes = width * height * 3 // 2
    raw = path.read_bytes()
    nframes = len(raw) // frame_bytes
    if nframes <= 0:
        raise ValueError(f'No complete frames in {path}')
    if max_frames is not None:
        nframes = min(nframes, int(max_frames))
    out = np.empty((height, width, 3, nframes), dtype=np.uint8)
    off = 0
    for k in range(nframes):
        y = np.frombuffer(raw, dtype=np.uint8, count=width*height, offset=off).reshape(height, width)
        off += width * height
        u = np.frombuffer(raw, dtype=np.uint8, count=(width//2)*(height//2), offset=off).reshape(height//2, width//2)
        off += (width//2) * (height//2)
        v = np.frombuffer(raw, dtype=np.uint8, count=(width//2)*(height//2), offset=off).reshape(height//2, width//2)
        off += (width//2) * (height//2)
        rgb = ycbcr_to_rgb_uint8(y, upsample_420(u, height, width), upsample_420(v, height, width))
        out[..., k] = rgb
    return normalize01(out)


def read_y4m_rgb(path: str | Path, max_frames: int | None = None) -> np.ndarray:
    path = Path(path)
    frames = []
    with path.open('rb') as f:
        header = f.readline().decode('ascii', errors='replace').strip()
        if not header.startswith('YUV4MPEG2'):
            raise ValueError('Not a Y4M file')
        toks = header.split()
        w = h = None
        chroma = '420'
        for tok in toks[1:]:
            if tok.startswith('W'):
                w = int(tok[1:])
            elif tok.startswith('H'):
                h = int(tok[1:])
            elif tok.startswith('C'):
                chroma = tok[1:].lower()
        if not w or not h:
            raise ValueError('Cannot parse Y4M header')
        if not chroma.startswith('420'):
            raise ValueError(f'Only 420 Y4M is supported in this package, got {chroma}')
        uv_bytes = 2 * ((w + 1) // 2) * ((h + 1) // 2)
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            tag = f.readline()
            if not tag:
                break
            if not tag.startswith(b'FRAME'):
                raise ValueError(f'Unexpected Y4M frame marker: {tag[:32]!r}')
            y = f.read(w * h)
            if len(y) != w * h:
                break
            uv = f.read(uv_bytes)
            if len(uv) != uv_bytes:
                raise ValueError('Truncated chroma planes.')
            u = np.frombuffer(uv[:uv_bytes//2], dtype=np.uint8).reshape((h+1)//2, (w+1)//2)
            v = np.frombuffer(uv[uv_bytes//2:], dtype=np.uint8).reshape((h+1)//2, (w+1)//2)
            yy = np.frombuffer(y, dtype=np.uint8).reshape(h, w)
            rgb = ycbcr_to_rgb_uint8(yy, upsample_420(u, h, w), upsample_420(v, h, w))
            frames.append(rgb)
    if not frames:
        raise ValueError('No frames found in Y4M file.')
    arr = np.stack(frames, axis=3)
    return normalize01(arr)


def read_generic_video_rgb(path: str | Path, max_frames: int | None = None) -> np.ndarray:
    path = Path(path)
    try:
        import imageio.v3 as iio
        frames = []
        for i, fr in enumerate(iio.imiter(path)):
            if max_frames is not None and i >= max_frames:
                break
            fr = np.asarray(fr)
            if fr.ndim == 2:
                fr = np.repeat(fr[:, :, None], 3, axis=2)
            if fr.shape[2] > 3:
                fr = fr[:, :, :3]
            frames.append(fr)
        if not frames:
            raise ValueError('No frames read.')
        return normalize01(np.stack(frames, axis=3))
    except Exception as e:
        raise RuntimeError(f'Failed to read generic video {path}. Install imageio[ffmpeg] if needed. Details: {e}')


def load_color_video(path: str | Path, width: int | None = None, height: int | None = None,
                     max_frames: int | None = None, fmt: str | None = None) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    fmt = (fmt or '').lower().strip()
    if ext == '.yuv' or fmt in {'yuv420p', 'i420'}:
        if width is None or height is None:
            raise ValueError('Raw .yuv requires width and height from manifest.')
        x = read_yuv420p_rgb(path, int(width), int(height), max_frames=max_frames)
    elif ext == '.y4m':
        x = read_y4m_rgb(path, max_frames=max_frames)
    elif ext == '.npy':
        x = np.load(path)
        if max_frames is not None and x.shape[-1] > max_frames:
            x = x[..., :max_frames]
        x = normalize01(x)
    elif ext == '.npz':
        d = np.load(path)
        key = 'gt' if 'gt' in d else d.files[0]
        x = d[key]
        if max_frames is not None and x.shape[-1] > max_frames:
            x = x[..., :max_frames]
        x = normalize01(x)
    elif ext == '.mat':
        import scipy.io as sio
        d = sio.loadmat(path)
        key = next((k for k,v in d.items() if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 4), None)
        if key is None:
            raise KeyError(f'No 4-D color tensor found in {path}')
        x = d[key]
        if max_frames is not None and x.shape[-1] > max_frames:
            x = x[..., :max_frames]
        x = normalize01(x)
    else:
        x = read_generic_video_rgb(path, max_frames=max_frames)
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError(f'Expected HxWx3xT array, got {x.shape}')
    if x.shape[2] != 3:
        raise ValueError(f'Expected RGB channel dimension=3, got {x.shape}')
    return x.astype(np.float32)


def load_manifest(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Manifest not found: {path}')
    rows = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if not row.get('video'):
                continue
            row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            rows.append(row)
    return rows


def save_rgb_case(base_path: str | Path, gt_rgb, observed_rgb, mask2d, meta: dict):
    import scipy.io as sio
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(base) + '.npz', gt_rgb=gt_rgb.astype(np.float32), observed_rgb=observed_rgb.astype(np.float32), mask2d=mask2d.astype(np.uint8))
    sio.savemat(str(base) + '.mat', {
        'gt_rgb': gt_rgb.astype(np.float32),
        'observed_rgb': observed_rgb.astype(np.float32),
        'mask2d': mask2d.astype(np.uint8),
    }, do_compression=True)
    Path(str(base) + '.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
