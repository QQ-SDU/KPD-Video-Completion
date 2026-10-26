from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .io_utils import load_video, resize_hwt, make_tube_mask, linear_spatial_init, save_case

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--name", default="video")
    p.add_argument("--key", default=None)
    p.add_argument("--layout", default="HWT", choices=["HWT","THW"])
    p.add_argument("--sr", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out-dir", default="data/cases")
    p.add_argument("--make-linear-init", action="store_true")
    args = p.parse_args()

    gt = load_video(args.input, key=args.key, layout=args.layout, max_frames=args.max_frames)
    gt = resize_hwt(gt, args.height, args.width)
    h,w,t = gt.shape
    mask = make_tube_mask(h,w,t,args.sr,args.seed)
    observed = gt * mask.astype(np.float32)

    if args.make_linear_init:
        print("Building deterministic linear interpolation initialization...")
        init_linear = linear_spatial_init(observed, mask)
    else:
        mean_obs = float(observed[mask].mean()) if np.any(mask) else 0.0
        init_linear = np.where(mask, observed, mean_obs).astype(np.float32)

    sr_tag = f"{int(round(args.sr*100)):03d}"
    base = Path(args.out_dir) / f"{args.name}_sr{sr_tag}_seed{args.seed}"
    meta = {
        "name": args.name,
        "input": str(Path(args.input).resolve()),
        "shape_hwt": [int(h),int(w),int(t)],
        "sampling_rate_requested": args.sr,
        "sampling_rate_realized": float(mask[:,:,0].mean()),
        "seed": args.seed,
        "sampling": "mode-3 tube-wise exact-count spatial mask",
        "intensity_range": [float(gt.min()), float(gt.max())],
    }
    save_case(base, gt, observed, mask, init_linear, meta)
    print(f"Created {base}.npz")
    print(f"Created {base}.mat")
    print(f"Created {base}.json")
    print(f"Shape={gt.shape}, realized SR={mask[:,:,0].mean():.6f}")

if __name__ == "__main__":
    main()
