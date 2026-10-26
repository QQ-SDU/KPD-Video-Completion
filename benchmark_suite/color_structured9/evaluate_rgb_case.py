from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity


def psnr(a, b, data_range=1.0, mask=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = (a - b) ** 2
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if not np.any(m):
            return float('nan')
        mse = float(d[m].mean())
    else:
        mse = float(d.mean())
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / mse)


def temporal_gradient_mae(gt, rec, mask3):
    dg = np.diff(gt, axis=3)
    dr = np.diff(rec, axis=3)
    e = np.abs(dg - dr)
    m = mask3[:, :, :, :-1]
    return float(e[m].mean()) if np.any(m) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb-case', required=True)
    ap.add_argument('--recon', required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    case = np.load(args.rgb_case)
    recd = np.load(args.recon)
    gt = np.asarray(case['gt_rgb'], dtype=np.float32)
    observed = np.asarray(case['observed_rgb'], dtype=np.float32)
    mask2d = np.asarray(case['mask2d']).astype(bool)
    rec = np.asarray(recd['recon_rgb'], dtype=np.float32)
    if rec.shape != gt.shape:
        raise ValueError(f'Shape mismatch: recon={rec.shape}, gt={gt.shape}')
    mask4 = np.repeat(mask2d[:, :, None, None], gt.shape[2], axis=2)
    mask4 = np.repeat(mask4, gt.shape[3], axis=3)
    miss4 = ~mask4

    # Enforce data consistency in the evaluator.
    rec = rec.copy()
    rec[mask4] = observed[mask4]
    rec = np.clip(rec, 0.0, 1.0)

    ssim_vals = []
    for k in range(gt.shape[3]):
        ssim_vals.append(structural_similarity(gt[:, :, :, k], rec[:, :, :, k], data_range=1.0, channel_axis=2))

    row = {
        'method': args.method,
        'rgb_case': Path(args.rgb_case).stem,
        'shape_hwct': list(map(int, gt.shape)),
        'observed_ratio': float(mask2d.mean()),
        'psnr_full': psnr(gt, rec, 1.0),
        'psnr_missing': psnr(gt, rec, 1.0, mask=miss4),
        'mae_missing': float(np.abs(gt[miss4] - rec[miss4]).mean()),
        'temporal_gradient_mae_missing': temporal_gradient_mae(gt, rec, miss4),
        'ssim_frame_mean': float(np.mean(ssim_vals)),
        'ssim_frame_std': float(np.std(ssim_vals)),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
