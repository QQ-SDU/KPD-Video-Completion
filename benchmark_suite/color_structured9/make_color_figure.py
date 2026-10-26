from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from color_structured9.settings import CASES_RGB_DIR, RESULTS_RGB_DIR, FIGURES_DIR, METHODS_ORDER, DEFAULT_FRAME_INDEX
from color_structured9.color_video_io import load_manifest


def load_rgb_case(npz_path: Path):
    d = np.load(npz_path)
    gt = np.asarray(d['gt_rgb'], dtype=np.float32)
    observed = np.asarray(d['observed_rgb'], dtype=np.float32)
    mask2d = np.asarray(d['mask2d']).astype(bool)
    return gt, observed, mask2d


def load_recon(method: str, video: str, mask_type: str):
    p = RESULTS_RGB_DIR / method / f'{method}__{video}__{mask_type}.npz'
    if not p.exists():
        return None
    d = np.load(p)
    return np.asarray(d['recon_rgb'], dtype=np.float32)


def resolve_roi(manifest_row: dict, h: int, w: int):
    def _get(key):
        v = manifest_row.get(key, '')
        return None if v is None or str(v).strip() == '' else int(v)
    x = _get('roi_x'); y = _get('roi_y'); rw = _get('roi_w'); rh = _get('roi_h')
    if None in {x, y, rw, rh}:
        s = max(24, int(round(min(h, w) * 0.22)))
        x = (w - s) // 2
        y = (h - s) // 2
        rw = rh = s
    x = max(0, min(w - 1, x)); y = max(0, min(h - 1, y))
    rw = max(4, min(w - x, rw)); rh = max(4, min(h - y, rh))
    return x, y, rw, rh


def main():
    ap = argparse.ArgumentParser(description='Build qualitative color-video figures with zoom and error map.')
    ap.add_argument('--manifest', default=str(Path(__file__).resolve().parents[1] / 'data_color_structured9/manifest.csv'))
    ap.add_argument('--videos', nargs='*', default=['Bus', 'Container', 'Coastguard'])
    ap.add_argument('--masks', nargs='*', default=['grid', 'text', 'block'])
    ap.add_argument('--methods', nargs='*', default=METHODS_ORDER)
    ap.add_argument('--frame', type=int, default=DEFAULT_FRAME_INDEX)
    ap.add_argument('--include-observed', action='store_true', default=True)
    ap.add_argument('--include-gt', action='store_true', default=True)
    args = ap.parse_args()

    manifest = {r['video'].strip().lower(): r for r in load_manifest(args.manifest)}
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for video in args.videos:
        mrow = manifest.get(video.lower(), {})
        for mask_type in args.masks:
            case_npz = CASES_RGB_DIR / f'{video}__{mask_type}.npz'
            if not case_npz.exists():
                print(f'[WARN] missing prepared case: {case_npz}')
                continue
            gt, obs, mask2d = load_rgb_case(case_npz)
            h, w, _, t = gt.shape
            fi = max(0, min(t - 1, args.frame))
            x0, y0, rw, rh = resolve_roi(mrow, h, w)
            items = []
            if args.include_observed:
                items.append(('Observed', obs))
            for method in args.methods:
                rec = load_recon(method, video, mask_type)
                if rec is not None:
                    items.append((method, np.clip(rec, 0.0, 1.0)))
            if args.include_gt:
                items.append(('Ground truth', gt))
            if len(items) <= 1:
                print(f'[WARN] no reconstructions found for {video} / {mask_type}')
                continue

            n = len(items)
            fig, axs = plt.subplots(3, n, figsize=(2.2 * n, 6.6), gridspec_kw={'height_ratios':[1.0, 0.68, 0.68]}, squeeze=False)
            for j, (name, arr) in enumerate(items):
                frame = arr[:, :, :, fi]
                ax = axs[0, j]
                ax.imshow(frame)
                ax.add_patch(Rectangle((x0, y0), rw, rh, fill=False, edgecolor='red', linewidth=1.4))
                ax.set_title(name, fontsize=9, fontweight='bold')
                ax.axis('off')

                zoom = arr[y0:y0+rh, x0:x0+rw, :, fi]
                az = axs[1, j]
                az.imshow(zoom)
                az.axis('off')

                err = np.mean(np.abs(frame - gt[:, :, :, fi]), axis=2)
                ae = axs[2, j]
                ae.imshow(err, cmap='gray', vmin=0.0, vmax=max(0.15, float(np.quantile(err, 0.98))))
                ae.axis('off')
            axs[1, 0].set_ylabel('Zoom', fontsize=10, fontweight='bold')
            axs[2, 0].set_ylabel('Error map', fontsize=10, fontweight='bold')
            plt.subplots_adjust(left=0.01, right=0.995, top=0.91, bottom=0.01, wspace=0.03, hspace=0.04)
            out_dir = FIGURES_DIR / video
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f'{video}__{mask_type}__frame{fi:02d}'
            fig.savefig(out_dir / f'{stem}.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
            fig.savefig(out_dir / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)
            print(f'Saved figure: {out_dir / (stem + ".png")}')


if __name__ == '__main__':
    main()