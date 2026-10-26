from __future__ import annotations
import argparse, json, secrets
from pathlib import Path
import numpy as np
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Benchmark'))
from benchmark.io_utils import linear_spatial_init, save_case

from color_structured9.settings import INPUT_DIR, MANIFEST, CASES_DIR, CASES_RGB_DIR, MASK_TYPES, CHANNELS, ASSETS_DIR, PREVIEW_DIR
from color_structured9.color_video_io import load_color_video, load_manifest, save_rgb_case
from color_structured9.structured_masks import build_observed_mask


def parse_int_or_none(v):
    if v is None or str(v).strip() == '':
        return None
    return int(v)


def _save_rgb_png(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser(description='Prepare structured-mask color video cases for 9-method comparison.')
    ap.add_argument('--manifest', default=str(MANIFEST))
    ap.add_argument('--input-dir', default=str(INPUT_DIR))
    ap.add_argument('--videos', nargs='*', default=None, help='Optional subset of video names, e.g. Bus Container Coastguard')
    ap.add_argument('--masks', nargs='*', default=MASK_TYPES)
    ap.add_argument('--seed', type=int, default=None, help='Optional reproducibility seed; omitted => generate a fresh seed.')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(32)
    print(f"Run seed: {args.seed}")

    rows = load_manifest(args.manifest)
    wanted = {v.lower() for v in args.videos} if args.videos else None

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    CASES_RGB_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    prepared = []
    for row in rows:
        enabled = str(row.get('enabled', '1')).strip().lower() not in {'0', 'false', 'no'}
        video = row['video'].strip()
        if not enabled:
            continue
        if wanted and video.lower() not in wanted:
            continue
        file_name = row.get('file', '').strip()
        path = Path(args.input_dir) / file_name
        if not path.exists():
            raise FileNotFoundError(f'Missing source file for {video}: {path}')
        width = parse_int_or_none(row.get('width'))
        height = parse_int_or_none(row.get('height'))
        max_frames = parse_int_or_none(row.get('max_frames'))
        fmt = row.get('format', '')
        frame_preview = parse_int_or_none(row.get('preview_frame'))
        print(f'Loading {video} from {path} ...')
        gt_rgb = load_color_video(path, width=width, height=height, max_frames=max_frames, fmt=fmt)
        h, w, c, t = gt_rgb.shape
        if c != 3:
            raise ValueError(f'{video} must be RGB, got shape={gt_rgb.shape}')
        preview_idx = int(np.clip(frame_preview if frame_preview is not None else min(14, t - 1), 0, t - 1))

        roi = {
            'x': parse_int_or_none(row.get('roi_x')),
            'y': parse_int_or_none(row.get('roi_y')),
            'w': parse_int_or_none(row.get('roi_w')),
            'h': parse_int_or_none(row.get('roi_h')),
        }

        for mask_type in args.masks:
            stem = f'{video}__{mask_type}'
            rgb_npz = CASES_RGB_DIR / f'{stem}.npz'
            if rgb_npz.exists() and not args.force:
                print(f'[SKIP existing] {stem}')
                prepared.append(stem)
                continue
            m2 = build_observed_mask(mask_type, h, w, seed=args.seed)
            mask3 = np.repeat(m2[:, :, None], t, axis=2)
            observed_rgb = gt_rgb * m2[:, :, None, None].astype(np.float32)
            meta = {
                'video': video,
                'file': str(path.resolve()),
                'shape_hwct': [int(h), int(w), int(c), int(t)],
                'mask_type': mask_type,
                'observed_ratio': float(m2.mean()),
                'missing_ratio': float(1.0 - m2.mean()),
                'seed': int(args.seed),
                'mask_protocol': 'same 2-D structured spatial mask applied to all frames and all RGB channels',
                'roi': roi,
                'preview_frame_index': preview_idx,
            }
            save_rgb_case(CASES_RGB_DIR / stem, gt_rgb, observed_rgb, m2, meta)
            print(f'  Saved RGB case: {stem} | missing_ratio={1.0 - m2.mean():.6f}')

            # Save plotting-friendly image assets.
            asset_dir = ASSETS_DIR / video / mask_type
            asset_dir.mkdir(parents=True, exist_ok=True)
            mask_missing_png = np.where(m2, 255, 0).astype(np.uint8)
            mask_observed_png = np.where(m2, 255, 0).astype(np.uint8)
            Image.fromarray(mask_missing_png).save(asset_dir / 'mask_observed_white.png')
            Image.fromarray(np.where(~m2, 255, 0).astype(np.uint8)).save(asset_dir / 'mask_missing_white.png')
            _save_rgb_png(asset_dir / f'gt_frame_{preview_idx:03d}.png', gt_rgb[:, :, :, preview_idx])
            _save_rgb_png(asset_dir / f'observed_frame_{preview_idx:03d}.png', observed_rgb[:, :, :, preview_idx])

            # Also store all frame images for later plotting if desired.
            frame_dir_gt = PREVIEW_DIR / video / mask_type / 'gt_frames'
            frame_dir_obs = PREVIEW_DIR / video / mask_type / 'observed_frames'
            frame_dir_gt.mkdir(parents=True, exist_ok=True)
            frame_dir_obs.mkdir(parents=True, exist_ok=True)
            for k in range(t):
                _save_rgb_png(frame_dir_gt / f'frame_{k:03d}.png', gt_rgb[:, :, :, k])
                _save_rgb_png(frame_dir_obs / f'frame_{k:03d}.png', observed_rgb[:, :, :, k])

            for cname, ci in CHANNELS:
                gt = gt_rgb[:, :, ci, :].astype(np.float32)
                observed = observed_rgb[:, :, ci, :].astype(np.float32)
                init_linear = linear_spatial_init(observed, mask3)
                case_meta = dict(meta)
                case_meta.update({
                    'channel_name': cname,
                    'channel_index': int(ci),
                    'shape_hwt': [int(h), int(w), int(t)],
                })
                save_case(CASES_DIR / f'{stem}__{cname}', gt, observed, mask3, init_linear, case_meta)
            prepared.append(stem)

    out_list = CASES_RGB_DIR / 'prepared_cases.json'
    out_list.write_text(json.dumps({'prepared': prepared}, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Prepared {len(prepared)} RGB case groups. Index saved to {out_list}')


if __name__ == '__main__':
    main()
