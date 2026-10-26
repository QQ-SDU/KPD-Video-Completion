from __future__ import annotations
import argparse, csv, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'workflow'))
from common import load_method_defaults
sys.path.insert(0, str(ROOT / 'Benchmark'))
from benchmark.evaluate_output import load_recon, align_shape

from color_structured9.settings import (
    CASES_DIR, CASES_RGB_DIR, RESULTS_CH_DIR, RESULTS_RGB_DIR, METRICS_DIR,
    LOGS_DIR, METHODS_ORDER, CHANNELS, MASK_TYPES
)
from color_structured9.color_video_io import load_manifest


def parse_seed_from_stem(stem: str) -> int:
    # Color-structured cases use one run-level seed. No fixed seed is stored.
    env = os.environ.get("KPD_COLOR_SEED")
    if env is not None and str(env).strip() != "":
        return int(env)
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def build_python_cmd(method: str, case: Path, out: Path, cfg: dict):
    seed = parse_seed_from_stem(case.stem)
    if method == 'S2NTNN':
        c = cfg[method]
        return [sys.executable, str(ROOT/'Methods/S2NTNN/benchmark_runner_s2ntnn.py'), '--case', str(case), '--out', str(out), '--seed', str(seed), '--max-iter', str(c['max_iter']), '--lr', str(c['lr']), '--lambda-nuc', str(c['lambda_nuc']), '--n4-mult', str(c['n4_mult'])]
    if method == 'HLRTF':
        c = cfg[method]
        return [sys.executable, str(ROOT/'Methods/HLRTF/benchmark_runner_hlrtf.py'), '--case', str(case), '--out', str(out), '--seed', str(seed), '--max-iter', str(c['max_iter']), '--lr', str(c['lr']), '--gamma', str(c['gamma']), '--shrink', str(c['shrink'])]
    if method == 'LRTFR':
        c = cfg[method]
        return [sys.executable, str(ROOT/'Methods/LRTFR/run_same_mask.py'), '--case', str(case), '--out', str(out), '--seed', str(seed), '--max-iter', str(c['max_iter']), '--lr', str(c['lr']), '--weight-decay', str(c['weight_decay']), '--omega', str(c['omega']), '--down', ','.join(map(str, c['down']))]
    if method == 'DRO-TFF':
        c = cfg[method]
        return [sys.executable, str(ROOT/'Benchmark/adapters/dro_tff/run_drotff_same_mask.py'), '--repo', str(ROOT/'Methods/DRO-TFF'), '--case', str(case), '--out', str(out), '--seed', str(seed), '--max-iter', str(c['max_iter']), '--lr', str(c['lr']), '--beta', str(c['beta']), '--omega', str(c['omega']), '--layers', str(c['layers']), '--gamma', str(c['gamma'])]
    if method == 'OTLRM':
        c = cfg[method]
        return [sys.executable, str(ROOT/'Methods/OTLRM/run_same_mask.py'), '--case', str(case), '--out', str(out), '--seed', str(seed), '--max-iter', str(c['max_iter']), '--lr', str(c['lr']), '--shrink', str(c['shrink']), '--gamma', str(c['gamma']), '--weight-decay', str(c['weight_decay'])]
    if method == 'Ours':
        c = cfg[method]
        return [sys.executable, str(ROOT/'Methods/Ours/run_same_mask.py'), '--case', str(case), '--out', str(out), '--seed', str(seed), '--rank', str(c['rank']), '--masked-sweeps', str(c['masked_sweeps']), '--lam-space', str(c['lambda_s']), '--lam-time', str(c['lambda_t']), '--ridge', str(c['lambda_0']), '--crossfit-folds', str(c['crossfit_folds']), '--hidden', str(c['hidden']), '--max-residual', str(c['max_residual']), '--patch-t', str(c['patch_t']), '--patch-h', str(c['patch_h']), '--patch-w', str(c['patch_w']), '--batch-size', str(c['batch_size']), '--steps', str(c['steps']), '--lr', str(c['lr']), '--smooth-l1-beta', str(c['smooth_l1_beta']), '--detail-weight', str(c['detail_weight']), '--magnitude-scale', str(c['magnitude_scale']), '--magnitude-cap', str(c['magnitude_cap']), '--residual-penalty', str(c['residual_penalty']), '--device', 'auto']
    raise KeyError(method)


def matlab_escape(s: str) -> str:
    return s.replace('\\', '/').replace("'", "''")


def build_matlab_cmd(method: str, case: Path, out: Path, seed: int):
    script_dir = matlab_escape(str((ROOT/'color_structured9').resolve()))
    case_s = matlab_escape(str(case.resolve()))
    out_s = matlab_escape(str(out.resolve()))
    expr = f"addpath('{script_dir}'); run_one_matlab_method('{method}','{case_s}','{out_s}',{seed});"
    return ['matlab', '-batch', expr]


def valid_file(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 256


def merge_rgb_group(video: str, mask_type: str, method: str, rgb_case_npz: Path, out_npz: Path, per_channel_dir: Path):
    case = np.load(rgb_case_npz)
    gt_rgb = np.asarray(case['gt_rgb'], dtype=np.float32)
    obs_rgb = np.asarray(case['observed_rgb'], dtype=np.float32)
    mask2d = np.asarray(case['mask2d']).astype(bool)
    recon = np.zeros_like(gt_rgb, dtype=np.float32)
    for cname, ci in CHANNELS:
        p = per_channel_dir / f'{method}__{video}__{mask_type}__{cname}.mat'
        if not p.exists():
            raise FileNotFoundError(f'Missing channel reconstruction: {p}')
        x, runtime = load_recon(p)
        x = align_shape(x, gt_rgb[:, :, 0, :].shape)
        x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
        recon[:, :, ci, :] = x
    obs4 = np.repeat(mask2d[:, :, None, None], 3, axis=2)
    obs4 = np.repeat(obs4, gt_rgb.shape[3], axis=3)
    recon[obs4] = obs_rgb[obs4]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, recon_rgb=recon.astype(np.float32))


def collect_metrics_rows(metrics_dir: Path):
    rows = []
    for p in sorted(metrics_dir.glob('*.json')):
        try:
            rows.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception:
            pass
    return rows


def save_summary_csv(rows, path: Path):
    if not rows:
        return
    keys = ['rgb_case', 'method', 'observed_ratio', 'psnr_full', 'psnr_missing', 'ssim_frame_mean', 'ssim_frame_std', 'mae_missing', 'temporal_gradient_mae_missing']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})


def main():
    ap = argparse.ArgumentParser(description='Run 9-method structured-mask color benchmark.')
    ap.add_argument('--manifest', default=str(ROOT/'data_color_structured9/manifest.csv'))
    ap.add_argument('--videos', nargs='*', default=['Bus', 'Container', 'Coastguard'])
    ap.add_argument('--masks', nargs='*', default=MASK_TYPES)
    ap.add_argument('--methods', nargs='*', default=METHODS_ORDER)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--stop-on-error', action='store_true')
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    wanted_videos = {v.lower() for v in args.videos}
    wanted_masks = {m.lower() for m in args.masks}
    cfg = load_method_defaults()

    RESULTS_CH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_RGB_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    status_path = LOGS_DIR / 'run_status.csv'
    new = not status_path.exists()
    total_jobs = sum(1 for r in rows if r.get('video', '').lower() in wanted_videos and str(r.get('enabled', '1')).strip().lower() not in {'0','false','no'}) * len(wanted_masks) * len(args.methods) * len(CHANNELS)
    job = 0
    matlab_required = any(m in {'TNN', 'KBR', 'SALTS'} for m in args.methods)
    if matlab_required and shutil.which('matlab') is None:
        raise SystemExit('MATLAB not found in PATH, but TNN/KBR/SALTS are requested.')

    failures = []
    with status_path.open('a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['timestamp', 'video', 'mask_type', 'method', 'channel', 'status', 'returncode', 'wall_seconds', 'output'])
        for row in rows:
            enabled = str(row.get('enabled', '1')).strip().lower() not in {'0', 'false', 'no'}
            video = row.get('video', '').strip()
            if not enabled or video.lower() not in wanted_videos:
                continue
            for mask_type in args.masks:
                if mask_type.lower() not in wanted_masks:
                    continue
                rgb_case_npz = CASES_RGB_DIR / f'{video}__{mask_type}.npz'
                if not rgb_case_npz.exists():
                    raise FileNotFoundError(f'Prepared RGB case not found: {rgb_case_npz}. Run prepare_color_cases.py first.')
                for method in args.methods:
                    per_method_dir = RESULTS_CH_DIR / method
                    per_method_dir.mkdir(parents=True, exist_ok=True)
                    need_run = False
                    for cname, _ in CHANNELS:
                        out = per_method_dir / f'{method}__{video}__{mask_type}__{cname}.mat'
                        if not valid_file(out) or args.overwrite:
                            need_run = True
                    for cname, _ in CHANNELS:
                        job += 1
                        case = CASES_DIR / f'{video}__{mask_type}__{cname}.mat'
                        out = per_method_dir / f'{method}__{video}__{mask_type}__{cname}.mat'
                        if valid_file(out) and not args.overwrite:
                            print(f'[{job}/{total_jobs}] SKIP {method} :: {video} :: {mask_type} :: {cname}')
                            writer.writerow([time.strftime('%Y-%m-%dT%H:%M:%S'), video, mask_type, method, cname, 'SKIP', 0, 0, str(out)])
                            f.flush()
                            continue
                        print('\n' + '='*90)
                        print(f'[{job}/{total_jobs}] RUN {method} :: {video} :: {mask_type} :: {cname}')
                        print('='*90)
                        t0 = time.perf_counter()
                        seed = parse_seed_from_stem(case.stem)
                        if method in {'TNN', 'KBR', 'SALTS'}:
                            cmd = build_matlab_cmd(method, case, out, seed)
                        else:
                            cmd = build_python_cmd(method, case, out, cfg)
                        r = subprocess.run(cmd, cwd=str(ROOT))
                        wall = time.perf_counter() - t0
                        status = 'DONE' if r.returncode == 0 and valid_file(out) else 'FAILED'
                        writer.writerow([time.strftime('%Y-%m-%dT%H:%M:%S'), video, mask_type, method, cname, status, r.returncode, f'{wall:.3f}', str(out)])
                        f.flush()
                        if status != 'DONE':
                            failures.append((video, mask_type, method, cname, r.returncode))
                            if args.stop_on_error:
                                raise SystemExit(r.returncode or 1)
                    # merge and evaluate RGB
                    out_rgb = RESULTS_RGB_DIR / method / f'{method}__{video}__{mask_type}.npz'
                    metrics_json = METRICS_DIR / f'metrics__{method}__{video}__{mask_type}.json'
                    if (not valid_file(out_rgb)) or args.overwrite or need_run:
                        merge_rgb_group(video, mask_type, method, rgb_case_npz, out_rgb, per_method_dir)
                    if (not metrics_json.exists()) or args.overwrite or need_run:
                        eval_cmd = [sys.executable, str(ROOT/'color_structured9/evaluate_rgb_case.py'), '--rgb-case', str(rgb_case_npz), '--recon', str(out_rgb), '--method', method, '--out-json', str(metrics_json)]
                        er = subprocess.run(eval_cmd, cwd=str(ROOT))
                        if er.returncode != 0:
                            failures.append((video, mask_type, method, 'RGB_EVAL', er.returncode))
                            if args.stop_on_error:
                                raise SystemExit(er.returncode)

    summary_rows = collect_metrics_rows(METRICS_DIR)
    save_summary_csv(summary_rows, METRICS_DIR / 'summary_all.csv')
    if failures:
        print('\n[WARN] Some jobs failed:')
        for x in failures:
            print('  ', x)
    else:
        print('\nAll requested color-structured jobs finished successfully.')


if __name__ == '__main__':
    main()
