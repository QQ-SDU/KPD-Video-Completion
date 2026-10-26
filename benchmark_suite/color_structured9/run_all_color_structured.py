from __future__ import annotations
import argparse, os, secrets, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from color_structured9.settings import MASK_TYPES, METHODS_ORDER



def run(cmd, title):
    print('\n' + '#' * 96)
    print(title)
    print('#' * 96)
    print('>', ' '.join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description='One-shot pipeline for Bus/Container/Coastguard structured-mask comparison.')
    ap.add_argument('--videos', nargs='*', default=['Bus', 'Container', 'Coastguard'])
    ap.add_argument('--masks', nargs='*', default=MASK_TYPES)
    ap.add_argument('--methods', nargs='*', default=METHODS_ORDER)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--seed', type=int, default=None, help='Optional reproducibility seed; omitted => generate a fresh seed.')
    args = ap.parse_args()
    seed = int(args.seed) if args.seed is not None else secrets.randbits(32)
    os.environ["KPD_COLOR_SEED"] = str(seed)
    print(f"Run seed: {seed}")

    run([sys.executable, str(ROOT/'color_structured9/check_color_data.py')], '1/4 Check source videos and manifest')
    prep = [sys.executable, str(ROOT/'color_structured9/prepare_color_cases.py'), '--videos', *args.videos, '--masks', *args.masks, '--seed', str(seed)]
    if args.overwrite:
        prep.append('--force')
    run(prep, '2/4 Prepare RGB and per-channel benchmark cases')
    suite = [sys.executable, str(ROOT/'color_structured9/run_color_suite.py'), '--videos', *args.videos, '--masks', *args.masks, '--methods', *args.methods]
    if args.overwrite:
        suite.append('--overwrite')
    run(suite, '3/4 Run 9 methods and evaluate RGB reconstructions')
    figs = [sys.executable, str(ROOT/'color_structured9/make_color_figure.py'), '--videos', *args.videos, '--masks', *args.masks, '--methods', *args.methods]
    run(figs, '4/4 Build qualitative figures (Observed + 9 methods + GT)')
    print('\nPipeline finished. Outputs are under artifacts_color_structured9/.')


if __name__ == '__main__':
    main()