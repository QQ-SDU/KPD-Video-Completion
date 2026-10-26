from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from color_structured9.color_video_io import load_manifest
from color_structured9.settings import INPUT_DIR, MANIFEST



def main():
    ap = argparse.ArgumentParser(description='Check color structured benchmark inputs.')
    ap.add_argument('--manifest', default=str(MANIFEST))
    ap.add_argument('--input-dir', default=str(INPUT_DIR))
    args = ap.parse_args()
    rows = load_manifest(args.manifest)
    ok = True
    print('Input dir:', Path(args.input_dir).resolve())
    for row in rows:
        enabled = str(row.get('enabled', '1')).strip().lower() not in {'0', 'false', 'no'}
        p = Path(args.input_dir) / row.get('file', '')
        print(f"- {row.get('video','?'):12s} | enabled={enabled} | exists={p.exists()} | {p}")
        if enabled and not p.exists():
            ok = False
    if not ok:
        raise SystemExit('Some enabled source videos are missing. Please fill data_color_structured9/inputs and edit manifest.csv if needed.')
    print('All enabled source files are present.')


if __name__ == '__main__':
    main()