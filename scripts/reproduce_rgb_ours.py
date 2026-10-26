#!/usr/bin/env python3
"""Prepare DAVIS RGB 360p fixed-mask cases and fit our frozen KPD method only.

External baselines are deliberately not executed or included.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261014


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--davis-root',type=Path,required=True,help='DAVIS 2017 dataset root (JPEGImages/480p).')
    ap.add_argument('--videos',default='camel,parkour',help='Comma-separated DAVIS sequence names.')
    ap.add_argument('--work-dir',type=Path,default=Path('local_runs/rgb360'))
    ap.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    ap.add_argument('--stage',choices=['prepare','fit','both'],default='both')
    ap.add_argument('--dry-run',action='store_true',help='Show commands, do not compute.')
    a=ap.parse_args(argv)
    vids=[s.strip() for s in a.videos.split(',') if s.strip()]
    if not vids or len(set(vids))!=len(vids):ap.error('Supply distinct DAVIS sequence names.')
    if a.stage in ('fit','both') and a.device == 'auto':
        # Downstream frozen_benchmark accepts auto; the solver selects an available device.
        pass
    work=a.work_dir.expanduser().resolve()
    data=work/'rgb_data'; cases=work/'prepared_cases'; output=work/'ours_frozen'
    empty_v12=work/'unused_v12'; empty_v14=work/'unused_v14'
    commands=[]
    if a.stage in ('prepare','both'):
        commands.append([sys.executable,str(ROOT/'experiments/10_davis2017_rgb_all9/prepare_davis_rgb.py'),
                         '--davis-root',str(a.davis_root.expanduser().resolve()),'--out',str(data),
                         '--sequences',','.join(vids),'--height','360','--width','640','--frames','50'])
        commands.append([sys.executable,str(ROOT/'experiments/10_davis2017_rgb_all9/rgb9.py'),
                         'prepare','--data',str(data),'--output',str(cases),'--videos',','.join(vids),
                         '--rates','0.2,0.3,0.4','--seed',str(SEED)])
    if a.stage in ('fit','both'):
        commands.append([sys.executable,str(ROOT/'experiments/17_frozen_rgb_360p_reuse/frozen_benchmark.py'),
                         '--source',str(cases),'--v12-results',str(empty_v12),
                         '--v14-results',str(empty_v14),'--output',str(output),
                         '--videos',','.join(vids),'--rates','0.2,0.3,0.4',
                         '--device',a.device,'--fit-only','--force-retrain','--stop-on-error'])
    for i,cmd in enumerate(commands,1):
        print(f'STEP {i}/{len(commands)}: '+subprocess.list2cmdline(cmd),flush=True)
        if not a.dry_run:
            subprocess.run(cmd,cwd=ROOT,check=True)
    if a.dry_run:
        print('DRY RUN ONLY. No source data or results generated.',flush=True)
    else:
        print('Completed requested stages. Our reconstructions are in',output,flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
