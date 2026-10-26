#!/usr/bin/env python3
"""Fixed DAVIS 360x640 RGB benchmark: 50 frames, 9 methods, SR .2/.3/.4, seed 20261014."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'11_davis2017_rgb_highres'))
from run_highres_all9 import main as highres_main

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',default='blackswan',help='blackswan, camel, blackswan,camel or all')
    p.add_argument('--jobs',type=int,default=1,help='number of videos processed concurrently (recommended 1)')
    p.add_argument('--davis-root',default=os.environ.get('DAVIS_ROOT',str(Path.home() / "kpd_data" / 'DAVIS')))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--skip-preprocess',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stop-on-error',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--no-evaluate',action='store_true')
    p.add_argument('--check-gpu',action='store_true')
    a=p.parse_args(argv)
    cmd=['--size','360p','--frames','50','--videos',a.videos,'--jobs',str(a.jobs),'--davis-root',a.davis_root]
    for flag in ['prepare-only','skip-preprocess','dry-run','stop-on-error','overwrite','no-evaluate','check-gpu']:
        if getattr(a,flag.replace('-','_')):cmd.append('--'+flag)
    return highres_main(cmd)
if __name__=='__main__':raise SystemExit(main())
