#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple read-only progress monitor for the rank x spatial-kernel experiment."""
import argparse, time
from pathlib import Path
from rank_kernel_rgb360 import VIDEO_NAMES, stem, rgbfile, is_valid

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'results/rank_kernel_rgb360p_seed20261014')
    p.add_argument('--videos',default='blackswan')
    p.add_argument('--rates',default='0.3')
    p.add_argument('--ranks',default='4,8,12')
    p.add_argument('--kernels',default='3,5')
    p.add_argument('--watch',type=int,default=0,help='seconds between updates, 0 for once')
    a=p.parse_args()
    vids=list(VIDEO_NAMES) if a.videos=='all' else [s.strip() for s in a.videos.split(',')]
    rates=[float(x) for x in a.rates.split(',')];ranks=[int(x) for x in a.ranks.split(',')];kernels=[int(x) for x in a.kernels.split(',')]
    while True:
        todo=[];done=0
        for v in vids:
            for r in rates:
                for k in ranks:
                    for kern in kernels:
                        f=rgbfile(a.output,stem(v,r),k,kern)
                        if is_valid(f,(360,640,50,3)):done+=1
                        else:todo.append((v,r,k,kern))
        total=done+len(todo)
        print(f'Completed {done}/{total} RGB settings ({100*done/max(1,total):.1f}%), remaining {len(todo)}',flush=True)
        for v,r,k,kern in todo[:6]:print(f'  pending {v} SR={r:.1f} K={k} spatial={kern}',flush=True)
        if not a.watch:break
        time.sleep(a.watch)
if __name__=='__main__':main()
