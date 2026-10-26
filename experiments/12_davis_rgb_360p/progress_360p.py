#!/usr/bin/env python3
"""Monitor completed RGB .mat outputs without reading the large .mat arrays."""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'results'/'davis2017_rgb_all9_seed20261014_360p_f050'
NAMES=['DRO-TFF','HLRTF','KBR','LRTFR','OTLRM','TNN','SALTS','S2NTNN','Ours']
def show(out):
    idx=out/'cases_index.csv'
    if not idx.exists():print('Not prepared:',idx);return
    with idx.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    done=[];todo=[]
    for row in rows:
        stem=row['case']
        for method in NAMES:
            f=out/'reconstructions'/method/(method+'__'+stem+'.mat')
            (done if f.exists() and f.stat().st_size>200 else todo).append((stem,method))
    print(f'[{time.strftime("%H:%M:%S")}] Completed RGB results: {len(done)}/{len(rows)*9} ({100*len(done)/max(1,len(rows)*9):.1f}%), prepared cases={len(rows)}')
    if todo:print('Next uncompleted:',*todo[:5],sep='\n  ')
    for v in sorted(set(r['name'] for r in rows)):
        log=out/'video_logs'/(v+'.log')
        if log.exists():
            txt=log.read_text(encoding='utf-8',errors='replace').splitlines()
            important=[s for s in txt if s.startswith(('RUN','OK','FAILED','SKIP RGB','RGB COMPLETE'))]
            if important:print(' ',v,':',important[-1][:170])
    print('NOTE: Progress counts existing RGB output files; evaluation validates their numerical contents.')
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--watch',type=int,default=0,help='refresh interval seconds, 0=once')
    a=p.parse_args()
    try:
        while True:
            show(a.output)
            if a.watch<=0:break
            time.sleep(a.watch)
    except KeyboardInterrupt:pass
if __name__=='__main__':main()
