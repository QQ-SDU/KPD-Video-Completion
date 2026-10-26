#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only progress, fast output-existence scan (without loading large MAT files)."""
import argparse
import time
from pathlib import Path
from rank_width_rgb360 import (ROOT,DEFAULT_OUTPUT,DEFAULT_V9,candidate_v9,old,
                               ALL_CONFIGS,SHAPE,config_id,new_rgb,new_channel,v9_rgb,valid)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--v9-results',type=Path)
    p.add_argument('--videos',default='blackswan')
    p.add_argument('--rates',default='0.3')
    p.add_argument('--watch',type=int,default=0)
    p.add_argument('--validate',action='store_true',help='Expensive: read and check full RGB MAT arrays')
    a=p.parse_args()
    vids=old.names(a.videos)
    rates=old.nums(a.rates,float,(0.2,0.3,0.4))
    v9=candidate_v9(a.v9_results)
    while True:
        done=0;pending=[]
        for v in vids:
            for sr in rates:
                case=old.stem(v,sr)
                for k,h in ALL_CONFIGS:
                    path=v9_rgb(v9,case) if (k,h)==(12,12) else new_rgb(a.output,case,k,h)
                    ok=(valid(path) if a.validate else path.is_file() and path.stat().st_size>100)
                    if ok:done+=1
                    else:
                        partial=sum(new_channel(a.output,case,k,h,c).is_file() for c in old.CHANNELS) if (k,h)!=(12,12) else 0
                        pending.append((v,sr,config_id(k,h),partial))
        total=done+len(pending)
        print(time.strftime('[%H:%M:%S]'),f'RGB results {done}/{total} ({done/max(1,total)*100:.1f}%)',
              '[validated]' if a.validate else '[file existence only]',flush=True)
        for row in pending[:8]:print('  NEXT',row,flush=True)
        if a.output.joinpath('events.jsonl').is_file():
            lines=a.output.joinpath('events.jsonl').read_text(encoding='utf-8').splitlines()
            if lines:print('  LAST:',lines[-1][:240],flush=True)
        if not a.watch:break
        time.sleep(max(5,a.watch))
if __name__=='__main__':main()
