#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only v11 progress viewer: file existence, not numerical validation by default."""
import argparse,time
from pathlib import Path
from steps_patch_rgb360 import (_candidate_v10,DEFAULT_OUT,ALL_CONFIGS,NEW_CONFIGS,case_id,
                                baseline_rgb,rgb_path,channel_path,legacy,_valid_baseline,_valid_rgb)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--v10-results',type=Path)
    p.add_argument('--videos',default='blackswan')
    p.add_argument('--rates',default='0.3')
    p.add_argument('--watch',type=int,default=0)
    p.add_argument('--validate',action='store_true',help='Open and validate all MAT files (slow)')
    a=p.parse_args()
    vids=legacy.names(a.videos);rates=legacy.nums(a.rates,float,(.2,.3,.4))
    v10=_candidate_v10(a.v10_results)
    while True:
        count=0;missing=[]
        for video in vids:
            for sr in rates:
                case=case_id(video,sr)
                for cfg,(steps,patch) in ALL_CONFIGS.items():
                    f=baseline_rgb(v10,case) if cfg=='A0' else rgb_path(a.output,case,steps,patch)
                    ok=(_valid_baseline(f) if cfg=='A0' else _valid_rgb(f,steps,patch)) if a.validate else (f.is_file() and f.stat().st_size>100)
                    if ok:count+=1
                    else:
                        channel_done=sum(channel_path(a.output,case,steps,patch,ch).is_file() for ch in legacy.CHANNELS) if cfg!='A0' else 0
                        missing.append((video,sr,cfg,f'{channel_done}/3 channels'))
        total=count+len(missing)
        print(time.strftime('[%H:%M:%S]'),f'RGB configs: {count}/{total} ({100*count/max(total,1):.1f}%)',
              'MAT validated' if a.validate else 'file existence only',flush=True)
        for r in missing[:10]:print('  NEXT:',r,flush=True)
        ev=a.output/'events.jsonl'
        if ev.is_file():
            last=ev.read_text(encoding='utf-8').splitlines()
            if last:print('  LAST:',last[-1][:250],flush=True)
        if a.watch<=0:break
        time.sleep(max(5,a.watch))

if __name__=='__main__':main()
