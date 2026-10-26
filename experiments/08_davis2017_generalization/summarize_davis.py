#!/usr/bin/env python3
"""Aggregate completed new-video experiments without ever fabricating missing results."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path

METRICS=('psnr_full','psnr_missing','ssim_full','mae_missing','temporal_gradient_mae','duration_total_seconds')

def save_csv(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    a=p.parse_args()
    root=Path(a.input).resolve()
    cfg=json.loads((root/'RUN_CONFIG.json').read_text(encoding='utf-8'))
    items=[json.loads(f.read_text(encoding='utf-8')) for f in sorted((root/'per_case').glob('*.json'))]
    total=len(cfg['video_names'])*len(cfg['rates'])*len(cfg['seeds'])*len(cfg['methods'])
    if not items:raise SystemExit('No completed cases; do not report unrun experiments.')
    complete_keys={(i['name'],round(i['rate_requested'],5),i['seed_requested'],i['method']) for i in items}
    expected={(n,round(sr,5),s,m) for n in cfg['video_names'] for sr in cfg['rates'] for s in cfg['seeds'] for m in cfg['methods']}
    missing=sorted(expected-complete_keys)
    print(f'Completed {len(complete_keys)}/{len(expected)} cases')
    if missing:print('INCOMPLETE; remaining cases:', missing[:12], '...', len(missing))
    if cfg['smoke_test']:
        print('SMOKE TEST RESULTS ARE NOT SCIENTIFIC EVIDENCE. DO NOT INCLUDE IN PAPER.')
    save_csv(root/'PER_CASE_METRICS.csv',items)
    agg=[]
    for sr in cfg['rates']:
        for meth in cfg['methods']:
            subset=[r for r in items if abs(r['rate_requested']-sr)<1e-8 and r['method']==meth]
            if not subset:continue
            row=dict(SR=sr,method=meth,n_cases=len(subset),n_videos=len({r['name'] for r in subset}),n_seeds=len({r['seed_requested'] for r in subset}))
            for metric in METRICS:
                numbers=[float(r[metric]) for r in subset]
                row[metric+'_mean']=statistics.mean(numbers)
                row[metric+'_std']=statistics.stdev(numbers) if len(numbers)>1 else 0.0
            agg.append(row)
    save_csv(root/'AGGREGATE.csv',agg)
    paired=[]
    for sr in cfg['rates']:
        for n in cfg['video_names']:
            for seed in cfg['seeds']:
                rr={r['method']:r for r in items if abs(r['rate_requested']-sr)<1e-8 and r['name']==n and r['seed_requested']==seed}
                if not {'CROSSFIT','DIRECT'}<=rr.keys():continue
                a1,a2=rr['CROSSFIT'],rr['DIRECT']
                paired.append(dict(SR=sr,video=n,seed=seed,
                    delta_psnr_full=a1['psnr_full']-a2['psnr_full'],
                    delta_psnr_missing=a1['psnr_missing']-a2['psnr_missing'],
                    delta_ssim=a1['ssim_full']-a2['ssim_full'],
                    crossfit_wins_full_psnr=int(a1['psnr_full']>a2['psnr_full'])))
    save_csv(root/'PAIRED_CROSSFIT_VS_DIRECT.csv',paired)
    per_video=[]
    for sr in cfg['rates']:
        for video in cfg['video_names']:
            for meth in cfg['methods']:
                rr=[r for r in items if abs(r['rate_requested']-sr)<1e-8 and r['name']==video and r['method']==meth]
                if not rr:continue
                rec=dict(SR=sr,video=video,method=meth,n_seeds=len(rr))
                for metric in METRICS:
                    rec[metric+'_mean']=statistics.mean(float(r[metric]) for r in rr)
                per_video.append(rec)
    save_csv(root/'PER_VIDEO_MEANS.csv',per_video)
    gains=[]
    for sr in cfg['rates']:
        for video in cfg['video_names']:
            rr=[r['delta_psnr_missing'] for r in paired if r['SR']==sr and r['video']==video]
            if rr:
                gains.append(dict(SR=sr,video=video,n_seeds=len(rr),delta_missing_psnr_mean=statistics.mean(rr),
                                  fraction_positive=sum(x>0 for x in rr)/len(rr)))
    save_csv(root/'PAIRED_GAIN_BY_VIDEO.csv',gains)
    print('Written:',root/'PER_CASE_METRICS.csv',root/'AGGREGATE.csv',root/'PAIRED_CROSSFIT_VS_DIRECT.csv',root/'PER_VIDEO_MEANS.csv',root/'PAIRED_GAIN_BY_VIDEO.csv')
    if missing or cfg['smoke_test']:
        print('Paper reporting blocked until all cases complete under the full frozen settings.')
    else:
        print('All requested frozen-protocol cases completed. Report mean over ALL videos and seeds, not only wins.')

if __name__=='__main__':main()
