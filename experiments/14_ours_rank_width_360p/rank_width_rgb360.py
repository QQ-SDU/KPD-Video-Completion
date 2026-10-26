#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""360p RGB: five new KPD-rank/CNN-width configs and reusable v9 baseline.

Comparison:
  A1  K=12, hidden=12  [v9 reference, read only]
  A2  K=16, hidden=12  [NEW]
  A3  K=20, hidden=12  [NEW]
  A4  K=24, hidden=12  [NEW]
  B1  K=12, hidden=24  [NEW]
  B2  K=12, hidden=32  [NEW]

Fixed: spatial kernel 3, temporal kernel 3, 1200 steps, 5-fold CrossFit,
16x48x48 patches, seed 20261014, identical prepared 360p tube-wise masks.
Model process loads observed/mask ONLY. Ground truth is read by evaluation ONLY.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments'/'13_ours_rank_kernel_360p'))
import rank_kernel_rgb360 as old

SEED=20261014
SHAPE=(360,640,50,3)
NEW_CONFIGS=((16,12),(20,12),(24,12),(12,24),(12,32))
ALL_CONFIGS=((12,12),)+NEW_CONFIGS
DEFAULT_SOURCE=Path(os.environ.get('DAVIS_360P_SOURCE',str(Path.home() / "kpd_data" / 'KPD' / 'results' / 'davis2017_rgb_all9_seed20261014_360p_f050')))
DEFAULT_OUTPUT=ROOT/'results'/'rank_width_rgb360p_seed20261014'
DEFAULT_V9=Path(os.environ.get('KPD_V9_RESULTS',str(Path.home() / "kpd_data" / 'KPD_RankKernel_v9' / 'results' / 'rank_kernel_rgb360p_seed20261014')))


def config_id(k,h):return f'K{k:02d}_C{h:02d}_S3'
def new_rgb(out,case,k,h):return out/'rgb'/config_id(k,h)/(case+'.mat')
def new_channel(out,case,k,h,ch):return out/'channels'/config_id(k,h)/(case+f'__{ch}.mat')
def v9_rgb(v9,case):return v9/'rgb'/'K12_S3'/(case+'.mat')

def candidate_v9(cli):
    """Explicit path wins; otherwise discover common v9 output locations."""
    options=([cli] if cli is not None else [DEFAULT_V9,ROOT/'results'/'rank_kernel_rgb360p_seed20261014',
                                      Path(str(Path.home() / "kpd_data" / 'KPD' / 'results' / 'rank_kernel_rgb360p_seed20261014'))])
    for p in options:
        if p is not None and p.is_dir():return p.resolve()
    return options[0].resolve()

def valid(path,shape=None):return old.is_valid(path,SHAPE if shape is None else shape)

def _append_event(out, record):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n')


def _runtime_value(mat):
    val=mat.get('runtime_seconds')
    if val is None:return float('nan')
    return float(np.asarray(val).ravel()[0])


def run_one(src,out,video,sr,k,h,args):
    case=old.stem(video,sr)
    case_file=old.casefile(src,case)
    if not case_file.is_file():raise FileNotFoundError('Prepared RGB case missing: '+str(case_file))
    final=new_rgb(out,case,k,h)
    if valid(final) and not args.overwrite:
        print('SKIP COMPLETE:',config_id(k,h),case,flush=True)
        return True
    for ch in old.CHANNELS:
        inp=old.cfile(src,case,ch)
        if not inp.is_file():raise FileNotFoundError('Prepared channel missing: '+str(inp))
        dst=new_channel(out,case,k,h,ch)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if valid(dst,SHAPE[:3]) and not args.overwrite:
            print('SKIP CHANNEL:',config_id(k,h),ch,case,flush=True)
            continue
        tmp=dst.with_name(dst.stem+'.partial.mat')
        cmd=[sys.executable,str(old.METHOD),'--case',str(inp),'--out',str(tmp),
             '--seed',str(SEED),'--rank',str(k),'--hidden',str(h),'--spatial-kernel','3',
             '--steps','1200','--crossfit-folds','5',
             '--patch-t','16','--patch-h','48','--patch-w','48','--device',args.device]
        log=out/'logs'/config_id(k,h)/(case+f'__{ch}.log')
        print('RUN',config_id(k,h),case,ch,'\n Log:',log,flush=True)
        if args.dry_run:
            print('CMD:',subprocess.list2cmdline(cmd),flush=True)
            continue
        log.parent.mkdir(parents=True,exist_ok=True)
        if tmp.exists():tmp.unlink()
        side=tmp.with_suffix('.json')
        if side.exists():side.unlink()
        t0=time.monotonic()
        with log.open('w',encoding='utf-8') as f:
            f.write('CMD '+subprocess.list2cmdline(cmd)+'\n');f.flush()
            p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        secs=time.monotonic()-t0
        passed=p.returncode==0 and valid(tmp,SHAPE[:3])
        _append_event(out,dict(case=case,rank=k,hidden=h,channel=ch,
                               status='OK' if passed else 'FAILED',seconds=round(secs,2),log=str(log)))
        if not passed:
            print('FAILED:',config_id(k,h),ch,'exit=',p.returncode,'log=',log,flush=True)
            return False
        os.replace(tmp,dst)
        if side.exists():os.replace(side,dst.with_suffix('.json'))
        print('OK',config_id(k,h),ch,f'{secs:.1f}s',flush=True)
    if args.dry_run:return True
    # Assemble 3 color predictions, then project measured entries identically to v9.
    gt,obs,mask=old.load_truth(src,case)
    if gt.shape!=SHAPE:raise ValueError(f'Case {case} has shape {gt.shape}, expected {SHAPE}')
    rec=np.empty(SHAPE,dtype=np.float32)
    rt=0.0
    for ci,ch in enumerate(old.CHANNELS):
        d=loadmat(new_channel(out,case,k,h,ch),variable_names=['recon','runtime_seconds'])
        pred=np.asarray(d['recon'],dtype=np.float32)
        if pred.shape!=SHAPE[:3]:raise ValueError('Channel shape differs from protocol')
        rec[:,:,:,ci]=np.clip(pred,0,1)
        rt+=_runtime_value(d)
        del d,pred
    rec[mask]=obs[mask]
    final.parent.mkdir(parents=True,exist_ok=True)
    tmp=final.with_name(final.stem+'.partial.mat')
    savemat(tmp,{'recon':rec,'runtime_seconds':np.array([[rt]])},do_compression=True)
    if not valid(tmp):raise RuntimeError('RGB reconstruction did not pass validation')
    os.replace(tmp,final)
    _append_event(out,dict(case=case,rank=k,hidden=h,channel='RGB',status='OK',seconds=rt))
    print('RGB COMPLETE',config_id(k,h),case,flush=True)
    del rec,gt,obs,mask
    gc.collect()
    return True


def _write_csv(path,rows,cols):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        w.writeheader();w.writerows(rows)

ROW_COLS=['video','sampling_rate','seed','rank','hidden','spatial_kernel','temporal_kernel','training_steps',
          'configuration','source_version','split_role','runtime_seconds','psnr_full',
          'psnr_missing','ssim_full','mae_missing']
SUMMARY_COLS=['sampling_rate','rank','hidden','spatial_kernel','configuration','completed_videos',
              'psnr_missing_mean','ssim_full_mean','runtime_seconds_mean',
              'psnr_missing_std_across_videos']

def evaluate(src,out,v9,videos,rates,include_legacy=True):
    rows=[]; missing=[]
    for video in videos:
        for sr in rates:
            case=old.stem(video,sr)
            if not old.casefile(src,case).exists():
                print('SKIP evaluation: prepared case missing',case)
                continue
            gt=obs=mask=None
            for k,h in ALL_CONFIGS:
                if (k,h)==(12,12) and not include_legacy:continue
                path=v9_rgb(v9,case) if (k,h)==(12,12) else new_rgb(out,case,k,h)
                if not valid(path):
                    missing.append(dict(video=video,sampling_rate=sr,rank=k,hidden=h,
                                        expected_rgb=str(path),status='MISSING'))
                    continue
                if gt is None:
                    gt,obs,mask=old.load_truth(src,case)
                    if gt.shape!=SHAPE:raise ValueError(f'Bad source shape: {gt.shape}')
                d=loadmat(path,variable_names=['recon','runtime_seconds'])
                recon=np.asarray(d['recon'],dtype=np.float32)
                m=old.metric(gt,obs,mask,recon)
                rows.append(dict(video=video,sampling_rate=sr,seed=SEED,rank=k,hidden=h,
                                 spatial_kernel=3,temporal_kernel=3,training_steps=1200,
                                 configuration=config_id(k,h),source_version='v9_reference' if (k,h)==(12,12) else 'v10',
                                 split_role='development',runtime_seconds=_runtime_value(d),**m))
                print(f'[{video} SR={sr:.1f}] K={k}, C={h}, PSNR_missing={m["psnr_missing"]:.4f} dB; SSIM={m["ssim_full"]:.5f}')
                del d,recon
            del gt,obs,mask
            gc.collect()
    # Merge incremental CSV rows; never mix configurations or replace other cases.
    output_csv=out/'results_per_video.csv'
    previous=[]
    if output_csv.is_file():
        with output_csv.open('r',encoding='utf-8-sig',newline='') as f:
            previous=list(csv.DictReader(f))
    def key(r):
        return (r['video'],round(float(r['sampling_rate']),6),int(r['seed']),int(r['rank']),int(r['hidden']))
    # Selected video/rate rows are rebuilt from currently verified MAT files.
    # This prevents stale A1 rows if the user corrects the v9 reference path.
    selected={(v,round(float(sr),6)) for v in videos for sr in rates}
    merged={key(x):x for x in previous if (x['video'],round(float(x['sampling_rate']),6)) not in selected}
    merged.update({key(x):x for x in rows})
    allrows=sorted(merged.values(),key=key)
    _write_csv(output_csv,allrows,ROW_COLS)
    _write_csv(out/'missing_configs.csv',missing,['video','sampling_rate','rank','hidden','expected_rgb','status'])
    summary=[]
    combos=sorted({(float(r['sampling_rate']),int(r['rank']),int(r['hidden'])) for r in allrows})
    for sr,k,h in combos:
        subset=[r for r in allrows if float(r['sampling_rate'])==sr and int(r['rank'])==k and int(r['hidden'])==h]
        p=np.asarray([float(r['psnr_missing']) for r in subset])
        summary.append(dict(sampling_rate=sr,rank=k,hidden=h,spatial_kernel=3,
                            configuration=config_id(k,h),completed_videos=len(subset),
                            psnr_missing_mean=float(p.mean()),
                            ssim_full_mean=float(np.mean([float(r['ssim_full']) for r in subset])),
                            runtime_seconds_mean=float(np.mean([float(r['runtime_seconds']) for r in subset])),
                            psnr_missing_std_across_videos=float(p.std(ddof=1)) if len(p)>1 else ''))
    _write_csv(out/'summary_rank_width.csv',summary,SUMMARY_COLS)
    print(f'EVALUATED: {len(rows)} current selected; {len(allrows)} stored records;'
          f' {len(missing)} missing among selected; output: {output_csv}',flush=True)
    return len(missing)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=out/'summary_rank_width.csv'
    if not path.is_file():
        print('Nothing to plot yet:',path);return
    with path.open('r',newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if not rows:return
    figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    for sr in sorted({float(r['sampling_rate']) for r in rows}):
        for mode in ('rank','width'):
            if mode=='rank':
                subset=sorted([r for r in rows if float(r['sampling_rate'])==sr and int(r['hidden'])==12],key=lambda r:int(r['rank']))
                xs=[int(r['rank']) for r in subset]; xlabel='KPD rank K  (CNN width = 12)'
            else:
                subset=sorted([r for r in rows if float(r['sampling_rate'])==sr and int(r['rank'])==12],key=lambda r:int(r['hidden']))
                xs=[int(r['hidden']) for r in subset];xlabel='CNN hidden channels C  (KPD rank = 12)'
            if not subset:continue
            fig,ax=plt.subplots(figsize=(5.3,3.2))
            ys=[float(r['psnr_missing_mean']) for r in subset]
            ax.plot(xs,ys,'o-',linewidth=1.8)
            ax.set_xticks(xs);ax.set_xlabel(xlabel);ax.set_ylabel('Missing-region PSNR (dB)')
            ax.grid(alpha=.25)
            for xx,yy,rr in zip(xs,ys,subset):
                ax.annotate(f'{yy:.2f}  (n={rr["completed_videos"]})',(xx,yy),
                            textcoords='offset points',xytext=(0,7),ha='center',fontsize=8)
            if len({r['completed_videos'] for r in subset})>1:
                ax.set_title('PARTIAL: unequal video counts across configurations',fontsize=9)
            fig.tight_layout()
            name=figdir/f'{mode}_PSNR_SR{int(round(sr*100)):03d}'
            fig.savefig(name.with_suffix('.pdf'),bbox_inches='tight')
            fig.savefig(name.with_suffix('.png'),dpi=300,bbox_inches='tight')
            plt.close(fig)
            print('PLOT',name.with_suffix('.pdf'))


def _check_protocol(out,source,overwrite=False):
    protocol=out/'RANK_WIDTH_PROTOCOL.json'
    expected=dict(shape=list(SHAPE),seed=SEED,source=str(source),
                  configs=[[k,h] for k,h in ALL_CONFIGS],spatial_kernel=3,temporal_kernel=3,
                  steps=1200,crossfit_folds=5,patch=[16,48,48],
                  batch_size=2,masked_sweeps=12,max_residual=0.25,
                  method='KPD+CrossFit, observed/mask only; v9 baseline read-only')
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8'))!=expected:
        raise ValueError('Protocol differs from existing results directory. Choose another --output.')
    out.mkdir(parents=True,exist_ok=True)
    protocol.write_text(json.dumps(expected,indent=2),encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-output',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--v9-results',type=Path,default=None,help='v9 output root holding rgb/K12_S3. Baseline is read only.')
    p.add_argument('--videos',default='blackswan')
    p.add_argument('--rates',default='0.3')
    p.add_argument('--configs',default='all',help='all, or A2,A3,A4,B1,B2 (select NEW configs only)')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--no-evaluate',action='store_true')
    p.add_argument('--no-v9-baseline',action='store_true')
    p.add_argument('--plot-only',action='store_true')
    p.add_argument('--stop-on-error',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args(argv)
    if args.evaluate_only and args.no_evaluate:p.error('Conflicting evaluation arguments.')
    try:
        videos=old.names(args.videos)
        rates=old.nums(args.rates,float,(0.2,0.3,0.4))
    except ValueError as exc:p.error(str(exc))
    config_map=dict(A2=(16,12),A3=(20,12),A4=(24,12),B1=(12,24),B2=(12,32))
    names=list(config_map) if args.configs=='all' else [x.strip().upper() for x in args.configs.split(',') if x.strip()]
    if not names or len(set(names))!=len(names) or set(names)-set(config_map):p.error('Choose --configs all or A2,A3,A4,B1,B2')
    selected=[config_map[n] for n in names]
    src=args.source_output.expanduser().resolve()
    out=args.output.expanduser().resolve()
    v9=candidate_v9(args.v9_results)
    print('OURS Rank × Width v10 | 360x640x50 RGB | SR:',rates)
    print('SOURCE:',src,'\nOUTPUT:',out,'\nV9 REFERENCE (read only):',v9)
    print('VIDEOS:',videos,'\nNEW CONFIGS:',[(config_id(k,h),n) for n,(k,h) in config_map.items() if (k,h) in selected])
    print('FIXED: kernel=3x3x3, folds=5, steps=1200, patch=16x48x48, seed=',SEED,flush=True)
    if args.plot_only:
        plot(out);return 0
    if not (src/'cases').is_dir() or not (src/'channel_cases').is_dir():
        p.error('Missing prepared cases/channel_cases under '+str(src))
    if not args.dry_run:_check_protocol(out,src)
    missing_baseline=0
    if not args.no_v9_baseline:
        for video in videos:
            for sr in rates:
                if not valid(v9_rgb(v9,old.stem(video,sr))):missing_baseline+=1
        if missing_baseline:print('WARNING v9 baseline absent for',missing_baseline,'cases. No baseline will be invented/retrained.')
    if args.dry_run:
        for v in videos:
            for sr in rates:
                for k,h in selected:
                    c=old.stem(v,sr)
                    for ch in old.CHANNELS:
                        if not valid(new_channel(out,c,k,h,ch),SHAPE[:3]) and not valid(new_rgb(out,c,k,h)):
                            cmd=[str(old.METHOD),'--rank',str(k),'--hidden',str(h),'--spatial-kernel','3','--steps','1200']
                            print('WOULD RUN',v,sr,config_id(k,h),ch,' '.join(cmd))
        return 0
    failures=[]
    if not args.evaluate_only:
        for v in videos:
            for sr in rates:
                for k,h in selected:
                    try:
                        ok=run_one(src,out,v,sr,k,h,args)
                        if not ok:failures.append((v,sr,k,h))
                        if failures and args.stop_on_error:raise RuntimeError('Stopping after failed channel: '+str(failures[-1]))
                    except Exception as exc:
                        print('ERROR',v,sr,k,h,str(exc),flush=True)
                        failures.append((v,sr,k,h))
                        if args.stop_on_error:raise
    if not args.no_evaluate:
        # Evaluate all available new configs, not just new selections, plus old v9 baseline.
        evaluate(src,out,v9,videos,rates,include_legacy=not args.no_v9_baseline)
        plot(out)
    if failures:
        print('FAILED NEW CONFIGS:',failures)
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
