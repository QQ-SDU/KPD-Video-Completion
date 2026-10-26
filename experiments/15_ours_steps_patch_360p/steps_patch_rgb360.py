#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""360p RGB Ours KPD+CrossFit: steps x spatial-patch sensitivity.

Frozen KPD rank=12; CNN width=32; spatial kernel=3, temporal kernel=3;
5 folds; patch_t=16; same sampling masks and seed as v8/v10.

A0  1200 steps, patch 48  : read-only v10 baseline, NEVER retrained.
S2  2400 steps, patch 48  : NEW
S3  3600 steps, patch 48  : NEW
P64 2400 steps, patch 64  : NEW
P96 2400 steps, patch 96  : NEW

Method subprocess reads observed/mask ONLY; GT is read by evaluation only.
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
sys.path.insert(0,str(ROOT/'experiments'/'13_ours_rank_kernel_360p'))
import rank_kernel_rgb360 as legacy

SHAPE=(360,640,50,3)
SEED=20261014
K=12
HIDDEN=32
KERNEL=3
FOLDS=5
PATCH_T=16
BATCH=2
SWEEPS=12
MAX_RESIDUAL=0.25
# Names map to (steps, spatial patch). The baseline is ALWAYS read-only.
BASELINE=('A0',1200,48)
NEW_CONFIGS={'S2':(2400,48),'S3':(3600,48),'P64':(2400,64),'P96':(2400,96)}
ALL_CONFIGS={'A0':(1200,48),**NEW_CONFIGS}
DEFAULT_SOURCE=Path(os.getenv('DAVIS_360P_SOURCE',str(Path.home() / "kpd_data" / 'KPD' / 'results' / 'davis2017_rgb_all9_seed20261014_360p_f050')))
DEFAULT_OUT=ROOT/'results'/'steps_patch_rgb360p_seed20261014'
DEFAULT_V10=Path(os.getenv('KPD_V10_RESULTS',str(Path.home() / "kpd_data" / 'KPD_RankWidth_v10' / 'results' / 'rank_width_rgb360p_seed20261014')))
V10_NAME='K12_C32_S3'


def case_id(video,sr):return legacy.stem(video,sr)
def label(steps,patch):return f'K12_C32_T{steps}_P{patch}'
def rgb_path(out,case,steps,patch):return out/'rgb'/label(steps,patch)/(case+'.mat')
def channel_path(out,case,steps,patch,ch):return out/'channels'/label(steps,patch)/(case+f'__{ch}.mat')
def baseline_rgb(base,case):return base/'rgb'/V10_NAME/(case+'.mat')
def source_channel(source,case,ch):return legacy.cfile(source,case,ch)


def _candidate_v10(specified):
    if specified is not None:return specified.expanduser().resolve()
    paths=[DEFAULT_V10,ROOT/'results'/'rank_width_rgb360p_seed20261014',
           Path(str(Path.home() / "kpd_data" / 'KPD' / 'results' / 'rank_width_rgb360p_seed20261014'))]
    return next((p.resolve() for p in paths if p.is_dir()),paths[0].resolve())


def _valid_mat(path,shape):
    if not path.is_file() or path.stat().st_size<100:return False
    try:
        a=loadmat(path,variable_names=['recon'])['recon']
        return a.shape==shape and np.issubdtype(a.dtype,np.number) and bool(np.isfinite(a).all())
    except (OSError,KeyError,ValueError,TypeError):return False


def _valid_channel(path,steps,patch):
    if not _valid_mat(path,SHAPE[:3]):return False
    # Do not silently accept data whose available recorded hyperparameters contradict the tag.
    sidecar=path.with_suffix('.json')
    if sidecar.is_file():
        try:
            d=json.loads(sidecar.read_text(encoding='utf-8'))
            c=d['config']
            return (int(c['steps'])==steps and list(map(int,c['patch']))==[PATCH_T,patch,patch]
                    and int(c['hidden'])==HIDDEN and int(c['rank'])==K
                    and int(c['spatial_kernel'])==KERNEL and int(d['seed'])==SEED)
        except (OSError,ValueError,KeyError,TypeError):return False
    # Older MAT without sidecar: only accept if it carries consistent steps+rank markers.
    try:
        d=loadmat(path,variable_names=['rank','cnn_steps'])
        return int(d['rank'].ravel()[0])==K and int(d['cnn_steps'].ravel()[0])==steps
    except (OSError,KeyError,ValueError,TypeError):return False


def _valid_rgb(path,steps,patch):
    if not _valid_mat(path,SHAPE):return False
    # New v11 MATs are tagged with hyperparameters to prevent mismatched files.
    try:
        d=loadmat(path,variable_names=['rank','hidden','steps','patch_h','patch_w','seed'])
        return (int(d['rank'].ravel()[0])==K and int(d['hidden'].ravel()[0])==HIDDEN
                and int(d['steps'].ravel()[0])==steps and int(d['patch_h'].ravel()[0])==patch
                and int(d['patch_w'].ravel()[0])==patch and int(d['seed'].ravel()[0])==SEED)
    except (OSError,KeyError,ValueError,TypeError):return False


def _valid_baseline(path):
    # v10 MAT only stores reconstruction and runtime; validate its result shape.
    # v10 experimental protocol identifies this as K12_C32 / 1200 / P48.
    return _valid_mat(path,SHAPE)


def _event(out,**row):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')


def _rt(d):
    try:return float(np.asarray(d['runtime_seconds']).reshape(-1)[0])
    except (KeyError,ValueError,TypeError):return float('nan')


def _protocol(out,source):
    data=dict(protocol_version=11,source=str(source),size=list(SHAPE),seed=SEED,rank=K,hidden=HIDDEN,
              spatial_kernel=KERNEL,temporal_kernel=3,folds=FOLDS,
              masked_sweeps=SWEEPS,patch_t=PATCH_T,batch=BATCH,max_residual=MAX_RESIDUAL,
              configs={key:list(value) for key,value in ALL_CONFIGS.items()},
              same_mask_source='v8 prepared case and channels',
              test_ground_truth_access='evaluation process only')
    path=out/'STEPS_PATCH_PROTOCOL.json'
    if path.is_file() and json.loads(path.read_text(encoding='utf-8'))!=data:
        raise ValueError('Existing results directory uses another protocol. Supply a new --output.')
    out.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def run_one(source,out,video,sr,steps,patch,args):
    case=case_id(video,sr)
    if not legacy.casefile(source,case).is_file():
        raise FileNotFoundError('Prepared RGB case missing: '+str(legacy.casefile(source,case)))
    final=rgb_path(out,case,steps,patch)
    if not args.overwrite and _valid_rgb(final,steps,patch):
        print('SKIP RGB',label(steps,patch),case,flush=True)
        return True
    for ch in legacy.CHANNELS:
        inp=source_channel(source,case,ch)
        if not inp.is_file():raise FileNotFoundError('Prepared channel missing: '+str(inp))
        dst=channel_path(out,case,steps,patch,ch)
        if not args.overwrite and _valid_channel(dst,steps,patch):
            print('SKIP CHANNEL',label(steps,patch),case,ch,flush=True);continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        tmp=dst.with_name(dst.stem+'.partial.mat')
        cmd=[sys.executable,str(legacy.METHOD),'--case',str(inp),'--out',str(tmp),
             '--seed',str(SEED),'--rank',str(K),'--hidden',str(HIDDEN),
             '--spatial-kernel',str(KERNEL),'--masked-sweeps',str(SWEEPS),
             '--crossfit-folds',str(FOLDS),'--steps',str(steps),
             '--patch-t',str(PATCH_T),'--patch-h',str(patch),'--patch-w',str(patch),
             '--batch-size',str(BATCH),'--max-residual',str(MAX_RESIDUAL),
             '--device',args.device]
        logfile=out/'logs'/label(steps,patch)/(case+f'__{ch}.log')
        print('RUN',label(steps,patch),case,ch,'=>',logfile,flush=True)
        if args.dry_run:
            print('CMD',subprocess.list2cmdline(cmd),flush=True)
            continue
        logfile.parent.mkdir(parents=True,exist_ok=True)
        tmp.unlink(missing_ok=True)
        tmp.with_suffix('.json').unlink(missing_ok=True)
        start=time.monotonic()
        with logfile.open('w',encoding='utf-8') as f:
            f.write('CMD '+subprocess.list2cmdline(cmd)+'\n');f.flush()
            p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        seconds=time.monotonic()-start
        if p.returncode!=0 or not _valid_channel(tmp,steps,patch):
            # In --dry-run no file was created; this branch is real-run only.
            _event(out,case=case,config=label(steps,patch),channel=ch,status='FAILED',exit=p.returncode,
                   duration_seconds=round(seconds,3),log=str(logfile))
            print('FAILED',label(steps,patch),case,ch,logfile,flush=True)
            return False
        os.replace(tmp,dst)
        os.replace(tmp.with_suffix('.json'),dst.with_suffix('.json')) if tmp.with_suffix('.json').exists() else None
        _event(out,case=case,config=label(steps,patch),channel=ch,status='OK',duration_seconds=round(seconds,3))
        print('CHANNEL COMPLETE',label(steps,patch),case,ch,round(seconds,1),'s',flush=True)
    if args.dry_run:return True
    # GT is loaded only in the assembly/evaluation process; it does NOT enter solver subprocess.
    gt,obs,mask=legacy.load_truth(source,case)
    if gt.shape!=SHAPE or obs.shape!=SHAPE or mask.shape!=SHAPE:
        raise ValueError(f'Source dimensions wrong: {gt.shape} / {obs.shape} / {mask.shape}')
    recon=np.empty(SHAPE,dtype=np.float32)
    seconds=0.
    for ci,ch in enumerate(legacy.CHANNELS):
        d=loadmat(channel_path(out,case,steps,patch,ch),variable_names=['recon','runtime_seconds'])
        rec=np.asarray(d['recon'],dtype=np.float32)
        if rec.shape!=SHAPE[:3]:raise ValueError('Unexpected channel dimensions')
        recon[:,:,:,ci]=np.clip(rec,0,1)
        seconds+=_rt(d)
        del d,rec
    recon[mask]=obs[mask]
    final.parent.mkdir(parents=True,exist_ok=True)
    tmp=final.with_name(final.stem+'.partial.mat')
    savemat(tmp,{'recon':recon,'runtime_seconds':np.asarray([[seconds]]),
                 'rank':np.asarray([[K]]),'hidden':np.asarray([[HIDDEN]]),
                 'steps':np.asarray([[steps]]),'patch_h':np.asarray([[patch]]),
                 'patch_w':np.asarray([[patch]]),'seed':np.asarray([[SEED]])},do_compression=True)
    if not _valid_rgb(tmp,steps,patch):raise RuntimeError('Assembled RGB reconstruction is invalid')
    os.replace(tmp,final)
    _event(out,case=case,config=label(steps,patch),channel='RGB',status='OK',runtime_seconds=seconds)
    print('RGB COMPLETE',label(steps,patch),case,flush=True)
    del gt,obs,mask,recon
    gc.collect()
    return True


COLS=['video','sampling_rate','seed','rank','hidden','spatial_kernel','temporal_kernel','training_steps',
      'patch_t','patch_h','patch_w','configuration','source_version','split_role','runtime_seconds',
      'psnr_full','psnr_missing','ssim_full','mae_missing']
SUM_COLS=['sampling_rate','training_steps','patch_h','configuration','completed_videos',
          'psnr_missing_mean','ssim_full_mean','runtime_seconds_mean','psnr_missing_std_across_videos']
MISSING_COLS=['video','sampling_rate','configuration','path','status']


def _csvwrite(path,rows,cols):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def evaluate(source,out,v10,videos,rates,baseline=True):
    new_rows=[];missing=[]
    for video in videos:
        for sr in rates:
            case=case_id(video,sr)
            if not legacy.casefile(source,case).is_file():
                print('MISSING PREPARED',case)
                for cfg,(steps,patch) in ALL_CONFIGS.items():
                    if not baseline and cfg=='A0':continue
                    missing.append(dict(video=video,sampling_rate=sr,configuration=cfg,
                                        path=str(legacy.casefile(source,case)),status='SOURCE_MISSING'))
                continue
            gt=obs=mask=None
            for cfg,(steps,patch) in ALL_CONFIGS.items():
                if cfg=='A0' and not baseline:continue
                path=baseline_rgb(v10,case) if cfg=='A0' else rgb_path(out,case,steps,patch)
                valid=_valid_baseline(path) if cfg=='A0' else _valid_rgb(path,steps,patch)
                if not valid:
                    missing.append(dict(video=video,sampling_rate=sr,configuration=cfg,
                                        path=str(path),status='MISSING_OR_INVALID'))
                    continue
                if gt is None:
                    gt,obs,mask=legacy.load_truth(source,case)
                    if gt.shape!=SHAPE:raise ValueError('Source shape mismatch '+str(gt.shape))
                d=loadmat(path,variable_names=['recon','runtime_seconds'])
                rec=np.asarray(d['recon'],dtype=np.float32)
                m=legacy.metric(gt,obs,mask,rec)
                new_rows.append(dict(video=video,sampling_rate=sr,seed=SEED,rank=K,hidden=HIDDEN,
                     spatial_kernel=KERNEL,temporal_kernel=3,training_steps=steps,
                     patch_t=PATCH_T,patch_h=patch,patch_w=patch,
                     configuration=cfg,source_version='v10_read_only' if cfg=='A0' else 'v11',
                     split_role='development',runtime_seconds=_rt(d),**m))
                print(f'EVAL [{video} SR={sr:.1f}] {cfg} steps={steps} patch={patch} missingPSNR={m["psnr_missing"]:.4f} SSIM={m["ssim_full"]:.5f}',flush=True)
                del d,rec
            del gt,obs,mask
            gc.collect()
    oldrows=[]
    csv_path=out/'results_per_video.csv'
    if csv_path.is_file():
        with csv_path.open('r',encoding='utf-8-sig',newline='') as f:oldrows=list(csv.DictReader(f))
    def key(r):return (r['video'],round(float(r['sampling_rate']),6),int(r['seed']),r['configuration'])
    selected={(v,round(float(sr),6)) for v in videos for sr in rates}
    # Remove stale current selection before rebuilding from on-disk MAT validity.
    rows={key(r):r for r in oldrows if (r['video'],round(float(r['sampling_rate']),6)) not in selected}
    rows.update({key(r):r for r in new_rows})
    rows=sorted(rows.values(),key=key)
    _csvwrite(csv_path,rows,COLS)
    _csvwrite(out/'missing_configs.csv',missing,MISSING_COLS)
    summary=[]
    pairs=sorted({(round(float(r['sampling_rate']),6),r['configuration']) for r in rows})
    for sr,cfg in pairs:
        subset=[r for r in rows if round(float(r['sampling_rate']),6)==sr and r['configuration']==cfg]
        vals=np.asarray([float(r['psnr_missing']) for r in subset]); st,p=ALL_CONFIGS[cfg]
        summary.append(dict(sampling_rate=sr,training_steps=st,patch_h=p,configuration=cfg,
            completed_videos=len(subset),psnr_missing_mean=float(np.mean(vals)),
            ssim_full_mean=float(np.mean([float(r['ssim_full']) for r in subset])),
            runtime_seconds_mean=float(np.mean([float(r['runtime_seconds']) for r in subset])),
            psnr_missing_std_across_videos=float(np.std(vals,ddof=1)) if len(subset)>1 else ''))
    _csvwrite(out/'summary_steps_patch.csv',summary,SUM_COLS)
    print(f'EVAL DONE: {len(new_rows)} current valid; {len(rows)} retained; {len(missing)} missing configs',flush=True)
    return len(missing)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f=out/'results_per_video.csv'
    if not f.is_file():print('No CSV:',f);return
    with f.open('r',encoding='utf-8-sig',newline='') as fs:rows=list(csv.DictReader(fs))
    if not rows:print('CSV empty:',f);return
    figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    # Aggregate per video before across-video averaging, to prevent double weighting.
    for sr in sorted({float(r['sampling_rate']) for r in rows}):
        for mode in ('steps','patch'):
            selected=[]
            for cfg,(steps,patch) in ALL_CONFIGS.items():
                if mode=='steps' and patch!=48:continue
                if mode=='patch' and steps!=2400:continue
                cur=[r for r in rows if abs(float(r['sampling_rate'])-sr)<1e-7 and r['configuration']==cfg]
                if not cur:continue
                per_video={}
                for r in cur:per_video.setdefault(r['video'],[]).append(float(r['psnr_missing']))
                values=[float(np.mean(v)) for v in per_video.values()]
                selected.append((steps if mode=='steps' else patch,float(np.mean(values)),len(values)))
            if not selected:continue
            selected.sort()
            xs=[x[0] for x in selected];ys=[x[1] for x in selected]
            fig,ax=plt.subplots(figsize=(5.4,3.5))
            ax.plot(xs,ys,'o-',linewidth=1.8)
            ax.set_xticks(xs)
            ax.set_xlabel('CNN training steps  (patch=48)' if mode=='steps' else 'Spatial patch size  (steps=2400)')
            ax.set_ylabel('Missing-region PSNR (dB)')
            ax.grid(alpha=.25)
            for x,y,n in selected:
                ax.annotate(f'{y:.3f} (n={n})',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
            if len({x[2] for x in selected})>1:
                ax.set_title('PARTIAL: unequal video counts; compare cautiously',fontsize=9)
            fig.tight_layout()
            stem=figdir/f'{mode}_PSNR_SR{int(round(sr*100)):03d}'
            fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight')
            fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight')
            plt.close(fig)
            print('FIG',stem.with_suffix('.pdf'))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source-output',type=Path,default=DEFAULT_SOURCE,
                    help='v8 RGB results root containing cases/ and channel_cases/')
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--v10-results',type=Path,default=None,
                    help='v10 results root with rgb/K12_C32_S3; read-only baseline')
    ap.add_argument('--videos',default='blackswan')
    ap.add_argument('--rates',default='0.3')
    ap.add_argument('--configs',default='all',help='all or S2,S3,P64,P96 (new configs only)')
    ap.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--evaluate-only',action='store_true')
    ap.add_argument('--plot-only',action='store_true')
    ap.add_argument('--no-evaluate',action='store_true')
    ap.add_argument('--no-v10-baseline',action='store_true')
    ap.add_argument('--stop-on-error',action='store_true')
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args(argv)
    if args.evaluate_only and args.no_evaluate:ap.error('--evaluate-only conflicts with --no-evaluate')
    if args.plot_only:
        plot(args.output.expanduser().resolve());return 0
    try:
        vids=legacy.names(args.videos)
        rates=legacy.nums(args.rates,float,(0.2,0.3,0.4))
    except ValueError as exc:ap.error(str(exc))
    chosen=list(NEW_CONFIGS) if args.configs.lower()=='all' else [c.strip().upper() for c in args.configs.split(',') if c.strip()]
    if not chosen or len(set(chosen))!=len(chosen) or set(chosen)-set(NEW_CONFIGS):
        ap.error('Expected --configs all or comma separated S2,S3,P64,P96')
    source=args.source_output.expanduser().resolve()
    out=args.output.expanduser().resolve()
    v10=_candidate_v10(args.v10_results)
    print('Ours v11 CNN Steps x Patch | fixed: RGB 360x640x50, rank12 hidden32 kernel3 folds5 seed',SEED,flush=True)
    print('SOURCE:',source,'\nOUTPUT:',out,'\nV10 baseline (READ ONLY):',v10,flush=True)
    print('VIDEOS:',vids,'RATES:',rates,'NEW:',[(c,NEW_CONFIGS[c]) for c in chosen],flush=True)
    if not (source/'cases').is_dir() or not (source/'channel_cases').is_dir():
        ap.error('Missing prepared v8 RGB cases/channel_cases at '+str(source))
    if not args.dry_run:_protocol(out,source)
    if not args.no_v10_baseline:
        missing=[case_id(v,r) for v in vids for r in rates
                 if not _valid_baseline(baseline_rgb(v10,case_id(v,r)))]
        if missing:print('WARNING: v10 baseline missing/invalid for',missing,'(will not invent/retrain it).',flush=True)
    if args.dry_run:
        for v in vids:
            for r in rates:
                for cfg in chosen:
                    s,p=NEW_CONFIGS[cfg]
                    run_one(source,out,v,r,s,p,args)
        return 0
    failures=[]
    if not args.evaluate_only:
        for v in vids:
            for r in rates:
                for cfg in chosen:
                    s,p=NEW_CONFIGS[cfg]
                    try:
                        if not run_one(source,out,v,r,s,p,args):failures.append((v,r,cfg))
                    except Exception as exc:
                        print('ERROR',v,r,cfg,repr(exc),flush=True)
                        failures.append((v,r,cfg))
                    if failures and args.stop_on_error:
                        print('STOP ON ERROR:',failures[-1],flush=True)
                        return 2
    if not args.no_evaluate:
        evaluate(source,out,v10,vids,rates,baseline=not args.no_v10_baseline)
        plot(out)
    if failures:print('FAILED:',failures,flush=True)
    return 2 if failures else 0


if __name__=='__main__':sys.exit(main())
