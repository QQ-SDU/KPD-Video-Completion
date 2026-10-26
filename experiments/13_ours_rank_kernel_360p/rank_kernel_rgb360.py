#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAVIS 360p RGB rank × spatial-kernel diagnosis: ONLY the full CrossFit method.

Input is the UNCHANGED video/mask/channel case files of the nine-method 360p experiment.
Each model subprocess loads ONLY observed and mask, never gt; evaluation reads gt later.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from skimage.metrics import structural_similarity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHOD = ROOT/'benchmark_suite'/'Methods'/'Ours'/'run_same_mask.py'
SEED = 20261014
CHANNELS = ('R', 'G', 'B')
VIDEO_NAMES = ('blackswan','bmx-trees','breakdance','camel','car-roundabout','drift-chicane','kite-surf','parkour')
DEFAULT_SOURCE = Path(os.environ.get('DAVIS_360P_SOURCE',str(Path.home() / "kpd_data" / 'KPD' / 'results' / 'davis2017_rgb_all9_seed20261014_360p_f050')))
DEFAULT_OUTPUT = ROOT/'results'/'rank_kernel_rgb360p_seed20261014'

def names(s):
    a=list(VIDEO_NAMES) if s=='all' else [n.strip() for n in s.split(',') if n.strip()]
    if not a or len(set(a))!=len(a) or set(a)-set(VIDEO_NAMES):
        raise ValueError('Invalid --videos: '+s)
    return a

def nums(s, t, allowed):
    a=[t(x.strip()) for x in s.split(',') if x.strip()]
    if not a or len(set(a))!=len(a) or set(a)-set(allowed):
        raise ValueError(f'Expected distinct values among {sorted(allowed)}, got {s}')
    return a

def stem(video,sr):return f'{video}__sr{int(round(sr*100)):03d}__seed{SEED}'
def tag(rank,kernel):return f'K{rank:02d}_S{kernel}'
def cfile(source,case,channel):return source/'channel_cases'/(case+f'__{channel}.mat')
def cresult(out,case,rank,kernel,channel):return out/'channels'/tag(rank,kernel)/(case+f'__{channel}.mat')
def rgbfile(out,case,rank,kernel):return out/'rgb'/tag(rank,kernel)/(case+'.mat')
def casefile(source,case):return source/'cases'/(case+'.npz')
def is_valid(path, shape):
    if not path.is_file() or path.stat().st_size<100:return False
    try:
        rec=loadmat(path,variable_names=['recon'])['recon']
        return rec.shape==tuple(shape) and np.issubdtype(rec.dtype,np.number) and np.isfinite(rec).all()
    except (OSError,KeyError,ValueError,TypeError):return False

def load_truth(source,case):
    with np.load(casefile(source,case),allow_pickle=False) as d:
        return d['gt'].astype(np.float32),d['observed'].astype(np.float32),d['mask'].astype(bool)

def log_event(out,entry):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')

def read_rgb(path):
    return np.asarray(loadmat(path,variable_names=['recon'])['recon'],dtype=np.float32)

def run_one(source,out,video,sr,rank,kernel,a):
    case=stem(video,sr)
    if not casefile(source,case).exists():
        raise FileNotFoundError('Missing RGB prepared case '+str(casefile(source,case))+'\nPrepare it in the original v8 project first.')
    shape=(360,640,50,3)
    rgb=rgbfile(out,case,rank,kernel)
    if is_valid(rgb,shape) and not a.overwrite:
        print('SKIP RGB',case,tag(rank,kernel),flush=True);return True
    for ch in CHANNELS:
        inp=cfile(source,case,ch)
        if not inp.is_file():raise FileNotFoundError('Missing prepared channel case: '+str(inp))
        dst=cresult(out,case,rank,kernel,ch)
        dst.parent.mkdir(parents=True,exist_ok=True)
        # Header dimensions are available without reading any ground-truth values.
        expected=shape[:3]
        if is_valid(dst,expected) and not a.overwrite:
            print('SKIP CHANNEL',tag(rank,kernel),case,ch,flush=True);continue
        tmp=dst.with_name(dst.stem+'.partial.mat')
        cmd=[sys.executable,str(METHOD),'--case',str(inp),'--out',str(tmp),
             '--seed',str(SEED),'--rank',str(rank),'--spatial-kernel',str(kernel),
             '--device',a.device]
        log=out/'logs'/tag(rank,kernel)/(case+f'__{ch}.log')
        print('RUN',tag(rank,kernel),case,ch,flush=True)
        print('   Log:',log,flush=True)
        print('   CMD:',subprocess.list2cmdline(cmd),flush=True) if a.dry_run else None
        if a.dry_run:continue
        log.parent.mkdir(parents=True,exist_ok=True)
        if tmp.exists():tmp.unlink()
        start=time.monotonic()
        with log.open('w',encoding='utf-8') as f:
            f.write('CMD '+subprocess.list2cmdline(cmd)+'\n');f.flush()
            p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        elapsed=time.monotonic()-start
        ok=p.returncode==0 and is_valid(tmp,expected)
        log_event(out,dict(case=case,rank=rank,spatial_kernel=kernel,channel=ch,
                           status='OK' if ok else 'FAILED',seconds=round(elapsed,2),log=str(log)))
        if not ok:
            print('FAILED',tag(rank,kernel),case,ch,'See',log,flush=True)
            if a.stop_on_error:raise RuntimeError(str(log))
            return False
        os.replace(tmp,dst)
        side=tmp.with_suffix('.json')
        if side.exists():os.replace(side,dst.with_suffix('.json'))
        print('OK',tag(rank,kernel),case,ch,f'{elapsed:.1f}s',flush=True)
    if a.dry_run:return True
    # RGB case and ground-truth are loaded here ONLY after 3 method subprocesses finish.
    gt,obs,mask=load_truth(source,case)
    if gt.shape!=shape:raise ValueError(f'Unexpected RGB shape: {gt.shape} instead of {shape}')
    recon=np.empty(shape,dtype=np.float32)
    runtime=0.
    for ci,ch in enumerate(CHANNELS):
        d=loadmat(cresult(out,case,rank,kernel,ch),variable_names=['recon','runtime_seconds'])
        x=np.asarray(d['recon'],dtype=np.float32)
        if x.shape!=shape[:3]:raise ValueError(f'Unexpected {ch} shape: {x.shape}')
        recon[:,:,:,ci]=np.clip(x,0,1)
        val=d.get('runtime_seconds')
        runtime+=float(val[0,0]) if val is not None else float('nan')
    recon[mask]=obs[mask]
    rgb.parent.mkdir(parents=True,exist_ok=True)
    partial=rgb.with_name(rgb.stem+'.partial.mat')
    savemat(partial,{'recon':recon,'runtime_seconds':np.array([[runtime]])},do_compression=True)
    if not is_valid(partial,shape):raise RuntimeError('Invalid assembled RGB output')
    os.replace(partial,rgb)
    log_event(out,dict(case=case,rank=rank,spatial_kernel=kernel,channel='RGB',status='OK',seconds=runtime))
    print('RGB COMPLETE',tag(rank,kernel),case,flush=True)
    return True

def metric(gt,obs,mask,rec):
    rec=np.clip(rec,0,1).copy()
    rec[mask]=obs[mask]
    missing=~mask
    mfe=float(np.mean((gt[missing]-rec[missing])**2))
    fe=float(np.mean((gt-rec)**2))
    score=float(np.mean([structural_similarity(gt[:,:,i,:],rec[:,:,i,:],channel_axis=-1,data_range=1.0)
                         for i in range(gt.shape[2])]))
    return dict(psnr_full=-10*np.log10(max(fe,1e-14)),
                psnr_missing=-10*np.log10(max(mfe,1e-14)),ssim_full=score,
                mae_missing=float(np.mean(np.abs(gt[missing]-rec[missing]))))

def csvwrite(path,items):
    if not items:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)

def evaluate(source,out,videos,rates,ranks,kernels):
    rows=[];missing=[]
    for video in videos:
        for sr in rates:
            case=stem(video,sr)
            if not casefile(source,case).exists():continue
            gt=obs=mask=None
            for rank in ranks:
                for kernel in kernels:
                    f=rgbfile(out,case,rank,kernel)
                    if not is_valid(f,(360,640,50,3)):
                        missing.append(dict(case=case,rank=rank,spatial_kernel=kernel))
                        continue
                    if gt is None:gt,obs,mask=load_truth(source,case)
                    d=loadmat(f,variable_names=['recon','runtime_seconds'])
                    m=metric(gt,obs,mask,np.asarray(d['recon'],dtype=np.float32))
                    rows.append(dict(video=video,sampling_rate=sr,seed=SEED,rank=rank,
                                     spatial_kernel=kernel,
                                     temporal_kernel=3,
                                     runtime_seconds=float(d['runtime_seconds'].ravel()[0]),**m))
                    print(f'{video:16} SR={sr:.1f} K={rank:2} Kernel=(3,{kernel},{kernel}) Missing-PSNR={m["psnr_missing"]:.3f} SSIM={m["ssim_full"]:.4f}')
    # Merge with previously evaluated selections rather than overwriting them.
    # This permits "one video today, more videos tomorrow" incremental work.
    saved=out/'results_per_video.csv'
    previous=[]
    if saved.exists():
        with saved.open('r',encoding='utf-8-sig',newline='') as f:
            previous=list(csv.DictReader(f))
    def row_key(r):
        return (r['video'],round(float(r['sampling_rate']),5),int(r['rank']),int(r['spatial_kernel']))
    merged={row_key(r):r for r in previous}
    merged.update({row_key(r):r for r in rows})
    allrows=sorted(merged.values(),key=lambda r:row_key(r))
    csvwrite(saved,allrows)
    csvwrite(out/'missing_configs.csv',missing)
    aggregates=[]
    global_rates=sorted({float(r['sampling_rate']) for r in allrows})
    global_ranks=sorted({int(r['rank']) for r in allrows})
    global_kernels=sorted({int(r['spatial_kernel']) for r in allrows})
    for sr in global_rates:
        for rank in global_ranks:
            for kernel in global_kernels:
                data=[r for r in allrows if float(r['sampling_rate'])==sr and int(r['rank'])==rank and int(r['spatial_kernel'])==kernel]
                if not data:continue
                aggregates.append(dict(sampling_rate=sr,rank=rank,spatial_kernel=kernel,
                   completed_videos=len(data),
                   psnr_missing_mean=float(np.mean([float(r['psnr_missing']) for r in data])),
                   ssim_full_mean=float(np.mean([float(r['ssim_full']) for r in data])),
                   runtime_seconds_mean=float(np.mean([float(r['runtime_seconds']) for r in data]))))
    csvwrite(out/'summary_rank_kernel.csv',aggregates)
    print('EVALUATION',len(rows),'results;',len(missing),'missing; CSV:',out)
    return 0 if len(missing)==0 else 2

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-output',type=Path,default=DEFAULT_SOURCE,
       help='Prepared original 360p RGB results root (need cases/ and channel_cases/).')
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--videos',default='blackswan')
    p.add_argument('--rates',default='0.3',help='0.3 for first debug, 0.2,0.3,0.4 for full')
    p.add_argument('--ranks',default='4,8,12')
    p.add_argument('--kernels',default='3,5',help='SPATIAL kernel sizes; temporal kernel remains 3')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--no-evaluate',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--stop-on-error',action='store_true')
    a=p.parse_args(argv)
    if a.evaluate_only and a.no_evaluate:p.error('--evaluate-only and --no-evaluate conflict')
    try:
        videos=names(a.videos)
        rates=nums(a.rates,float,(0.2,0.3,0.4))
        ranks=nums(a.ranks,int,(4,8,12))
        kernels=nums(a.kernels,int,(3,5))
    except ValueError as exc:p.error(str(exc))
    source=a.source_output.expanduser().resolve(); out=a.output.expanduser().resolve()
    print('KPD + CROSSFIT ONLY | Rank x spatial kernel diagnostic',flush=True)
    print('SOURCE:',source,'\nOUTPUT:',out,flush=True)
    print('VIDEOS',videos,'RATES',rates,'RANKS',ranks,'SPATIAL KERNELS',kernels,flush=True)
    print('FIXED: seed',SEED,'steps 1200, folds 5, hidden 12, patch 16x48x48, residual bound 0.25',flush=True)
    print('TOTAL:',len(videos)*len(rates)*len(ranks)*len(kernels),'RGB configs,',
          len(videos)*len(rates)*len(ranks)*len(kernels)*3,'channel fits',flush=True)
    if not (source/'channel_cases').is_dir() or not (source/'cases').is_dir():
        p.error('Prepared source cases missing: '+str(source)+' (must contain cases/ and channel_cases/)')
    out.mkdir(parents=True,exist_ok=True)
    proto=out/'DIAGNOSTIC_PROTOCOL.json'
    expected={'seed':SEED,'source':str(source),'shape':[360,640,50,3],'network':'CrossFit',
              'training_steps':1200,'hidden_channels':12,'patch':[16,48,48],
              'factor_regularization_unchanged':True,'case_mask_source':'original360p',
              'kernel_policy':'temporal 3 fixed; spatial 3 or 5; 2nd layer dilation=2'}
    if proto.exists() and json.loads(proto.read_text(encoding='utf-8'))!=expected:
        p.error('Protocol mismatch; choose different --output to avoid mixed experiments.')
    proto.write_text(json.dumps(expected,indent=2,ensure_ascii=False),encoding='utf-8')
    errs=[]
    if not a.evaluate_only:
        for video in videos:
            for sr in rates:
                for rank in ranks:
                    for kernel in kernels:
                        try:
                            ok=run_one(source,out,video,sr,rank,kernel,a)
                            if not ok:errs.append((video,sr,rank,kernel))
                        except Exception as e:
                            print('ERROR',video,sr,rank,kernel,e,flush=True)
                            errs.append((video,sr,rank,kernel))
                            if a.stop_on_error:raise
    if a.dry_run:
        print('DRY RUN complete. No model executed.');return 0
    if not a.no_evaluate:
        evaluate(source,out,videos,rates,ranks,kernels)
    if errs:print('FAILED CONFIGS:',errs);return 2
    return 0

if __name__=='__main__':sys.exit(main())
