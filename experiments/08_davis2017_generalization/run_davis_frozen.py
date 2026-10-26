#!/usr/bin/env python3
"""Unseen DAVIS sequence evaluation with the FROZEN Table-I settings.

Methods use the identical observed tube masks. All model fitting and CNN training
receive only 'observed' and 'mask'; reference video is used for evaluation alone.
The last saved per-case record allows safe resume with --skip-existing.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'/'ours_v7'))
sys.path.insert(0,str(ROOT/'experiments'/'common_ablation'))
from kpd_video_core import smooth_fill, psnr, masked_psnr, mean_ssim
from detail_refiner_tube import exact_tube_mask, set_seed
from masked_kpd import run_masked_clsa
from crossfit_tube_refiner import build_crossfit_folds, train_crossfit_tube_cnn
from detail_refiner_v41 import DetailResidualCNN, infer_detail
from ablation_refiner import train_direct_tube_ablation

METHODS=('KPD','DIRECT','CROSSFIT')
FROZEN=dict(rank=4,masked_sweeps=12,lam_space=0.005,lam_time=0.05,ridge=1e-5,
            folds=5,hidden=12,steps=1200,patch_t=16,patch_h=48,patch_w=48,
            batch_size=2,lr=1e-3,max_residual=0.25,smooth_l1_beta=0.02,
            detail_weight=2.0,magnitude_scale=0.05,magnitude_cap=3.0,residual_penalty=1e-4)

def read_manifest(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def case_seed(base, name):
    # Derived deterministically, so masks differ among clips but match across methods.
    key=int(hashlib.sha256(name.encode('utf-8')).hexdigest()[:8],16)
    return (int(base)+key)%(2**32)

def metrics(gt,rec,mask):
    miss=~mask
    diff=np.diff(gt,axis=2)-np.diff(rec,axis=2)
    miss_t=miss[:,:,:-1]&miss[:,:,1:]
    return dict(psnr_full=psnr(gt,rec),ssim_full=mean_ssim(gt,rec),
                psnr_missing=masked_psnr(gt,rec,miss),
                mae_missing=float(np.mean(np.abs((gt-rec)[miss]))),
                temporal_gradient_mae=float(np.mean(np.abs(diff[miss_t]))))

def fitting(observed,mask,seed,cfg):
    ini=smooth_fill(observed,mask)
    return run_masked_clsa(observed=observed,mask=mask,initial_tensor=ini,
        rank=cfg['rank'],outer_sweeps=cfg['masked_sweeps'],lam_space=cfg['lam_space'],
        lam_time=cfg['lam_time'],ridge=cfg['ridge'],seed=seed,init='warm',
        project_observed=True,verbose=False,evaluation_gt=None,
        balance_scales=True,check_monotonic=True)

def project(raw,observed,mask):
    rec=np.clip(raw,0,1).copy();rec[mask]=observed[mask];return rec

def fit_residual(method,observed,mask,backbone,seed,device,cfg):
    set_seed(seed+1000)
    model=DetailResidualCNN(hidden=cfg['hidden'],max_residual=cfg['max_residual']).to(device)
    t0=time.perf_counter()
    common=dict(steps=cfg['steps'],patch_t=cfg['patch_t'],patch_h=cfg['patch_h'],
                patch_w=cfg['patch_w'],batch_size=cfg['batch_size'],lr=cfg['lr'],
                detail_weight=cfg['detail_weight'],amp=(device.type=='cuda'),
                print_every=max(1,cfg['steps']//3),seed=seed+2000,
                smooth_l1_beta=cfg['smooth_l1_beta'],magnitude_scale=cfg['magnitude_scale'],
                magnitude_cap=cfg['magnitude_cap'],residual_penalty=cfg['residual_penalty'])
    if method=='DIRECT':
        train_direct_tube_ablation(model,backbone,observed,mask,device,
            input_mode='FULL',**common)
        auxiliary_seconds=0.0
    elif method=='CROSSFIT':
        states=build_crossfit_folds(observed,mask,rank=cfg['rank'],
            masked_sweeps=cfg['masked_sweeps'],lam_space=cfg['lam_space'],
            lam_time=cfg['lam_time'],ridge=cfg['ridge'],n_folds=cfg['folds'],seed=seed+500)
        auxiliary_seconds=time.perf_counter()-t0
        train_crossfit_tube_cnn(model,states,device,**common)
    else:
        raise ValueError(method)
    train_total=time.perf_counter()-t0
    pred=infer_detail(model,backbone,observed,mask,device)
    if device.type=='cuda':torch.cuda.synchronize()
    total=time.perf_counter()-t0
    del model
    return project(backbone+pred,observed,mask),total,auxiliary_seconds,train_total-auxiliary_seconds

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=None,help='Directory with .npy and manifest.csv')
    p.add_argument('--out',default=None)
    p.add_argument('--videos',default=None,help='Comma-separated subset; default all manifest rows')
    p.add_argument('--rates',default='0.2,0.4')
    p.add_argument('--seeds',default='20261014,20261015,20261016')
    p.add_argument('--methods',default='KPD,DIRECT,CROSSFIT')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--skip-existing',action='store_true',default=True)
    p.add_argument('--smoke-test',action='store_true',help='Only a code test, NOT for paper results')
    a=p.parse_args()
    data=Path(a.data).resolve() if a.data else ROOT/'benchmark_suite'/'data'/'davis2017_new'
    out=Path(a.out).resolve() if a.out else ROOT/'results'/'davis2017_new'
    out.mkdir(parents=True,exist_ok=True)
    all_rows=read_manifest(data/'manifest.csv')
    req_names=set(x.strip() for x in a.videos.split(',')) if a.videos else None
    videos=[r for r in all_rows if req_names is None or r['name'] in req_names]
    if req_names and set(r['name'] for r in videos)!=req_names: raise ValueError('Requested videos not all present')
    if not videos:raise ValueError('No input videos')
    rates=[float(s) for s in a.rates.split(',')]
    seeds=[int(s) for s in a.seeds.split(',')]
    methods=[s.strip().upper() for s in a.methods.split(',') if s.strip()]
    if any(x not in METHODS for x in methods):raise ValueError('Allowed methods: '+','.join(METHODS))
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('No CUDA device')
    device=torch.device('cuda' if torch.cuda.is_available() and a.device=='auto' else ('cpu' if a.device=='auto' else a.device))
    cfg=FROZEN.copy()
    if a.smoke_test:
        cfg.update(rank=2,masked_sweeps=1,folds=2,steps=2,patch_t=6,patch_h=16,patch_w=16,batch_size=1)
    run_cfg=dict(methods=methods,rates=rates,seeds=seeds,video_names=[r['name'] for r in videos],frozen=cfg,smoke_test=a.smoke_test,
                 regime='all observed tubes used for both Direct and CrossFit training',
                 selection='no test-ground-truth hyperparameter tuning',device=str(device))
    (out/'RUN_CONFIG.json').write_text(json.dumps(run_cfg,indent=2),encoding='utf-8')
    casefolder=out/'per_case';casefolder.mkdir(exist_ok=True)
    for row in videos:
        gt=np.asarray(np.load(data/row['file'],allow_pickle=False),dtype=np.float64)
        if gt.ndim!=3:raise ValueError('Expected HWT')
        if gt.min()<-1e-8 or gt.max()>1+1e-8:raise ValueError('Intensity outside [0,1]')
        for sr in rates:
            for requested_seed in seeds:
                seed=case_seed(requested_seed,row['name'])
                case_id=f"{row['name']}__sr{int(round(sr*100)):03d}__seed{requested_seed}"
                mask=exact_tube_mask(gt.shape,sr,seed)
                observed=np.zeros_like(gt)
                observed[mask]=gt[mask]
                pending=[method for method in methods if not(a.skip_existing and (casefolder/f'{case_id}__{method}.json').exists())]
                if not pending:
                    print('SKIP complete case:',case_id,flush=True);continue
                print(f'\n=== {case_id} shape={gt.shape} running {pending} device={device} ===',flush=True)
                set_seed(seed)
                if device.type=='cuda':torch.cuda.synchronize()
                t0=time.perf_counter()
                kp=fitting(observed,mask,seed,cfg)
                if device.type=='cuda':torch.cuda.synchronize()
                kpd_time=time.perf_counter()-t0
                base=kp.backbone
                for method in pending:
                    if method=='KPD':
                        rec=project(base,observed,mask)
                        training=0.; extra=0.; residual_total=0.
                    else:
                        rec,residual_total,extra,training=fit_residual(method,observed,mask,base,seed,device,cfg)
                    vals=metrics(gt,rec,mask)
                    entry=dict(name=row['name'],method=method,rate_requested=sr,
                        realized_sr=float(mask[:,:,0].mean()),seed_requested=requested_seed,
                        seed_actual=seed,shape_hwt=list(gt.shape),
                        duration_total_seconds=float(kpd_time+residual_total),
                        kpd_seconds=float(kpd_time),auxiliary_fit_seconds=float(extra),
                        cnn_train_seconds=float(training),
                        kpd_objective_monotone=all(bool(h['monotone_ok']) for h in kp.history),
                        **vals)
                    target=casefolder/f'{case_id}__{method}.json'
                    target.write_text(json.dumps(entry,indent=2),encoding='utf-8')
                    print(f"[{method}] PSNR full={vals['psnr_full']:.3f}; PSNR missing={vals['psnr_missing']:.3f}; SSIM={vals['ssim_full']:.4f}; wall={entry['duration_total_seconds']:.1f}s",flush=True)
    print('\nDone. Next: python experiments/08_davis2017_generalization/summarize_davis.py --input',out)

if __name__=='__main__':main()
