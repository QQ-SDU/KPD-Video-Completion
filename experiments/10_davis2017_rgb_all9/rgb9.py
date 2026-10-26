#!/usr/bin/env python3
"""DAVIS RGB all-nine benchmark: single shared tube mask, each 3D method per RGB channel.

The core methods and hyperparameters are unchanged from the grayscale benchmark.
Each method is trained/fitted independently on R,G,B (no grayscale-to-color trick).
No model process is passed a 4D GT; only each observed channel and its mask.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess, sys, shutil, time
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.io import loadmat, savemat
from skimage.metrics import structural_similarity
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BENCH=ROOT/'benchmark_suite'
sys.path.insert(0,str(ROOT/'experiments'/'09_davis2017_all9'))
import davis9 as base
from davis9 import ALL, RATES, DEFAULT_SEED, sha, read_csv, write_csv, selection, stamp, parse_values, fp
from benchmark.io_utils import make_tube_mask, linear_spatial_init, save_case
from benchmark.evaluate_output import load_recon,align_shape
from run_python_suite import build_cmd

CHANNELS=('R','G','B')

def _output(args):return Path(args.output).expanduser().resolve()
def _data(args):return Path(args.data).expanduser().resolve()
def _report(args):return base.report_path(_output(args),args)
def _index(output):return output/'cases_index.csv'

def load_video(record,data):
    file=data/record['file'];arr=np.load(file,allow_pickle=False)
    if arr.ndim!=4 or arr.shape[-1]!=3 or not np.issubdtype(arr.dtype,np.number):
        raise ValueError(f'Expected H,W,T,3 RGB data, got {arr.shape}: {file}')
    if not np.isfinite(arr).all() or arr.min()<0 or arr.max()>1:raise ValueError('Invalid RGB range')
    if record.get('sha256') and sha(file)!=record['sha256']:raise ValueError('RGB source SHA mismatch '+str(file))
    return np.asarray(arr,dtype=np.float32)

def paths(output,stem):return output/'cases'/(stem+'.npz')
def channel_case(out,stem,c):return out/'channel_cases'/(stem+f'__{c}.mat')
def channel_out(out,method,stem,c):return out/'channel_reconstructions'/method/(method+'__'+stem+f'__{c}.mat')
def rgb_out(out,method,stem):return out/'reconstructions'/method/(method+'__'+stem+'.mat')

def load_case_rgb(out,stem):
    with np.load(paths(out,stem),allow_pickle=False) as d:
        return d['gt'],d['observed'],d['mask'].astype(bool)

def check_rgb_output(file,shape):
    if not file.exists() or file.stat().st_size<200:return False,'missing/small'
    try:
        d=loadmat(file,variable_names=['recon']); x=d['recon']
        if tuple(x.shape)!=tuple(shape):return False,f'wrong RGB shape {x.shape}, expected {shape}'
        if not np.issubdtype(x.dtype,np.number) or not np.all(np.isfinite(x)):return False,'nonfinite/nonnumeric'
        return True,'ok'
    except Exception as exc:return False,str(exc)

def do_prepare(args):
    out=_output(args);data=_data(args);out.mkdir(parents=True,exist_ok=True)
    rows=selection(read_csv(data/'manifest.csv'),args)
    rates=parse_values(args.rates,float)
    if sorted(rates)!=RATES:raise ValueError('RGB protocol requires exactly SR=0.2,0.3,0.4')
    paramhash=sha(BENCH/'config'/'method_parameters.json')
    proto=out/'PROTOCOL.json'
    if proto.exists():
        former=json.loads(proto.read_text(encoding='utf-8'))
        if former.get('color')!='RGB' or former.get('seed')!=args.seed or former.get('frozen_params_sha256')!=paramhash:
            raise RuntimeError('Incompatible result directory. RGB and grayscale runs must never share outputs.')
    existing={x['case']:x for x in read_csv(_index(out))} if _index(out).exists() else {}
    new={}
    for record in rows:
        name=record['name'];video=load_video(record,data)
        h,w,t,c=video.shape
        for sr in rates:
            stem=f'{name}__sr{fp(sr)}__seed{args.seed}'
            mask3d=make_tube_mask(h,w,t,sr,args.seed)
            if int(mask3d[:,:,0].sum())!=round(h*w*sr):raise AssertionError('Mask count inconsistent')
            mask=np.broadcast_to(mask3d[:,:,:,None],video.shape).copy()
            entry=dict(case=stem,name=name,sampling_rate=sr,replicate_seed=args.seed,model_seed=args.seed,
                       observed_tubes=int(mask3d[:,:,0].sum()),total_tubes=h*w,realized_sr=float(mask3d[:,:,0].mean()),
                       height=h,width=w,frames=t,channels=3,color='RGB',layout='HWTC',
                       input_file=record['file'],input_sha256=sha(data/record['file']))
            if stem in existing:
                old=existing[stem]
                if any(str(old.get(k))!=str(entry.get(k)) for k in ['name','sampling_rate','model_seed','height','width','frames','input_sha256','color']):
                    raise RuntimeError('Existing RGB case differs: '+stem)
            new[stem]=entry
            npz=paths(out,stem)
            channels=[channel_case(out,stem,k) for k in CHANNELS]
            if npz.exists() and all(p.exists() for p in channels) and not args.overwrite:
                with np.load(npz,allow_pickle=False) as saved:
                    if not np.array_equal(saved['gt'],video) or not np.array_equal(saved['mask'],mask):
                        raise RuntimeError('Stale RGB case '+stem)
                print('SKIP existing RGB',stem,flush=True)
                continue
            obs=np.where(mask,video,0).astype(np.float32)
            npz.parent.mkdir(parents=True,exist_ok=True)
            tmp=npz.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,gt=video,observed=obs,mask=mask.astype(np.uint8))
            os.replace(tmp,npz)
            # Channel cases are identical to original grayscale 3D MAT interface, preserving methods.
            for idx,k in enumerate(CHANNELS):
                cgt=video[:,:,:,idx]
                cobs=obs[:,:,:,idx]
                init=linear_spatial_init(cobs,mask3d)
                cp=channel_case(out,stem,k)
                save_case(cp.with_suffix(''),cgt,cobs,mask3d,init,
                          dict(entry,channel=k,channel_index=idx,gt_only_for_evaluation=True))
            print('READY RGB',stem,video.shape,f'SR={mask3d[:,:,0].mean():.6f}',flush=True)
    idx=_index(out);merged={**existing,**new}
    ordered=sorted(merged.values(),key=lambda x:(x['name'],float(x['sampling_rate'])))
    idx.parent.mkdir(parents=True,exist_ok=True)
    temp=idx.with_suffix('.tmp.csv');write_csv(temp,ordered);os.replace(temp,idx)
    proto.write_text(json.dumps(dict(dataset='DAVIS 2017 RGB',color='RGB',array_layout='HWTC',
       original_methods_unchanged=True,channel_policy='Independent R G B estimation; common spatial tube mask',
       seed=args.seed,rates=RATES,methods=ALL,prepared_videos=sorted(set(r['name'] for r in ordered)),
       frozen_params_sha256=paramhash,preprocessing_sha256=sha(data/'PREPROCESSING.json')),
       indent=2,ensure_ascii=False),encoding='utf-8')
    print('RGB CASES prepared:',len(new),'in selection; cumulative',len(ordered),'cases /',len(ordered)*9,'method-level runs',flush=True)

def check_protocol(args):
    proto=_output(args)/'PROTOCOL.json'
    if not proto.exists():raise RuntimeError('Run prepare first')
    data=json.loads(proto.read_text(encoding='utf-8'))
    if data.get('color')!='RGB':raise RuntimeError('Refusing to use grayscale output for RGB')
    if data.get('frozen_params_sha256')!=sha(BENCH/'config'/'method_parameters.json'):
        raise RuntimeError('Frozen hyperparameters changed after preparation')

def do_run(args):
    check_protocol(args)
    out=_output(args)
    entries=selection(read_csv(_index(out)),args)
    requested_rates=set(parse_values(args.rates,float))
    if not requested_rates or requested_rates-set(RATES):
        raise ValueError('Run rates must be within 0.2,0.3,0.4')
    entries=[e for e in entries if round(float(e['sampling_rate']),6) in requested_rates]
    methods=ALL if args.methods=='all' else parse_values(args.methods,str)
    if not methods or any(m not in ALL for m in methods):raise ValueError('Unknown method selection')
    if any(m in base.MAT_METHODS for m in methods) and not args.dry_run:
        if shutil.which(os.environ.get('MATLAB_BIN','matlab')) is None:
            raise RuntimeError('MATLAB not found; set MATLAB_BIN on Windows or PATH')
    problems=[]
    for entry in entries:
        stem=entry['case'];shape=(int(entry['height']),int(entry['width']),int(entry['frames']),3)
        gt,obs,mask=load_case_rgb(out,stem)
        if tuple(gt.shape)!=shape:raise ValueError('RGB case shape mismatch '+stem)
        for method in methods:
            complete=rgb_out(out,method,stem)
            ok,_=check_rgb_output(complete,shape)
            if ok and not args.overwrite:
                print('SKIP RGB',method,stem,flush=True);continue
            parts=[];times=[]
            for k in CHANNELS:
                cc=channel_case(out,stem,k); rec=channel_out(out,method,stem,k)
                rec.parent.mkdir(parents=True,exist_ok=True)
                val,reason=base._valid_out(cc,rec)
                if not val or args.overwrite:
                    incomplete=rec.with_name(rec.stem+'.partial.mat')
                    cmd=build_cmd(method,cc,incomplete) if method in base.PY_METHODS else base._mat_command(method,cc,incomplete,args.seed)
                    if method in base.PY_METHODS:
                        # The legacy runner infers --seed from a filename ending '__seedNNN'.
                        # RGB channel suffixes ('__R', '__G', '__B') prevent that match.
                        # Explicitly override the inferred seed so ALL methods/channels
                        # use the protocol's single fixed seed (not the legacy default 1).
                        if '--seed' not in cmd:
                            raise RuntimeError('Missing --seed in frozen Python runner command')
                        cmd[cmd.index('--seed')+1]=str(args.seed)
                    log=out/'logs'/method/(stem+f'__{k}.log');log.parent.mkdir(parents=True,exist_ok=True)
                    print('RUN',method,stem,k,'->',log,flush=True)
                    if args.dry_run:
                        print('DRY RUN',cmd,flush=True)
                        continue
                    if incomplete.exists():incomplete.unlink()
                    start=time.perf_counter()
                    with log.open('w',encoding='utf-8',errors='replace') as f:
                        f.write('COMMAND: '+repr(cmd)+'\n');f.flush()
                        try:pr=subprocess.run(cmd,cwd=BENCH,stdout=f,stderr=subprocess.STDOUT)
                        except Exception as exc:
                            pr=None;f.write('\nLAUNCH ERROR: '+repr(exc)+'\n')
                    elapsed=time.perf_counter()-start
                    good,desc=base._valid_out(cc,incomplete)
                    if pr is None or pr.returncode!=0 or not good:
                        print('FAILED',method,stem,k,'rc',getattr(pr,'returncode',None),desc,'log',log,flush=True)
                        row=dict(timestamp=stamp(),method=method,case=stem,channel=k,status='FAILED',wall_seconds=round(elapsed,2),log=str(log),reason=desc)
                        _journal(out,row);problems.append(row)
                        if args.stop_on_error:raise RuntimeError('See '+str(log))
                        break
                    os.replace(incomplete,rec)
                    print('OK',method,stem,k,f'{elapsed:.1f}s',flush=True)
                    _journal(out,dict(timestamp=stamp(),method=method,case=stem,channel=k,status='OK',wall_seconds=round(elapsed,2),log=str(log)))
                if args.dry_run:continue
                recon,runtime=load_recon(rec)
                recon=align_shape(recon,tuple(gt.shape[:3]))
                recon=np.clip(recon,0.,1.).astype(np.float32)
                ci=CHANNELS.index(k)
                recon[mask[:,:,:,ci]]=obs[:,:,:,ci][mask[:,:,:,ci]]
                parts.append(recon)
                times.append(float(runtime) if runtime is not None else float('nan'))
            if args.dry_run:continue
            if len(parts)!=3:continue
            merged=np.stack(parts,axis=-1)
            if not np.array_equal(merged[mask],obs[mask]):raise AssertionError('RGB DC failed')
            complete.parent.mkdir(parents=True,exist_ok=True)
            if complete.exists() and args.overwrite:complete.unlink()
            partial=complete.with_name(complete.stem+'.partial.mat')
            # runtime_seconds is the sum of algorithm-reported runtimes, one per RGB channel.
            # If a runner lacks the field, report NaN rather than an invented wall-clock time.
            runtime=float(np.sum(times)) if all(np.isfinite(times)) else float('nan')
            savemat(partial,{'recon':merged.astype(np.float32),'runtime_seconds':runtime},do_compression=True)
            valid,why=check_rgb_output(partial,shape)
            if not valid:raise RuntimeError('Bad merged RGB output: '+why)
            os.replace(partial,complete)
            _journal(out,dict(timestamp=stamp(),method=method,case=stem,channel='RGB',status='OK',runtime_seconds=runtime))
            print('RGB COMPLETE',method,stem,flush=True)
    if problems:
        print('RGB RUN ERRORS:',len(problems));sys.exit(2)
    if args.dry_run:
        print('DRY RUN ONLY: No methods were executed');return
    print('ALL REQUESTED RGB RUNS COMPLETE',len(entries)*len(methods))

def _journal(out,row):
    # Parallel MATLAB workers use their own append-only journal to avoid
    # concurrent writers to a shared Windows text file.
    journal=os.environ.get('KPD_PARALLEL_JOURNAL')
    target=out/'task_status.jsonl' if not journal else out/'parallel_journals'/(journal+'.jsonl')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')

def metric_for(gt,obs,mask,rec):
    rec=np.clip(rec,0,1).astype(np.float32)
    before=float(np.mean(np.abs(rec[mask]-obs[mask])))
    rec=rec.copy();rec[mask]=obs[mask]
    miss=~mask
    full=float(np.mean((gt-rec)**2));missing=float(np.mean((gt[miss]-rec[miss])**2))
    ssim=float(np.mean([structural_similarity(gt[:,:,t,:],rec[:,:,t,:],channel_axis=-1,data_range=1.)
                        for t in range(gt.shape[2])]))
    grad=np.abs(np.diff(gt,axis=2)-np.diff(rec,axis=2))
    tm=miss[:,:,:-1,:]
    return dict(psnr_full=float(10*np.log10(1/max(full,1e-14))),
         psnr_missing=float(10*np.log10(1/max(missing,1e-14))),
         ssim_full=ssim,mae_missing=float(np.mean(np.abs(gt[miss]-rec[miss]))),
         temporal_gradient_mae_missing=float(np.mean(grad[tm])),before_dc_mae=before)

def do_evaluate(args):
    check_protocol(args);out=_output(args);rows=selection(read_csv(_index(out)),args)
    report=_report(args);report.mkdir(parents=True,exist_ok=True)
    measured=[];absent=[]
    for e in rows:
        stem=e['case'];gt,obs,mask=load_case_rgb(out,stem)
        for method in ALL:
            recpath=rgb_out(out,method,stem)
            ok,reason=check_rgb_output(recpath,gt.shape)
            if not ok:absent.append(dict(method=method,case=stem,reason=reason));continue
            try:
                d=loadmat(recpath,variable_names=['recon','runtime_seconds']); rec=d['recon']
                rt=float(d['runtime_seconds'].squeeze()) if 'runtime_seconds' in d else float('nan')
                vals=metric_for(gt,obs,mask,rec)
                measured.append(dict(method=method,case=stem,name=e['name'],sampling_rate=float(e['sampling_rate']),
                    replicate_seed=e['replicate_seed'],model_seed=e['model_seed'],method_runtime_seconds=rt,**vals))
            except Exception as exc:absent.append(dict(method=method,case=stem,reason=str(exc)))
    write_csv(report/'per_run_metrics.csv',measured)
    write_csv(report/'missing_or_failed.csv',absent,['method','case','reason'])
    if absent:
        (report/'SUMMARY_COMPLETE.csv').unlink(missing_ok=True)
        print('PARTIAL only, missing',len(absent),'runs; see missing_or_failed.csv',flush=True)
        if not args.allow_partial:raise SystemExit(3)
    aggregates=[]
    for sr in RATES:
        for method in ALL:
            data=[r for r in measured if r['method']==method and r['sampling_rate']==sr]
            if not data:continue
            item=dict(sampling_rate=sr,method=method,completed=len(data),expected=sum(float(x['sampling_rate'])==sr for x in rows))
            for key in ['psnr_full','psnr_missing','ssim_full','mae_missing','temporal_gradient_mae_missing','method_runtime_seconds']:
                vec=np.array([float(x[key]) for x in data],dtype=float)
                item[key+'_mean']=float(np.nanmean(vec)) if np.isfinite(vec).any() else float('nan')
                item[key+'_std']=float(np.nanstd(vec,ddof=1)) if np.isfinite(vec).sum()>1 else float('nan')
            aggregates.append(item)
    write_csv(report/('SUMMARY_PARTIAL.csv' if absent else 'SUMMARY_COMPLETE.csv'),aggregates)
    if not absent:(report/'SUMMARY_PARTIAL.csv').unlink(missing_ok=True)
    print('RGB EVALUATION',len(measured),'/',len(rows)*9,'--',('INCOMPLETE' if absent else 'COMPLETE'))

def _roi(gt,mask,frame,rh=42,rw=52):
    from scipy.ndimage import uniform_filter
    h,w=gt.shape[:2];rh=min(rh,h);rw=min(rw,w)
    img=np.mean(gt[:,:,frame,:],axis=-1)
    gy,gx=np.gradient(img)
    energy=uniform_filter((np.abs(gy)+np.abs(gx))*(~mask[:,:,frame,0]),size=(rh,rw))
    py,px=rh//2,rw//2
    energy[:py]=0;energy[h-(rh-py):]=0;energy[:,:px]=0;energy[:,w-(rw-px):]=0
    yy,xx=np.unravel_index(np.argmax(energy),energy.shape)
    return max(0,min(h-rh,int(yy-py))),max(0,min(w-rw,int(xx-px))),rh,rw

def do_visualize(args):
    out=_output(args);report=_report(args)
    if not (report/'SUMMARY_COMPLETE.csv').exists():raise RuntimeError('First complete all nine methods and run evaluate')
    entries=selection(read_csv(_index(out)),args)
    metrics=read_csv(report/'per_run_metrics.csv')
    indexed={}
    for r in metrics:indexed.setdefault(r['case'],{})[r['method']]=r
    candidates=[]
    for stem,d in indexed.items():
        if set(d)!=set(ALL):continue
        ours=float(d['Ours']['psnr_missing'])
        strongest=max(float(d[m]['psnr_missing']) for m in ALL if m!='Ours')
        candidates.append(dict(case=stem,name=d['Ours']['name'],sampling_rate=d['Ours']['sampling_rate'],
             ours_psnr_missing=ours,best_baseline_psnr_missing=strongest,delta_vs_best_baseline=ours-strongest))
    candidates.sort(key=lambda x:x['delta_vs_best_baseline'],reverse=True)
    write_csv(report/'QUALITATIVE_ALL_CANDIDATES.csv',candidates)
    if args.cases:
        stems=parse_values(args.cases,str)
    else:
        stems=[];used=set()
        for row in candidates:
            if row['name'] not in used and row['delta_vs_best_baseline']>0 and len(stems)<2:
                stems.append(row['case']);used.add(row['name'])
        negative=next((row['case'] for row in candidates[::-1] if row['name'] not in used),None)
        if negative:stems.append(negative)
    methods=parse_values(args.display_methods,str)
    if any(m not in ALL for m in methods):raise ValueError('Unknown visual method')
    names=['Observed']+methods+['GT']
    vis=report/'qualitative';vis.mkdir(parents=True,exist_ok=True)
    for stem in stems:
        gt,obs,mask=load_case_rgb(out,stem)
        t=(gt.shape[2]//2 if args.frame<0 else args.frame)
        if not 0<=t<gt.shape[2]:raise ValueError('Frame out of bounds')
        top,left,rh,rw=_roi(gt,mask,t)
        imgs={'Observed':obs,'GT':gt}
        for method in methods:
            d=loadmat(rgb_out(out,method,stem),variable_names=['recon']);x=np.clip(d['recon'],0,1).astype(np.float32)
            x[mask]=obs[mask];imgs[method]=x
        H,W=gt.shape[:2];gap=12;header=25;roi_width=W//2;roi_h=int(rh*roi_width/rw)
        canvas=Image.new('RGB',(gap+len(names)*(W+gap),header+H+12+roi_h+40),'white')
        pen=ImageDraw.Draw(canvas)
        for j,method in enumerate(names):
            x=gap+j*(W+gap)
            pen.text((x+4,5),method,fill=(10,10,10))
            img=Image.fromarray(np.round(imgs[method][:,:,t,:]*255).clip(0,255).astype(np.uint8),'RGB')
            canvas.paste(img,(x,header))
            pen.rectangle((x+left,header+top,x+left+rw,header+top+rh),outline=(225,35,35),width=2)
            crop=img.crop((left,top,left+rw,top+rh)).resize((roi_width,roi_h),Image.Resampling.BICUBIC)
            canvas.paste(crop,(x,header+H+12))
            if method not in ('Observed','GT'):
                psnr=float(indexed[stem][method]['psnr_missing'])
                pen.text((x+3,header+H+roi_h+16),f'M-PSNR {psnr:.2f}',fill=(25,25,25))
        if args.scale>1:canvas=canvas.resize((canvas.width*args.scale,canvas.height*args.scale),Image.Resampling.LANCZOS)
        name=vis/(stem+'.png');canvas.save(name)
        print('RGB VISUAL',name)
    (vis/'SELECTION_POLICY.txt').write_text('All nine methods evaluated on the same RGB videos and tube masks. '\
       'Illustrations selected with ranked missing-region PSNR, with a failure example. '\
       'ROI computed from GT geometry once per case and shared across methods.\n',encoding='utf-8')

def do_inspect(args):
    out=_output(args);rows=selection(read_csv(_index(out)),args);n=0
    for e in rows:
        shape=(int(e['height']),int(e['width']),int(e['frames']),3)
        for method in ALL:
            path=rgb_out(out,method,e['case'])
            ok,_=check_rgb_output(path,shape)
            if not ok:n+=1;print('MISSING/FAILED RGB',method,e['case'])
    print('RGB progress:',len(rows)*9-n,'/',len(rows)*9,'complete')
    if n:raise SystemExit(2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','evaluate','visualize','inspect'])
    p.add_argument('--data',default=str(BENCH/'data'/'davis2017_rgb_new'))
    p.add_argument('--output',default=str(ROOT/'results'/'davis2017_rgb_all9_single_seed'))
    p.add_argument('--videos',default=None)
    p.add_argument('--rates',default='0.2,0.3,0.4')
    p.add_argument('--seed',type=int,default=DEFAULT_SEED)
    p.add_argument('--methods',default='all')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stop-on-error',action='store_true')
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--cases',default=None)
    p.add_argument('--display-methods',default='KBR,LRTFR,OTLRM,SALTS,S2NTNN,Ours')
    p.add_argument('--frame',type=int,default=-1)
    p.add_argument('--scale',type=int,default=1)
    args=p.parse_args()
    return dict(prepare=do_prepare,run=do_run,evaluate=do_evaluate,visualize=do_visualize,inspect=do_inspect)[args.action](args)
if __name__=='__main__':main()
