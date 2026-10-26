#!/usr/bin/env python3
"""DAVIS2017 unified 9-method tube-wise completion benchmark.

Subcommands: prepare | run | evaluate | visualize | inspect.
No test GT is passed to the model runners: same saved case file contains it for
legacy compatibility, but the provided adapters read only observed/mask/init.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy.io as sio

ROOT=Path(__file__).resolve().parents[2]
BENCH=ROOT/'benchmark_suite'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BENCH/'Benchmark'))
sys.path.insert(0,str(BENCH/'workflow'))
from benchmark.io_utils import make_tube_mask, linear_spatial_init, save_case
from benchmark.evaluate_output import load_recon, align_shape
from benchmark.metrics import psnr, temporal_gradient_mae
from skimage.metrics import structural_similarity

ALL=['DRO-TFF','HLRTF','KBR','LRTFR','OTLRM','TNN','SALTS','S2NTNN','Ours']
PY_METHODS=['DRO-TFF','HLRTF','LRTFR','OTLRM','S2NTNN','Ours']
MAT_METHODS=['KBR','TNN','SALTS']
RATES=[0.2,0.3,0.4]
DEFAULT_SEED=20261014

def stamp(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None: fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def seed_for(name,rep):
    # One fixed seed per user-selected experiment; no video-dependent seed mixing.
    return int(rep)
def fp(sr):return f'{int(round(sr*100)):03d}'
def parse_values(s,typ):return [typ(x.strip()) for x in s.split(',') if x.strip()]
def ensure_rates(rates):
    if any(not 0<r<1 for r in rates):raise ValueError('SR must be between 0 and 1')
    if len(rates)!=len(set(rates)):raise ValueError('Duplicate sampling rates')
def run_dir(args):return Path(args.output).expanduser().resolve()
def prep_dir(args):return Path(args.data).expanduser().resolve()
def index_path(out):return out/'cases_index.csv'
def report_path(out,args):
    if not args.videos or args.videos.strip().lower()=='all':return out
    names=parse_values(args.videos,str)
    return out/'reports'/('selected__'+'__'.join(sorted(names)))
def selection(rows,args):
    req=set(parse_values(args.videos,str)) if args.videos and args.videos.strip().lower()!='all' else None
    if req and req-set(r['name'] for r in rows):raise ValueError('Missing videos in manifest: '+str(req-set(r['name'] for r in rows)))
    return [r for r in rows if req is None or r['name'] in req]
def shard_rows(rows,args):
    if args.shards < 1 or not 0<=args.shard<args.shards:raise ValueError('Invalid shard index')
    return [r for i,r in enumerate(rows) if i%args.shards==args.shard]
def validated_input(row,folder):
    file=folder/row['file']
    if not file.exists():raise FileNotFoundError(file)
    arr=np.load(file,allow_pickle=False)
    if arr.ndim!=3 or not np.isfinite(arr).all() or arr.min()<0 or arr.max()>1:raise ValueError(f'Invalid normalized HWT: {file}')
    if row.get('sha256') and sha(file)!=row['sha256']:raise ValueError('Input SHA256 mismatch '+str(file))
    return np.asarray(arr,dtype=np.float32)

def do_prepare(args):
    """Append chosen cases without overwriting cases_index.csv from earlier selections.

    All records in one results directory use the same one seed, frozen parameters,
    and a fixed set of three possible sampling rates.
    """
    folder=prep_dir(args); out=run_dir(args); cases=out/'cases';cases.mkdir(parents=True,exist_ok=True)
    rows=selection(read_csv(folder/'manifest.csv'),args)
    rates=parse_values(args.rates,float);ensure_rates(rates)
    if any(round(v,4) not in RATES for v in rates):
        raise ValueError('Only predeclared SR=0.2,0.3,0.4 are permitted in this experiment')
    seed=int(args.seed)
    protocol=out/'PROTOCOL.json'
    frozen_sha=sha(BENCH/'config'/'method_parameters.json')
    if protocol.exists():
        old=json.loads(protocol.read_text(encoding='utf-8'))
        if old.get('seed')!=seed or old.get('frozen_params_sha256')!=frozen_sha:
            raise RuntimeError('Incompatible existing output: seed or model parameters differ. Use a different --output directory.')
    idx=index_path(out)
    existing={e['case']:e for e in read_csv(idx)} if idx.exists() else {}
    additions={}
    for r in rows:
        name=r['name'];gt=validated_input(r,folder);h,w,t=gt.shape
        for sr in rates:
            actual=seed_for(name,seed)
            stem=f'{name}__sr{fp(sr)}__seed{actual}'
            base=cases/stem; npz=base.with_suffix('.npz');mat=base.with_suffix('.mat')
            mask=make_tube_mask(h,w,t,sr,actual)
            assert mask.shape==gt.shape and np.all(mask==mask[:,:,0,None])
            assert int(mask[:,:,0].sum())==int(round(sr*h*w))
            entry=dict(case=stem,name=name,sampling_rate=sr,replicate_seed=seed,model_seed=actual,
                       observed_tubes=int(mask[:,:,0].sum()),total_tubes=h*w,
                       realized_sr=float(mask[:,:,0].mean()),height=h,width=w,frames=t,
                       input_file=r['file'],input_sha256=sha(folder/r['file']))
            if stem in existing:
                old=existing[stem]
                if any(str(old.get(k))!=str(entry.get(k)) for k in
                       ['name','sampling_rate','model_seed','height','width','frames','input_sha256']):
                    raise RuntimeError('Existing case metadata differs for '+stem+'; use a separate output directory')
            additions[stem]=entry
            if npz.exists() and mat.exists() and not args.overwrite:
                with np.load(npz) as saved:
                    if not(np.array_equal(saved['mask'].astype(bool),mask) and np.array_equal(saved['gt'],gt)):
                        raise RuntimeError('Stale case file '+str(npz)+'; do not reuse another mask/data')
                print('SKIP existing',stem,flush=True);continue
            obs=np.zeros_like(gt);obs[mask]=gt[mask];init=linear_spatial_init(obs,mask)
            meta=dict(entry,video_name=name,sampling_rate_requested=sr,
                      sampling_type='exact-count random fixed-spatial tube mask, repeated across time',
                      gt_only_for_evaluation=True,config_frozen=True)
            save_case(base,gt,obs,mask,init,meta)
            print('READY',stem,f'SR={mask[:,:,0].mean():.6f}',flush=True)
    if not additions:raise ValueError('No cases selected')
    merged={**existing,**additions}
    ordered=sorted(merged.values(),key=lambda q:(q['name'],float(q['sampling_rate']),int(q['model_seed'])))
    tmp=idx.with_suffix('.csv.tmp');write_csv(tmp,ordered);os.replace(tmp,idx)
    info=dict(dataset='DAVIS2017 validation frames',rates=RATES,prepared_rates=sorted(set(float(e['sampling_rate']) for e in ordered)),
              seed=seed,seeds=[seed],prepared_videos=sorted(set(e['name'] for e in ordered)),
              images='144x256x50 HWT (default preparation)',methods=ALL,
              missing_fraction=[1-r for r in RATES],date=stamp(),
              frozen_params=str(BENCH/'config'/'method_parameters.json'),frozen_params_sha256=frozen_sha,
              preprocessing_sha256=sha(folder/'PREPROCESSING.json') if (folder/'PREPROCESSING.json').exists() else None)
    protocol.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PREPARED this selection:',len(additions),'cases; cumulative:',len(ordered),'cases; cumulative expected runs:',len(ordered)*len(ALL),flush=True)

def _valid_out(case,out):
    if not out.exists() or out.stat().st_size<200:return False,'missing/small'
    try:
        d=sio.loadmat(case,variable_names=['mask']);shape=d['mask'].shape
        rec,_=load_recon(out); rec=align_shape(rec,shape)
        if not np.all(np.isfinite(rec)):return False,'NaN/Inf output'
        return True,'ok'
    except Exception as e:return False,str(e)

def _mat_command(method,case,out,seed):
    def quote(s):return str(s).replace("'","''")
    cmd=f"addpath('{quote(HERE)}'); run_davis9_matlab('{quote(method)}','{quote(case)}','{quote(out)}',{int(seed)},'{quote(BENCH)}')"
    # Optional thread cap per independent MATLAB worker. No change when unset.
    nthreads=os.environ.get('KPD_MATLAB_THREADS')
    if nthreads:
        n=int(nthreads)
        if not 1<=n<=32:raise ValueError('KPD_MATLAB_THREADS must be in 1..32')
        cmd=f'maxNumCompThreads({n}); '+cmd
    return [os.environ.get('MATLAB_BIN','matlab'),'-batch',cmd]

def do_run(args):
    out=run_dir(args); idx=index_path(out)
    if not idx.exists():raise RuntimeError('Run prepare first')
    cases=shard_rows(selection(read_csv(idx),args),args)
    protocol=out/'PROTOCOL.json'
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')).get('frozen_params_sha256')!=sha(BENCH/'config'/'method_parameters.json'):
        raise RuntimeError('Method parameters changed since prepare; do not silently resume under different defaults.')
    methods=ALL if args.methods=='all' else (PY_METHODS if args.methods=='python' else MAT_METHODS if args.methods=='matlab' else parse_values(args.methods,str))
    if any(m not in ALL for m in methods):raise ValueError('Unknown methods '+str(methods))
    from run_python_suite import build_cmd
    if any(m in PY_METHODS for m in methods) and not args.dry_run:
        try:import torch;print('CUDA:',torch.cuda.is_available(),torch.__version__)
        except ImportError:raise RuntimeError('torch required for Python methods')
    if any(m in MAT_METHODS for m in methods) and not args.dry_run and shutil.which(os.environ.get('MATLAB_BIN','matlab')) is None:
        raise RuntimeError('MATLAB not found on PATH. Set MATLAB_BIN or run --methods python first.')
    failures=[]; counter=0
    for entry in cases:
        case=out/'cases'/(entry['case']+'.mat')
        if not case.exists():raise FileNotFoundError(case)
        for method in methods:
            counter+=1
            rec=out/'reconstructions'/method/(method+'__'+entry['case']+'.mat')
            rec.parent.mkdir(parents=True,exist_ok=True)
            valid,reason=_valid_out(case,rec)
            if valid and not args.overwrite:
                print(f'SKIP [{counter}] {method} {entry["case"]}',flush=True);continue
            if rec.exists() and not args.dry_run:
                bad=rec.with_suffix('.invalid_'+str(int(time.time()))+'.mat');os.replace(rec,bad)
                print('QUARANTINED',rec,'->',bad,'reason',reason)
            partial=rec.with_name(rec.stem+'.partial.mat')
            if partial.exists() and not args.dry_run:partial.unlink()
            seed=int(entry['model_seed'])
            cmd=build_cmd(method,case,partial) if method in PY_METHODS else _mat_command(method,case,partial,seed)
            log=out/'logs'/method/(entry['case']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            print('RUN',stamp(),method,entry['case'],'->',log,flush=True)
            if args.dry_run:
                print('DRY-RUN',repr(cmd),flush=True)
                continue
            t0=time.perf_counter()
            with log.open('w',encoding='utf-8',errors='replace') as fh:
                fh.write('COMMAND: '+repr(cmd)+'\n');fh.flush()
                proc=subprocess.run(cmd,cwd=BENCH,stdout=fh,stderr=subprocess.STDOUT,env=os.environ.copy())
            wall=time.perf_counter()-t0
            ok,detail=_valid_out(case,partial)
            if proc.returncode==0 and ok:
                os.replace(partial,rec);status='OK';print('OK',method,entry['case'],f'{wall:.1f}s',flush=True)
            else:
                status='FAILED';failures.append((method,entry['case']))
                print('FAILED',method,entry['case'],'rc=',proc.returncode,detail,log,flush=True)
            record=dict(timestamp=stamp(),method=method,case=entry['case'],status=status,rc=proc.returncode,
                        wall_seconds=round(wall,3),log=str(log),reason=detail)
            journal=out/'task_status.jsonl';
            with journal.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            if args.stop_on_error and status!='OK':raise RuntimeError(f'{method} {entry["case"]}: {log}')
    print('FINISHED',len(cases),'cases,',len(methods),'methods; failures=',len(failures))
    if failures:sys.exit(2)

def _evaluate_one(case,recon,force_dc=True):
    with np.load(case) as d:
        gt=np.asarray(d['gt'],dtype=np.float32); observed=np.asarray(d['observed'],dtype=np.float32);mask=d['mask'].astype(bool)
    x,runtime=load_recon(recon);x=align_shape(x,gt.shape)
    prior_dc_mae=float(np.mean(np.abs(x[mask]-observed[mask])))
    x=np.clip(x,0,1)
    if force_dc:x[mask]=observed[mask]
    missing=~mask
    full_mse=float(np.mean((gt-x)**2)); miss_mse=float(np.mean((gt[missing]-x[missing])**2))
    full_psnr=10*np.log10(1/max(full_mse,1e-14)); miss_psnr=10*np.log10(1/max(miss_mse,1e-14))
    ss=float(np.mean([structural_similarity(gt[:,:,t],x[:,:,t],data_range=1.) for t in range(gt.shape[2])]))
    tg=float(temporal_gradient_mae(gt,x,mask=missing))
    return dict(psnr_full=full_psnr,psnr_missing=miss_psnr,ssim_full=ss,
                mae_missing=float(np.mean(np.abs(gt[missing]-x[missing]))),temporal_gradient_mae_missing=tg,
                before_dc_mae=prior_dc_mae,method_runtime_seconds=runtime)

def do_evaluate(args):
    out=run_dir(args);rows=selection(read_csv(index_path(out)),args); metrics=[];missing=[]
    report=report_path(out,args);report.mkdir(parents=True,exist_ok=True)
    for entry in rows:
        stem=entry['case'];case=out/'cases'/(stem+'.npz')
        for method in ALL:
            rec=out/'reconstructions'/method/(method+'__'+stem+'.mat')
            valid,reason=_valid_out(out/'cases'/(stem+'.mat'),rec)
            if not valid:
                missing.append(dict(method=method,case=stem,reason=reason));continue
            try:
                vals=_evaluate_one(case,rec)
                metrics.append(dict(method=method,case=stem,name=entry['name'],sampling_rate=float(entry['sampling_rate']),
                                    replicate_seed=int(entry['replicate_seed']),model_seed=int(entry['model_seed']),**vals))
            except Exception as exc:missing.append(dict(method=method,case=stem,reason=str(exc)))
    write_csv(report/'per_run_metrics.csv',metrics)
    write_csv(report/'missing_or_failed.csv',missing,['method','case','reason'])
    if missing:
        print(f'INCOMPLETE: {len(missing)} of {len(rows)*len(ALL)} missing results; NO FINAL MEAN TABLE; see missing_or_failed.csv',flush=True)
        if not args.allow_partial:sys.exit(3)
    if not metrics:raise RuntimeError('No valid metrics')
    agg=[]
    for rate in sorted(set(r['sampling_rate'] for r in metrics)):
        for method in ALL:
            sub=[r for r in metrics if r['sampling_rate']==rate and r['method']==method]
            if not sub:continue
            entry=dict(sampling_rate=rate,method=method,completed=len(sub),expected=len([r for r in rows if float(r['sampling_rate'])==rate]))
            for metric in ['psnr_full','psnr_missing','ssim_full','mae_missing','temporal_gradient_mae_missing','method_runtime_seconds']:
                vals=np.array([r[metric] for r in sub if r[metric] is not None],float)
                entry[metric+'_mean']=float(vals.mean()) if len(vals) else None
                entry[metric+'_std']=float(vals.std(ddof=1)) if len(vals)>1 else None
            agg.append(entry)
    write_csv(report/('SUMMARY_PARTIAL.csv' if missing else 'SUMMARY_COMPLETE.csv'),agg)
    # Prevent a stale COMPLETE table from being mistaken for the current incomplete selection.
    if missing and (report/'SUMMARY_COMPLETE.csv').exists():(report/'SUMMARY_COMPLETE.csv').unlink()
    if not missing and (report/'SUMMARY_PARTIAL.csv').exists():(report/'SUMMARY_PARTIAL.csv').unlink()
    print('EVALUATED',len(metrics),'/',len(rows)*len(ALL),'runs; summary:',('PARTIAL' if missing else 'COMPLETE'),flush=True)
    if not missing: 
        for r in agg:
            print(f"SR={r['sampling_rate']:.1f} {r['method']:8s} PSNR(full)={r['psnr_full_mean']:.3f} PSNR(miss)={r['psnr_missing_mean']:.3f} SSIM={r['ssim_full_mean']:.4f}")

def _choose_roi(gt,mask,frame,roi_h=46,roi_w=60):
    h,w=gt.shape[:2];roi_h=min(h,roi_h);roi_w=min(w,roi_w)
    from scipy.ndimage import uniform_filter
    im=gt[:,:,frame]
    gy,gx=np.gradient(im)
    sal=(np.abs(gx)+np.abs(gy))* (~mask[:,:,frame])
    scores=uniform_filter(sal.astype(float),size=(roi_h,roi_w),mode='constant')
    pad_h=roi_h//2; pad_w=roi_w//2
    scores[:pad_h,:]=0;scores[h-(roi_h-pad_h):,:]=0
    scores[:,:pad_w]=0;scores[:,w-(roi_w-pad_w):]=0
    yy,xx=np.unravel_index(scores.argmax(),scores.shape)
    top=min(max(int(yy-pad_h),0),h-roi_h);left=min(max(int(xx-pad_w),0),w-roi_w)
    return top,left,roi_h,roi_w

def do_visualize(args):
    from PIL import Image,ImageDraw,ImageFont
    out=run_dir(args);report=report_path(out,args);summary=report/'SUMMARY_COMPLETE.csv'
    if not summary.exists():raise RuntimeError('Complete metrics required before visual shortlist. Run evaluate without missing outputs.')
    metrics=read_csv(report/'per_run_metrics.csv'); idx=selection(read_csv(index_path(out)),args)
    base_seed=min(int(r['replicate_seed']) for r in idx)
    bycase={}
    for row in metrics:
        if int(row['replicate_seed'])!=base_seed:continue
        bycase.setdefault(row['case'],{})[row['method']]=row
    shortlist=[]
    for stem,methods in bycase.items():
        if set(methods)!=set(ALL):continue
        ours=float(methods['Ours']['psnr_missing'])
        highest=max(float(methods[m]['psnr_missing']) for m in ALL if m!='Ours')
        shortlist.append(dict(case=stem,name=methods['Ours']['name'],sampling_rate=float(methods['Ours']['sampling_rate']),
                              delta_vs_best_baseline=ours-highest,ours_psnr_missing=ours,best_baseline_psnr_missing=highest))
    shortlist.sort(key=lambda x:x['delta_vs_best_baseline'],reverse=True)
    write_csv(report/'QUALITATIVE_ALL_CANDIDATES.csv',shortlist)
    if args.cases:
        stems=parse_values(args.cases,str)
        for s in stems:
            if s not in bycase:raise ValueError('Unknown visual case '+s)
    else:
        # Choose distinct videos for favorable examples, plus one transparent failure case.
        wins=[];used=set()
        for r in shortlist:
            if r['name'] not in used and r['delta_vs_best_baseline']>0 and len(wins)<2:
                wins.append(r['case']);used.add(r['name'])
        if shortlist:
            low=next((r['case'] for r in shortlist[::-1] if r['name'] not in used),None)
            stems=wins+([low] if low else [])
        else:stems=[]
    display=parse_values(args.display_methods,str)
    if any(m not in ALL for m in display):raise ValueError('Invalid visual display methods')
    methods=['Observed']+display+['GT'];frame=args.frame
    visdir=report/'qualitative';visdir.mkdir(exist_ok=True,parents=True)
    for stem in stems:
        case=out/'cases'/(stem+'.npz')
        with np.load(case) as arr:
            gt=arr['gt']; obs=arr['observed'];mask=arr['mask'].astype(bool)
        t=(gt.shape[2]//2 if frame<0 else min(frame,gt.shape[2]-1))
        top,left,rh,rw=_choose_roi(gt,mask,t)
        videos={'Observed':obs,'GT':gt}
        for method in display:
            f=out/'reconstructions'/method/(method+'__'+stem+'.mat')
            x,_=load_recon(f);x=align_shape(x,gt.shape);x=np.clip(x,0,1);x[mask]=obs[mask]
            videos[method]=x
        scale=args.scale;panel_w=gt.shape[1];panel_h=gt.shape[0]
        gap=10;header=29;small=gt.shape[1]//2;small_h=round(rh*small/rw)
        total_w=len(methods)*(panel_w+gap)+gap
        canvas=Image.new('RGB',(total_w,header+panel_h+12+small_h+35),'white');d=ImageDraw.Draw(canvas)
        for j,method in enumerate(methods):
            xx=gap+j*(panel_w+gap)
            img=Image.fromarray((np.clip(videos[method][:,:,t],0,1)*255).astype('uint8')).convert('RGB')
            d.text((xx+5,6),method,fill=(20,20,20))
            canvas.paste(img,(xx,header));d.rectangle((xx+left,header+top,xx+left+rw,header+top+rh),outline=(240,45,40),width=2)
            crop=img.crop((left,top,left+rw,top+rh)).resize((small,small_h),Image.Resampling.NEAREST)
            canvas.paste(crop,(xx,header+panel_h+12))
            if method not in ('GT','Observed'):
                psnr_m=float(bycase[stem][method]['psnr_missing'])
                d.text((xx+3,header+panel_h+small_h+17),f'M-PSNR {psnr_m:.2f}',fill=(20,20,20))
        if args.scale>1: canvas=canvas.resize((canvas.width*args.scale,canvas.height*args.scale),Image.Resampling.NEAREST)
        saved=visdir/(stem+'.png');canvas.save(saved)
        print('VISUAL',saved,flush=True)
    (visdir/'SELECTION_POLICY.txt').write_text(
        'All nine methods are quantitatively evaluated on all predefined clips/rates/seeds.\n'
        'Automatic qualitative candidates: two distinct-video cases with positive Ours vs strongest-baseline missing-PSNR margin, plus one worst non-overlapping case.\n'
        'All candidate scores are saved in QUALITATIVE_ALL_CANDIDATES.csv. Selection is illustrative, not an unbiased performance estimate.\n'
        'ROI is selected from GT gradient energy at missing tubes ONLY; it is identical for all methods.\n',encoding='utf-8')
    print('Chosen',stems,'(including illustrative negative when available)')

def do_inspect(args):
    out=run_dir(args);idx=selection(read_csv(index_path(out)),args); issues=[]
    for e in idx:
        case=out/'cases'/(e['case']+'.mat')
        for method in ALL:
            rec=out/'reconstructions'/method/(method+'__'+e['case']+'.mat')
            ok,reason=_valid_out(case,rec)
            if not ok:issues.append((method,e['case'],reason))
    print('CASes',len(idx),'expected outputs',len(idx)*len(ALL),'good',len(idx)*len(ALL)-len(issues),'missing or bad',len(issues))
    for m,c,r in issues[:25]:print('MISSING',m,c,r)
    if issues:sys.exit(2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','evaluate','visualize','inspect'])
    p.add_argument('--data',default=str(BENCH/'data'/'davis2017_new'))
    p.add_argument('--output',default=str(ROOT/'results'/'davis2017_all9'))
    p.add_argument('--videos',default=None,help='comma separated subset')
    p.add_argument('--rates',default='0.2,0.3,0.4',help='sampling rates, NOT missing rates')
    p.add_argument('--seed',type=int,default=DEFAULT_SEED,help='ONE fixed seed shared by all videos and methods')
    p.add_argument('--methods',default='all',help='all, python, matlab or comma-separated method names')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--stop-on-error',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='print commands only; never execute models')
    p.add_argument('--allow-partial',action='store_true',help='partial summary clearly labeled, not paper ready')
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--cases',default=None,help='manual visualization list of case IDs')
    p.add_argument('--display-methods',default='KBR,LRTFR,OTLRM,SALTS,S2NTNN,Ours')
    p.add_argument('--frame',type=int,default=-1)
    p.add_argument('--scale',type=int,default=1)
    args=p.parse_args()
    return {'prepare':do_prepare,'run':do_run,'evaluate':do_evaluate,
            'visualize':do_visualize,'inspect':do_inspect}[args.action](args)
if __name__=='__main__':main()
