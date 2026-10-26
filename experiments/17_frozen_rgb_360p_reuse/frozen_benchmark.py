#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v13: Reuse v8 eight RGB baselines, compute/reuse Ours-Frozen, audit and plot 8-column figure.

Formal nine = eight frozen baseline methods + Ours-Frozen. Ours-Original is a supplementary
within-method comparator. Keep all original outputs read-only. GT never enters solver subprocess.
Blackswan parameter-selection results are exploratory; never automatically label them as test.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / '15_ours_steps_patch_360p'))
sys.path.insert(0, str(ROOT / 'experiments' / '10_davis2017_rgb_all9'))
import steps_patch_rgb360 as v11
import rgb9 as orig

SEED=20261014
K=12
HIDDEN=32
STEPS=3600
PATCH=96
KERNEL=3
SHAPE=(360,640,50,3)
BASELINES=('DRO-TFF','HLRTF','KBR','LRTFR','OTLRM','TNN','SALTS','S2NTNN')
MAIN=('Observed','KBR','TNN','SALTS','S2NTNN','Ours-Original','Ours-Frozen','GT')
SUPPLEMENT=('Observed',*BASELINES,'Ours-Original','Ours-Frozen','GT')
METHODS=(*BASELINES,'Ours-Original','Ours-Frozen')
FORMAL=(*BASELINES,'Ours-Frozen')
RATES=(0.2,0.3,0.4)
DEFAULT_SOURCE=Path(os.environ.get('DAVIS_360P_SOURCE',str(Path.home() / "kpd_data" / 'KPD' / 'results' / 'davis2017_rgb_all9_seed20261014_360p_f050')))
DEFAULT_V12=Path(os.environ.get('KPD_V12_RESULTS',str(Path.home() / "kpd_data" / 'KPD_StepsPatch_v12' / 'results' / 'steps_patch_extended_rgb360p_seed20261014')))
DEFAULT_V14=Path(os.environ.get('KPD_V14_RESULTS',str(Path.home() / "kpd_data" / 'KPD_DAVIS_RGB360_Final_v14' / 'results' / 'DAVIS_RGB_360p_FINAL_K12_C32_T3600_P96')))
V14_REUSE=DEFAULT_V14
SUPPLEMENT_FORMAL=('Observed',*BASELINES,'Ours-Frozen','GT')
DEFAULT_OUT=ROOT/'results'/'frozen_rgb360p_seed20261014_K12_C32_T3600_P96'


def write_csv(path,rows,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)

def ffloat(x):
    try:return float(x)
    except (TypeError,ValueError,KeyError):return float('nan')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def case_id(video,sr):return f'{video}__sr{round(sr*100):03d}__seed{SEED}'

def frozen_out(out,case):return v11.rgb_path(out,case,STEPS,PATCH)

def v12_out(v12,case):return v11.rgb_path(v12,case,STEPS,PATCH)

def original_out(source,method,case):
    return orig.rgb_out(source,'Ours' if method=='Ours-Original' else method,case)

def _valid_mat(path,shape,strict_range=False):
    if not path.is_file() or path.stat().st_size<100:return False,'file missing / empty'
    try:
        with np.errstate(invalid='ignore'):
            a=loadmat(path,variable_names=['recon'])['recon']
            if tuple(a.shape)!=tuple(shape):return False,f'wrong shape {a.shape}'
            if not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all():return False,'nonfinite / nonnumeric'
            if strict_range and (a.min()<-0.001 or a.max()>1.001):return False,'outside normalized [0,1] range'
        return True,''
    except (OSError,ValueError,KeyError,TypeError) as e:return False,str(e)


def _validate_frozen(path):
    if not v11._valid_rgb(path,STEPS,PATCH):return False,'bad frozen shape / config metadata'
    return True,''


def same_path(a,b):
    return os.path.normcase(os.path.abspath(str(a))).rstrip('\\/').lower()==os.path.normcase(os.path.abspath(str(b))).rstrip('\\/').lower()


def check_source(source,v12=None,force_v12=False):
    p=source/'PROTOCOL.json'
    if not p.is_file():raise FileNotFoundError(f'Missing v8 PROTOCOL.json: {p}')
    d=json.loads(p.read_text(encoding='utf-8'))
    if d.get('color')!='RGB' or int(d.get('seed',-1))!=SEED:
        raise ValueError('v8 RGB source protocol mismatch: color or seed')
    if v12 is not None and v12.is_dir() and (v12/'STEPS_PATCH_V12_PROTOCOL.json').is_file():
        vd=json.loads((v12/'STEPS_PATCH_V12_PROTOCOL.json').read_text(encoding='utf-8'))
        old_source=vd.get('source_output')
        if old_source and not same_path(source,old_source):
            msg=f'v12 C2 was trained on another v8 source: {old_source} (current {source})'
            if not force_v12:raise ValueError(msg+'\nUse same --source or explicitly --trust-v12-source for moved but verified data')
            print('WARNING',msg,flush=True)
    # Verify old v14 frozen reconstructions were trained on the same v8 input source.
    v14_protocol=V14_REUSE/'FROZEN_V13_PROTOCOL.json'
    if v14_protocol.is_file():
        old=json.loads(v14_protocol.read_text(encoding='utf-8'))
        prev=old.get('input_source')
        if prev and not same_path(source,prev):
            msg=f'Old v14 outputs use a different v8 input source: {prev} (current {source})'
            if not force_v12:raise ValueError(msg+'; do not silently mix masks from different inputs')
            print('WARNING',msg,flush=True)
    return d


def load_case(source,case):
    p=orig.paths(source,case)
    if not p.is_file():raise FileNotFoundError(f'Prepared RGB input case missing: {p}')
    gt,obs,mask=orig.load_case_rgb(source,case)
    if tuple(gt.shape)!=SHAPE or tuple(obs.shape)!=SHAPE or tuple(mask.shape)!=SHAPE:
        raise ValueError(f'RGB 360p source mismatch in {case}: {gt.shape}/{obs.shape}/{mask.shape}')
    if not np.isfinite(gt).all() or not np.isfinite(obs).all() or np.max(gt)>1.001 or np.min(gt)<-0.001:
        raise ValueError('Invalid source range')
    if not np.array_equal(mask[:,:,:,0],mask[:,:,:,1]) or not np.array_equal(mask[:,:,:,0],mask[:,:,:,2]):
        raise ValueError('Nonidentical RGB masks: reject')
    if not np.array_equal(mask[:,:,0,0],mask[:,:,:,0][:,:,-1]):
        raise ValueError('Not tube-wise spatial mask: reject')
    if not np.allclose(obs[mask],gt[mask],atol=1e-6) or np.max(np.abs(obs[~mask]))>1e-7:
        raise ValueError('Observed case inconsistent with gt/mask')
    return gt,obs,mask


def locate_recon(source,v12,out,case,method):
    if method=='Ours-Frozen':
        dst=frozen_out(out,case)
        ok,_=_validate_frozen(dst)
        if ok:return dst,'v13_frozen'
        old_v14=frozen_out(V14_REUSE,case)
        ok,_=_validate_frozen(old_v14)
        if ok:return old_v14,'v14_read_only'
        src=v12_out(v12,case)
        ok,_=_validate_frozen(src)
        if ok:return src,'v12_C2_read_only'
        return None,'v12_and_v13_missing'
    p=original_out(source,method,case)
    ok,why=_valid_mat(p,SHAPE)
    if ok:return p,'v8_read_only'
    return None,why


def protocol(out,source,v12):
    d={'version':13,'dataset':'DAVIS2017 RGB 360x640x50','seed':SEED,'rates':list(RATES),
       'input_source':str(source),'reference_v12':str(v12),
       'frozen_config':{'rank':K,'hidden':HIDDEN,'steps':STEPS,'patch_t':16,'patch_h':PATCH,
                        'patch_w':PATCH,'spatial_kernel':KERNEL,'temporal_kernel':3,'crossfit_folds':5,
                        'masked_sweeps':12,'batch_size':2,'max_residual':0.25},
       'main_columns':['Observed','DRO-TFF','KBR','TNN','SALTS','S2NTNN','Ours-Frozen','GT'],
       'supplement_columns':list(SUPPLEMENT_FORMAL),
       'baseline_policy':'Read-only v8 results, exact same prepared RGB case, same masks.',
       'parameter_role':'Provisional configuration selected on Blackswan; exploratory/development; not unseen test.'}
    dest=out/'FROZEN_V13_PROTOCOL.json'
    if dest.is_file() and json.loads(dest.read_text(encoding='utf-8'))!=d:
        raise RuntimeError('Existing --output uses another protocol. Choose a new output folder.')
    out.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')


def run_frozen(source,v12,out,videos,rates,args):
    failures=[]
    for video in videos:
        for sr in rates:
            case=case_id(video,sr)
            # Fail early if new video not preprocessed by v8.
            if not orig.paths(source,case).exists():
                print('PREPARED CASE MISSING',case,'->',orig.paths(source,case),flush=True)
                failures.append((case,'case'));continue
            # Validate v12 C2 and source protocol BEFORE skipping an old result.
            existing=locate_recon(source,v12,out,case,'Ours-Frozen')
            if existing[0] is not None and not args.force_retrain:
                print('SKIP frozen already valid:',case,'from',existing[1],flush=True);continue
            if args.dry_run:
                print('DRY RUN would train frozen:',case,'K=12 C=32 steps=3600 patch=16x96x96',flush=True)
                continue
            print('RUN Ours-Frozen:',case,flush=True)
            try:
                runner=argparse.Namespace(device=args.device,dry_run=False,overwrite=args.force_retrain,
                                          stop_on_error=args.stop_on_error)
                ok=v11.run_one(source,out,video,sr,STEPS,PATCH,runner)
                if not ok:failures.append((case,'run failed'))
            except Exception as e:
                failures.append((case,str(e)))
                print('RUN ERROR',case,repr(e),flush=True)
            if failures and args.stop_on_error:break
        if failures and args.stop_on_error:break
    return failures


NUM_FIELDS=['psnr_full','psnr_missing','ssim_full','mae_missing','temporal_gradient_mae_missing','runtime_seconds']
ROWS_FIELDS=['video','case','sampling_rate','seed','method','configuration','split_role','status','source_version',
             'psnr_full','psnr_missing','ssim_full','mae_missing','temporal_gradient_mae_missing','runtime_seconds','reconstruction_file']


def evaluate(source,v12,out,videos,rates,split_role):
    records=[];missing=[];audits=[]
    for video in videos:
        for sr in rates:
            case=case_id(video,sr)
            try:
                gt,obs,mask=load_case(source,case)
            except Exception as e:
                print('MISSING/INVALID SOURCE',case,str(e),flush=True)
                for meth in METHODS:missing.append({'case':case,'method':meth,'reason':'source: '+str(e)})
                continue
            expected_sr=float(mask[:,:,0,0].mean())
            if abs(expected_sr-sr)>2/(SHAPE[0]*SHAPE[1]):
                raise ValueError(f'Observed sampling rate unexpectedly differs: {case}: {expected_sr}')
            audits.append({'case':case,'source_file':str(orig.paths(source,case)),
                           'source_sha256':sha256(orig.paths(source,case)),
                           'actual_sampling_rate':expected_sr,'mask_observed':int(mask[:,:,0,0].sum()),
                           'mask_total':int(mask.shape[0]*mask.shape[1])})
            for method in METHODS:
                p,source_ver=locate_recon(source,v12,out,case,method)
                if p is None:
                    # Historical Ours-Original is OPTIONAL and is not part of the formal nine.
                    if method=='Ours-Original':continue
                    missing.append({'case':case,'method':method,'reason':source_ver});continue
                try:
                    d=loadmat(p,variable_names=['recon','runtime_seconds'])
                    rec=np.asarray(d['recon'],dtype=np.float32)
                    if not np.isfinite(rec).all():raise ValueError('nonfinite recon')
                    # Validate observed values; audit excessive difference but do not silently modify files.
                    before=float(np.max(np.abs(rec[mask]-obs[mask])))
                    if before>0.002:raise ValueError(f'observed pixel mismatch max={before:.5g}; possible mask disagreement')
                    val=orig.metric_for(gt,obs,mask,rec)
                    runtime=float(np.asarray(d['runtime_seconds']).reshape(-1)[0]) if 'runtime_seconds' in d else float('nan')
                    role=split_role if video!='blackswan' else 'development_exploratory'
                    records.append({'video':video,'case':case,'sampling_rate':sr,'seed':SEED,'method':method,
                                    'configuration':'K12_C32_T3600_P96' if method=='Ours-Frozen' else 'v8_original',
                                    'split_role':role,'status':'VALID','source_version':source_ver,
                                    'runtime_seconds':runtime,'reconstruction_file':str(p),
                                    **{x:val[x] for x in ('psnr_full','psnr_missing','ssim_full','mae_missing','temporal_gradient_mae_missing')}})
                except Exception as e:missing.append({'case':case,'method':method,'reason':repr(e)})
                finally:
                    if 'd' in locals():del d
                    if 'rec' in locals():del rec
            del gt,obs,mask
            print('EVAL COMPLETE',case,f'present={sum(r["case"]==case for r in records)}/{len(METHODS)}',flush=True)
    # Preserve prior videos/rates when running a new subset later. A refreshed selected
    # case REPLACES its old metrics, including previously invalid entries.
    prior_path=out/'reports'/'per_run_metrics.csv'
    previous=[]
    if prior_path.is_file():
        with prior_path.open('r',encoding='utf-8-sig',newline='') as f:previous=list(csv.DictReader(f))
    selected={(v,round(sr,6)) for v in videos for sr in rates}
    merged={(r['video'],round(float(r['sampling_rate']),6),r['method']):r
            for r in previous if (r['video'],round(float(r['sampling_rate']),6)) not in selected}
    merged.update({(r['video'],round(float(r['sampling_rate']),6),r['method']):r for r in records})
    cumulative=sorted(merged.values(),key=lambda r:(r['video'],float(r['sampling_rate']),r['method']))
    write_csv(prior_path,cumulative,ROWS_FIELDS)
    write_csv(out/'reports'/'missing_or_invalid.csv',missing,['case','method','reason'])
    write_csv(out/'reports'/'source_mask_audit.csv',audits,
              ['case','source_file','source_sha256','actual_sampling_rate','mask_observed','mask_total'])
    # For main formal table 8 baselines + new ours, WITHOUT old-ours double counting.
    expected=len(videos)*len(rates)*len(FORMAL)
    found=sum(r['method'] in FORMAL for r in records)
    complete=found==expected
    write_csv(out/'reports'/'STATUS.csv',[{'status':'COMPLETE' if complete else 'PARTIAL',
              'formal_completed':found,'formal_expected':expected,'missing_files':len(missing),
              'selected_videos':','.join(videos),'selected_rates':','.join(str(x) for x in rates)}],
              ['status','formal_completed','formal_expected','missing_files','selected_videos','selected_rates'])
    cumulative_videos=sorted(set(r['video'] for r in cumulative))
    cumulative_rates=sorted(set(float(r['sampling_rate']) for r in cumulative))
    _summary_and_tex(out,cumulative,cumulative_videos,cumulative_rates,complete)
    _plot_curves(out,cumulative,cumulative_videos,cumulative_rates)
    print('EVALUATION:',found,'/',expected,'formal nine-method records;',
          'COMPLETE' if complete else 'PARTIAL',f'{len(missing)} missing/invalid overall',flush=True)
    return records,complete


def _summary_and_tex(out,records,videos,rates,complete):
    by={(r['video'],float(r['sampling_rate']),r['method']):r for r in records}
    aggs=[]
    for rate in rates:
        for method in METHODS:
            group=[by[(v,rate,method)] for v in videos if (v,rate,method) in by]
            row={'sampling_rate':rate,'method':method,'completed_videos':len(group),'expected_videos':len(videos)}
            for key in NUM_FIELDS:
                vals=np.array([ffloat(x.get(key)) for x in group],dtype=float)
                valid=vals[np.isfinite(vals)]
                row[key+'_mean']=float(valid.mean()) if len(valid) else ''
                row[key+'_std']=float(valid.std(ddof=1)) if len(valid)>1 else ''
            aggs.append(row)
    aggcols=['sampling_rate','method','completed_videos','expected_videos']+[c+suf for c in NUM_FIELDS for suf in ('_mean','_std')]
    write_csv(out/'reports'/'SUMMARY_BY_RATE.csv',aggs,aggcols)
    # LaTeX one row per method, columns grouped SR; asterisk where incomplete via N/M count.
    dest=out/'tables'/'TABLE_MAIN_9METHODS.tex';dest.parent.mkdir(parents=True,exist_ok=True)
    header=['Method']+[f'SR={sr:g}' for sr in rates]
    lines=['% AUTO-GENERATED. Missing-region PSNR in dB. Development vs unseen test must be disclosed.',
           '% 8 baselines + new frozen Ours; Ours-Original intentionally NOT a 10th formal baseline.',
           r'\begin{tabular}{l'+'r'*len(rates)+'}',r'\toprule',
           ' & '.join(header)+r' \\',r'\midrule']
    for m in FORMAL:
        cells=[m.replace('_',r'\_')]
        for sr in rates:
            matching=next((r for r in aggs if r['method']==m and float(r['sampling_rate'])==sr),None)
            if matching and matching['completed_videos']==matching['expected_videos']:
                cells.append(f'{ffloat(matching["psnr_missing_mean"]):.3f}')
            else:cells.append('--')
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    dest.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Separate before-vs-after for documentation.
    improvements=[]
    for video in videos:
        for sr in rates:
            a=by.get((video,sr,'Ours-Original'));b=by.get((video,sr,'Ours-Frozen'))
            if a and b:
                improvements.append({'video':video,'sampling_rate':sr,'old_psnr_missing':a['psnr_missing'],
                                     'new_psnr_missing':b['psnr_missing'],
                                     'delta_psnr_missing':float(b['psnr_missing'])-float(a['psnr_missing']),
                                     'old_ssim_full':a['ssim_full'],'new_ssim_full':b['ssim_full'],
                                     'delta_ssim_full':float(b['ssim_full'])-float(a['ssim_full'])})
    write_csv(out/'reports'/'OURS_BEFORE_AFTER.csv',improvements,
              ['video','sampling_rate','old_psnr_missing','new_psnr_missing','delta_psnr_missing',
               'old_ssim_full','new_ssim_full','delta_ssim_full'])


def _plot_curves(out,records,videos,rates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    for key,label in [('psnr_missing','Missing-region PSNR (dB)'),('ssim_full','Full-frame SSIM')]:
        fig,ax=plt.subplots(figsize=(8.8,4.8))
        for m in FORMAL:
            points=[]
            for sr in rates:
                # Plot ONLY if all selected videos present for that method+rate.
                rr=[r for r in records if r['method']==m and float(r['sampling_rate'])==sr]
                if len(rr)==len(videos):points.append((sr,float(np.mean([float(r[key]) for r in rr]))))
            if points:
                ax.plot([p[0] for p in points],[p[1] for p in points],marker='o',
                        linewidth=2.2 if m=='Ours-Frozen' else 1.2,
                        linestyle='-' if m=='Ours-Frozen' else ('--' if m=='Ours-Original' else '-'),
                        label=m)
        ax.set_xlabel('Sampling rate (observed fraction)');ax.set_ylabel(label)
        ax.set_title('Exploratory/development: Blackswan is tuning sequence' if 'blackswan' in videos
                     else 'Cross-video comparison — check split roles')
        ax.set_xticks(rates);ax.grid(alpha=.22)
        ax.legend(ncol=5,fontsize=7,loc='best')
        fig.tight_layout()
        p=figdir/('curve_'+key);fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight')
        fig.savefig(p.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig)


def _roi(gt,mask,t,rh=70,rw=105):
    # Method-neutral selection based ONLY on GT gradient at missing sites, never a method's performance.
    from scipy.ndimage import uniform_filter
    h,w=gt.shape[:2]
    rh=min(rh,h);rw=min(rw,w)
    grey=np.mean(gt[:,:,t,:],axis=-1)
    gy,gx=np.gradient(grey)
    energy=uniform_filter((np.abs(gy)+np.abs(gx))*(~mask[:,:,t,0]),size=(rh,rw),mode='constant')
    # Clip search to ROI centers that fit wholly inside frame.
    halfh=rh//2;halfw=rw//2
    energy[:halfh,:]=0;energy[h-(rh-halfh):,:]=0
    energy[:,:halfw]=0;energy[:,w-(rw-halfw):]=0
    yy,xx=np.unravel_index(int(np.argmax(energy)),energy.shape)
    return (int(np.clip(xx-halfw,0,w-rw)),int(np.clip(yy-halfh,0,h-rh)),rw,rh)


def visual(source,v12,out,videos,rates,frame,roi_spec,variant,require_complete,main_columns=None):
    from PIL import Image,ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=tuple(main_columns) if main_columns is not None else MAIN
    if (len(chosen)!=8 or len(set(chosen))!=8 or
        set(chosen)-set(SUPPLEMENT) or
        not {'Observed','Ours-Frozen','GT'}.issubset(chosen)):
        raise ValueError('Main visualization needs exactly 8 unique columns, including Observed, Ours-Frozen, GT')
    orders={'main':chosen,'supplement':SUPPLEMENT_FORMAL}
    choice=['main','supplement'] if variant=='both' else [variant]
    for video in videos:
        for sr in rates:
            case=case_id(video,sr)
            if not orig.paths(source,case).is_file():continue
            gt,obs,mask=load_case(source,case)
            t=SHAPE[2]//2 if frame<0 else frame
            if not (0<=t<SHAPE[2]):raise ValueError('Frame number invalid: '+str(t))
            roi=_roi(gt,mask,t) if roi_spec=='auto' else tuple(map(int,roi_spec.split(',')))
            x,y,w,h=roi
            if not (0<=x<SHAPE[1] and 0<=y<SHAPE[0] and w>0 and h>0 and x+w<=SHAPE[1] and y+h<=SHAPE[0]):
                raise ValueError('ROI must be x,y,w,h inside full image')
            # Only read the selected frame from result on heap; mat.load reads full recon but releases immediately.
            frames={'Observed':obs[:,:,t,:].copy(),'GT':gt[:,:,t,:].copy()}
            for method in METHODS:
                p,_=locate_recon(source,v12,out,case,method)
                if p is None:continue
                rec=np.asarray(loadmat(p,variable_names=['recon'])['recon'][:,:,t,:],dtype=np.float32)
                rec[mask[:,:,t,:]]=obs[:,:,t,:][mask[:,:,t,:]]
                frames[method]=np.clip(rec,0,1)
            for kind in choice:
                columns=orders[kind]
                absent=[m for m in columns if m not in frames]
                if absent:
                    print('FIG SKIPPED',case,kind,'missing',','.join(absent),flush=True)
                    if require_complete:raise RuntimeError('Missing reconstruction files for '+case+': '+str(absent))
                    continue
                # Explicit image-sized axes: no enormous blank band between full frame and ROI.
                # Journal figure* at 8 columns needs titles readable after width scaling.
                figure_width=15.8 if kind=='main' else 23.6
                margin=0.10
                gutter=0.105
                tile_width=(figure_width-2*margin-(len(columns)-1)*gutter)/len(columns)
                full_height=tile_width*SHAPE[0]/SHAPE[1]
                zoom_height=tile_width*h/w
                bottom=0.08
                between=0.10
                headroom=0.44
                figure_height=bottom+zoom_height+between+full_height+headroom
                fig=plt.figure(figsize=(figure_width,figure_height),facecolor='white')
                for ix,m in enumerate(columns):
                    x_fig=margin+ix*(tile_width+gutter)
                    full_ax=fig.add_axes([x_fig/figure_width,
                        (bottom+zoom_height+between)/figure_height,
                        tile_width/figure_width,full_height/figure_height])
                    zoom_ax=fig.add_axes([x_fig/figure_width,bottom/figure_height,
                        tile_width/figure_width,zoom_height/figure_height])
                    img=np.uint8(np.clip(frames[m]*255+0.5,0,255))
                    full_ax.imshow(img,aspect='auto')
                    full_ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,edgecolor='#e02424',lw=2.1))
                    full_ax.set_title(m,fontsize=16.0 if kind=='main' else 15.0,
                        fontweight='bold' if m=='Ours-Frozen' else 'normal',pad=7)
                    crop=Image.fromarray(img).crop((x,y,x+w,y+h))
                    zoom_ax.imshow(crop,aspect='auto')
                    for ax in (full_ax,zoom_ax):ax.set_axis_off()
                    zoom_ax.add_patch(plt.Rectangle((0,0),w-1,h-1,fill=False,edgecolor='#e02424',lw=1.7))
                ff=out/'figures'/('main_8columns' if kind=='main' else 'supplement_allmethods')
                ff.mkdir(parents=True,exist_ok=True)
                p=ff/(case+f'__frame{t:03d}')
                fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight')
                fig.savefig(p.with_suffix('.png'),dpi=220,bbox_inches='tight')
                plt.close(fig)
                (p.with_suffix('.json')).write_text(json.dumps({'case':case,'frame':t,'roi_xywh':[x,y,w,h],
                    'columns':list(columns),'selection':'GT gradient with missing-mask weighting; fixed across methods',
                    'role':'exploratory/development' if video=='blackswan' else 'requires split check'},
                    indent=2),encoding='utf-8')
                print('FIG GENERATED',kind,case,p.with_suffix('.pdf'),flush=True)


def parse_list(arg,convert=str):
    return [convert(z.strip()) for z in arg.split(',') if z.strip()]


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE,help='v8 read-only nine-method results root; contains cases/, reconstructions/')
    ap.add_argument('--v12-results',type=Path,default=DEFAULT_V12,help='v12 C2 read-only output root')
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--v14-results',type=Path,default=DEFAULT_V14,help='Prior v14 frozen output for read-only reuse')
    ap.add_argument('--videos',default='blackswan',help='Comma-separated or all; prepare new videos using v8 launcher first')
    ap.add_argument('--rates',default='0.2,0.3,0.4')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--stop-on-error',action='store_true')
    ap.add_argument('--force-retrain',action='store_true')
    ap.add_argument('--trust-v12-source',action='store_true',help='Use only when existing v12 result and source folder were moved, without changing cases')
    ap.add_argument('--evaluate-only',action='store_true')
    ap.add_argument('--fit-only',action='store_true',help='Compute/reuse frozen Ours only; defer report and figures until parallel CPU pool is finished')
    ap.add_argument('--figures-only',action='store_true')
    ap.add_argument('--figure-variant',choices=['main','supplement','both'],default='both')
    ap.add_argument('--main-columns',default=','.join(MAIN),help='Exactly 8 unique columns; must include Observed,Ours-Frozen,GT')
    ap.add_argument('--frame',type=int,default=20)
    ap.add_argument('--roi',default='auto',help='auto or x,y,w,h in source image coordinates')
    ap.add_argument('--require-complete',action='store_true',help='Fail rather than skip incomplete figure')
    ap.add_argument('--split-role',default='development',choices=['development','exploratory','test','legacy'],
                    help='Used only for future videos; Blackswan is ALWAYS tagged development/exploratory')
    a=ap.parse_args(argv)
    source=a.source.expanduser().resolve();v12=a.v12_results.expanduser().resolve();out=a.output.expanduser().resolve()
    index_file=source/'cases_index.csv'
    known=sorted({r['name'] for r in orig.read_csv(index_file)}) if index_file.is_file() else []
    vids=known if a.videos=='all' else parse_list(a.videos)
    if len(set(vids))!=len(vids) or not vids or set(vids)-set(known):
        ap.error('Videos must be distinct and present in prepared v8 cases_index.csv; choices: '+','.join(known))
    rates=parse_list(a.rates,float)
    if len(set(rates))!=len(rates) or not rates or set(rates)-set(RATES):ap.error('Invalid --rates; use 0.2,0.3,0.4')
    if sum((a.evaluate_only,a.figures_only,a.fit_only))>1:ap.error('Choose at most one of --evaluate-only, --figures-only, --fit-only')
    global V14_REUSE
    V14_REUSE=a.v14_results.expanduser().resolve()
    main_cols=parse_list(a.main_columns)
    if len(main_cols)!=8 or len(set(main_cols))!=8 or set(main_cols)-set(SUPPLEMENT) or not {'Observed','Ours-Frozen','GT'}.issubset(main_cols):
        ap.error('--main-columns needs exactly 8 distinct known columns including Observed,Ours-Frozen,GT')
    check_source(source,v12,a.trust_v12_source)
    if not a.dry_run:protocol(out,source,v12)
    if a.figures_only:
        visual(source,v12,out,vids,rates,a.frame,a.roi,a.figure_variant,a.require_complete,main_cols)
        return 0
    if not a.evaluate_only:
        failed=run_frozen(source,v12,out,vids,rates,a)
        if a.dry_run:return 2 if failed else 0
        if failed and a.stop_on_error:
            print('STOP: failed',failed,flush=True);return 2
    if a.fit_only:
        print('FIT-ONLY DONE; evaluation deferred until CPU workers finish.',flush=True)
        return 0
    if not a.dry_run:
        _,complete=evaluate(source,v12,out,vids,rates,a.split_role)
        if a.require_complete and not complete:
            raise RuntimeError('Incomplete nine-method results: see reports/missing_or_invalid.csv')
        visual(source,v12,out,vids,rates,a.frame,a.roi,a.figure_variant,a.require_complete,main_cols)
    return 0

if __name__=='__main__':sys.exit(main())
