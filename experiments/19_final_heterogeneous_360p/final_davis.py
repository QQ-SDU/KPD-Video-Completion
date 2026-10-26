#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAVIS 2017 RGB 360p v15: concurrent CPU MATLAB pool + sequential Python/GPU lane.

One/multiple/all DAVIS sequences. Nine formal methods = eight v8 baseline methods +
Ours-Frozen. Existing v8 Ours-Original may be read, but is NEVER trained on new videos.

No per-video tuning. Fixed K=12, hidden=32, steps=3600, patch=(16,96,96),
3x3x3 kernel, 5 cross-fit folds, seed=20261014. Three SRs: .2/.3/.4.
Source v8 results are append-only; GPU lane and MATLAB CPU pool use distinct method outputs.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments'/'17_frozen_rgb_360p_reuse'))
sys.path.insert(0,str(ROOT/'experiments'/'10_davis2017_rgb_all9'))
import frozen_benchmark as frozen
import rgb9

DATASET_DEFAULT=Path(os.environ.get('DAVIS_ROOT',str(ROOT/'data'/'DAVIS2017')))
V8_DEFAULT=Path(os.environ.get('DAVIS_360P_SOURCE',str(ROOT/'results'/'davis2017_rgb_all9_seed20261014_360p_f050')))
V12_DEFAULT=Path(os.environ.get('KPD_V12_RESULTS',str(ROOT/'results'/'previous_v12')))
V14_DEFAULT=Path(os.environ.get('KPD_V14_RESULTS',str(ROOT/'results'/'previous_v14')))
DATA_DEFAULT=ROOT/'benchmark_suite'/'data'/'davis2017_rgb_360p_f050'
OUT_DEFAULT=ROOT/'results'/'DAVIS_RGB_360p_FINAL_K12_C32_T3600_P96'
PREPROCESS=ROOT/'experiments'/'10_davis2017_rgb_all9'/'prepare_davis_rgb.py'
BASE_RUNNER=ROOT/'experiments'/'10_davis2017_rgb_all9'/'rgb9.py'
FROZEN_RUNNER=ROOT/'experiments'/'17_frozen_rgb_360p_reuse'/'frozen_benchmark.py'
EIGHT=('blackswan','bmx-trees','breakdance','camel','car-roundabout','drift-chicane','kite-surf','parkour')
RATES=(0.2,0.3,0.4)
CPU_METHODS=('KBR','TNN','SALTS')
SEQUENTIAL_METHODS=('DRO-TFF','HLRTF','LRTFR','OTLRM','S2NTNN')
MAIN_EIGHT='Observed,DRO-TFF,KBR,TNN,SALTS,S2NTNN,Ours-Frozen,GT'


def dataset_images_root(root):
    for p in [root/'JPEGImages'/'480p',root/'DAVIS'/'JPEGImages'/'480p',root/'2017'/'JPEGImages'/'480p',root]:
        if p.is_dir() and any(x.is_dir() for x in p.iterdir()):
            return p
    raise FileNotFoundError('No DAVIS JPEGImages/480p sequence folders under '+str(root))


def scan(root):
    folder=dataset_images_root(root)
    names=[p.name for p in folder.iterdir() if p.is_dir() and any(p.glob('*.jpg'))]
    prioritized=[v for v in EIGHT if v in names]
    return prioritized+sorted(set(names)-set(prioritized))


def choose(all_names,text,num,offset):
    if text.lower()=='all':names=list(all_names)
    else:names=[v.strip() for v in text.split(',') if v.strip()]
    if not names or len(names)!=len(set(names)):
        raise ValueError('Empty or duplicated video list')
    unknown=set(names)-set(all_names)
    if unknown:raise ValueError('Unknown videos in DAVIS JPEGImages/480p: '+', '.join(sorted(unknown)))
    if offset<0 or (num is not None and num<1):raise ValueError('--start-index >= 0, --num-videos >= 1')
    result=names[offset: (offset+num if num is not None else None)]
    if not result:raise ValueError('Selected zero videos. Check --start-index and --num-videos')
    return result


def run(cmd,env=None):
    print('COMMAND:',subprocess.list2cmdline(list(map(str,cmd))),flush=True)
    result=subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env)
    if result.returncode:raise RuntimeError('Command returned '+str(result.returncode)+': '+str(cmd))


def existing_cases(source,video):
    import numpy as np
    # v8 may have indexed cases but not materialized all three RGB channel inputs.
    stems=[frozen.case_id(video,sr) for sr in RATES]
    return all(rgb9.paths(source,c).is_file() and all(rgb9.channel_case(source,c,k).is_file() for k in 'RGB')
               for c in stems)


def setup_matlab():
    if 'MATLAB_BIN' in os.environ:return
    matlab_binary=shutil.which('matlab')
    if matlab_binary:os.environ['MATLAB_BIN']=matlab_binary


def prepare(davis_root,source,data,videos,dry_run=False):
    missing=[v for v in videos if not existing_cases(source,v)]
    print('Prepared v8 videos reused:',[v for v in videos if v not in missing],flush=True)
    if not missing:return
    print('NEW VIDEOS / INCOMPLETE SOURCE:',missing,flush=True)
    cmd=[sys.executable,PREPROCESS,'--davis-root',davis_root,'--out',data,
         '--sequences',','.join(missing),'--height',360,'--width',640,'--frames',50,'--append']
    cmd2=[sys.executable,BASE_RUNNER,'prepare','--data',data,'--output',source,
          '--videos',','.join(missing),'--seed',frozen.SEED,'--rates','0.2,0.3,0.4']
    if dry_run:
        for c in (cmd,cmd2):print('DRY RUN:',subprocess.list2cmdline(list(map(str,c))),flush=True)
        return
    run(cmd);run(cmd2)
    still=[v for v in missing if not existing_cases(source,v)]
    if still:raise RuntimeError('Cases still missing after preprocessing: '+str(still))


def missing_baselines(source,video):
    """Only the eight formal baselines; old Ours is not a formal baseline."""
    needs=[]
    for sr in RATES:
        case=frozen.case_id(video,sr)
        for method in frozen.BASELINES:
            good,_=rgb9.check_rgb_output(rgb9.rgb_out(source,method,case),frozen.SHAPE)
            if not good:needs.append((case,method))
    return needs


def _matlab_task(source,video,sr,method,threads):
    """One case x one CPU baseline; unique RGB outputs and journal per task."""
    assert method in CPU_METHODS
    case=frozen.case_id(video,sr)
    env=os.environ.copy()
    env['KPD_PARALLEL_JOURNAL']=case+'__'+method
    env['KPD_MATLAB_THREADS']=str(threads)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]=str(threads)
    cmd=[sys.executable,BASE_RUNNER,'run','--output',source,'--videos',video,
         '--rates',str(sr),'--methods',method,'--stop-on-error']
    log=source/'parallel_logs_v15'/(case+'__'+method+'.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    print('CPU WORKER START',case,method,'threads',threads,'log',log,flush=True)
    with log.open('a',encoding='utf-8') as fh:
        fh.write('\n=== V15 CPU TASK '+case+' '+method+' ===\n');fh.flush()
        result=subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdout=fh,stderr=subprocess.STDOUT)
    print('CPU WORKER DONE',case,method,'rc',result.returncode,flush=True)
    return case,method,result.returncode,str(log)


def _plan(source,videos):
    missing={v:missing_baselines(source,v) for v in videos}
    cpu=[];sequential=[]
    for v in videos:
        for case,m in missing[v]:
            if m in CPU_METHODS:
                sr=int(case.split('__sr')[1][:3])/100.
                cpu.append((v,sr,m))
            elif m in SEQUENTIAL_METHODS:
                sequential.append((case,m))
            else:
                raise AssertionError('Unexpected baseline method: '+m)
    return cpu,sequential


def _sequential_lane(source,v12,v14,out,videos,args,needed_py):
    """Exactly one baseline Python subprocess or frozen CNN process at a time."""
    if args.baseline_mode!='skip':
        for video in videos:
            if not any(case.startswith(video+'__') for case,_ in needed_py):
                print('GPU/PY SKIP valid baselines',video,flush=True)
                continue
            cmd=[sys.executable,BASE_RUNNER,'run','--output',source,'--videos',video,
                 '--rates','0.2,0.3,0.4','--methods',','.join(SEQUENTIAL_METHODS),
                 '--stop-on-error']
            print('GPU/PY ONE-WORKER START',video,flush=True)
            run(cmd)
            print('GPU/PY ONE-WORKER DONE',video,flush=True)
    print('FROZEN Ours ONE-WORKER START (may reuse v12/v14 outputs)',flush=True)
    run(frozen_args(source,v12,v14,out,videos,'fit',args))
    print('FROZEN Ours ONE-WORKER DONE',flush=True)


def run_heterogeneous(source,v12,v14,out,videos,args):
    """Start CPU pool *before* sequential lane, wait for BOTH before evaluation.

    Separate methods own separate output paths. CPU workers have unique logs
    and journals. No duplicate (case,method) jobs are submitted.
    """
    setup_matlab()
    cpu,py=_plan(source,videos) if args.baseline_mode!='skip' else ([],[])
    print('V15 SCHEDULE CPU tasks',len(cpu),'CPU workers',args.jobs,
          'threads/worker',args.matlab_threads,'sequential Python cases',len(py),
          'Ours-Frozen single lane',flush=True)
    if args.dry_run:
        for v,sr,m in cpu:print('DRY RUN CPU',v,sr,m,flush=True)
        for case,m in py:print('DRY RUN SEQUENTIAL',case,m,flush=True)
        print('DRY RUN Ours-Frozen, then unified evaluation; no solver launched.',flush=True)
        return
    errors=[]
    cpu_pool_size=max(1,min(args.jobs,len(cpu)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_pool_size,thread_name_prefix='matlab_cpu') as pool:
        futures=[pool.submit(_matlab_task,source,v,sr,m,args.matlab_threads) for v,sr,m in cpu]
        # CPU subprocesses are started BEFORE entering sequential Python/CUDA lane.
        # The context manager drains CPU workers even if the sequential lane fails.
        try:
            _sequential_lane(source,v12,v14,out,videos,args,py)
        except Exception as exc:
            errors.append(('sequential_lane',str(exc)))
            print('GPU/PY SEQUENTIAL ERROR',repr(exc),flush=True)
        for f in concurrent.futures.as_completed(futures):
            try:
                case,m,rc,log=f.result()
                if rc:errors.append((case,m,log))
            except Exception as exc:
                errors.append(('cpu_worker',repr(exc)))
    if errors:
        raise RuntimeError('Some jobs failed; valid finished RGB outputs are retained. '+repr(errors))
    print('BOTH QUEUES FINISHED. UNIFIED EVALUATION + FIGURES BEGIN.',flush=True)
    run(frozen_args(source,v12,v14,out,videos,'report',args))


def frozen_args(source,v12,v14,out,videos,subcommand,a):
    cmd=[sys.executable,FROZEN_RUNNER,
         '--source',source,'--v12-results',v12,'--v14-results',v14,'--output',out,
         '--videos',','.join(videos),'--rates','0.2,0.3,0.4',
         '--frame',str(a.frame),'--roi',a.roi,'--figure-variant',a.figure_variant,
         '--main-columns',a.main_columns,'--split-role',a.split_role]
    if a.trust_v12_source:cmd.append('--trust-v12-source')
    if a.require_complete:cmd.append('--require-complete')
    if subcommand=='fit':cmd.append('--fit-only')
    if subcommand=='report':cmd.append('--evaluate-only')
    if subcommand=='visualize':cmd.append('--figures-only')
    if subcommand in ('run','fit'):
        cmd+=['--device',a.device]
        if a.dry_run:cmd.append('--dry-run')
        if a.stop_on_error:cmd.append('--stop-on-error')
    return cmd


def progress(source,v12,v14,out,videos):
    counts={'formal':0,'original':0,'expected':len(videos)*len(RATES)*9}
    for v in videos:
        for sr in RATES:
            case=frozen.case_id(v,sr)
            methods=[*frozen.FORMAL,'Ours-Original']
            ready=[]
            for m in methods:
                if m=='Ours-Frozen':
                    p=frozen.frozen_out(out,case)
                    q=frozen.v12_out(v12,case)
                    q14=frozen.frozen_out(v14,case)
                    present=any(p0.is_file() and p0.stat().st_size>200 for p0 in (p,q,q14))
                else:
                    p=frozen.original_out(source,m,case)
                    present=p.is_file() and p.stat().st_size>200
                if present:ready.append(m)
            n=sum(x in frozen.FORMAL for x in ready)
            counts['formal']+=n
            counts['original']+=int('Ours-Original' in ready)
            print(f'{v:20s} SR={sr:.1f} | nine formal {n}/9 | original={"Ours-Original" in ready} | missing',
                  ', '.join(x for x in frozen.FORMAL if x not in ready),flush=True)
    print('PROGRESS',counts,'(file existence only; run report for full numerical validation)',flush=True)


def write_selection(out,root,source,videos,args):
    d={'timestamp':datetime.now().isoformat(timespec='seconds'),'videos':videos,
       'davis_root':str(root),'v8_source':str(source),'rates':list(RATES),
       'dimensions':[360,640,50,3],'seed':frozen.SEED,
       'frozen_config':{'rank':12,'hidden':32,'steps':3600,'patch':[16,96,96],
                        'kernel':[3,3,3],'crossfit_folds':5},
       'baseline_mode':args.baseline_mode,'development_sequence':'blackswan',
       'selection_order':'original-eight-first-then-alphabetical'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'SELECTED_VIDEOS_LAST.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'SELECTION_HISTORY.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description=__doc__)
    p.add_argument('action',choices=['list','prepare','run','report','visualize','video','progress'])
    p.add_argument('--davis-root',type=Path,default=DATASET_DEFAULT)
    p.add_argument('--source',type=Path,default=V8_DEFAULT,help='Existing v8 cases/results folder; NEW cases append without overwriting old results')
    p.add_argument('--data',type=Path,default=DATA_DEFAULT,help='Preprocessed RGB numpy data for previously unprepared sequences')
    p.add_argument('--v12-results',type=Path,default=V12_DEFAULT,help='Optional old v12 C2 read-only reuse folder')
    p.add_argument('--v14-results',type=Path,default=V14_DEFAULT,help='Completed v14 frozen results; read-only reuse, no copying')
    p.add_argument('--output',type=Path,default=OUT_DEFAULT,help='All new Ours-Frozen outputs, reports, figures, videos')
    p.add_argument('--videos',default='blackswan',help='blackswan,camel OR all (all discovered DAVIS videos)')
    p.add_argument('--num-videos',type=int,default=None,help='Choose N videos from sequence list')
    p.add_argument('--start-index',type=int,default=0,help='Start position (0-based) in deterministic discovered sequence list')
    p.add_argument('--jobs',type=int,default=4,help='Concurrent MATLAB CPU processes, running alongside the single Python/GPU lane')
    p.add_argument('--matlab-threads',type=int,default=4,help='Internal CPU threads per MATLAB subprocess when jobs>1')
    p.add_argument('--baseline-mode',choices=['missing','skip'],default='missing',
                   help='missing: reuse eight formal baseline methods, run only missing; skip: never run baselines')
    p.add_argument('--skip-preprocess',action='store_true',help='Require all selected 360p RGB cases to already exist')
    p.add_argument('--prepare-only',action='store_true',help='Alias for action prepare when action=run')
    p.add_argument('--frame',type=int,default=20,help='0-based figure frame, -1 for midpoint')
    p.add_argument('--roi',default='auto',help='auto or x,y,width,height')
    p.add_argument('--figure-variant',choices=['main','supplement','both'],default='both')
    p.add_argument('--main-columns',default=MAIN_EIGHT,help='Eight paper columns; no old Ours required')
    p.add_argument('--split-role',choices=['development','exploratory','test','legacy'],default='test')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='cuda')
    p.add_argument('--require-complete',action='store_true',help='Fail if any of the formal 9 result files are missing')
    p.add_argument('--stop-on-error',action='store_true',default=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--trust-v12-source',action='store_true',help='Only if old v12 source moved; ensure hashes match independently')
    p.add_argument('--fps',type=float,default=10.,help='Playback fps for supplementary MP4')
    p.add_argument('--video-layout',choices=['main','supplement'],default='main')
    p.add_argument('--video-cell-width',type=int,default=320)
    p.add_argument('--video-start',type=int,default=0)
    p.add_argument('--video-end',type=int,default=50,help='exclusive video end frame')
    args=p.parse_args(argv)
    if not 1<=args.jobs<=8:p.error('--jobs must be 1..8; recommended 2')
    if not 1<=args.matlab_threads<=32:p.error('--matlab-threads must be 1..32; recommended 4')
    print('V15 HETEROGENEOUS: CPU MATLAB up to',args.jobs,'x',args.matlab_threads,
          'threads + ONE sequential Python/GPU lane in parallel (5080 16GB).',flush=True)
    root=args.davis_root.expanduser().resolve();source=args.source.expanduser().resolve()
    data=args.data.expanduser().resolve();v12=args.v12_results.expanduser().resolve()
    v14=args.v14_results.expanduser().resolve();out=args.output.expanduser().resolve()
    frozen.V14_REUSE=v14
    all_names=scan(root)
    if args.action=='list':
        print('DAVIS SOURCE:',dataset_images_root(root),f'({len(all_names)} sequence folders)')
        for i,n in enumerate(all_names):print(f'{i:3d}  {n}')
        return 0
    try:videos=choose(all_names,args.videos,args.num_videos,args.start_index)
    except ValueError as e:p.error(str(e))
    print('SELECTED VIDEOS',len(videos),videos,flush=True)
    print('SOURCE',source,'OUTPUT',out,flush=True)
    print('FIXED Ours: K12 C32, 3x3x3, 3600 steps, patch 16x96x96, SR={0.2,0.3,0.4}, seed 20261014',flush=True)
    if not args.dry_run:write_selection(out,root,source,videos,args)
    if args.action in ('run','prepare'):
        if args.skip_preprocess:
            bad=[v for v in videos if not existing_cases(source,v)]
            if bad:raise RuntimeError('Prepared RGB cases missing: '+str(bad)+'; remove --skip-preprocess')
        else:prepare(root,source,data,videos,args.dry_run)
    if args.action=='prepare' or args.prepare_only:return 0
    if args.action=='run' and args.dry_run:
        print('DRY RUN: eight v8 baselines only; no Ours-Original training.',flush=True)
        run_heterogeneous(source,v12,v14,out,videos,args)
        print('         Selected videos:',videos,'No solver launched.',flush=True)
        return 0
    if args.action=='progress':
        progress(source,v12,v14,out,videos);return 0
    if args.action=='run':
        run_heterogeneous(source,v12,v14,out,videos,args)
    elif args.action in ('report','visualize'):
        run(frozen_args(source,v12,v14,out,videos,args.action,args))
    elif args.action=='video':
        from video_supplement import create_mp4
        for v in videos:
            for sr in RATES:
                create_mp4(source,v12,out,v,sr,args.video_layout,args.main_columns,
                           args.roi,args.fps,args.video_cell_width,args.video_start,args.video_end,
                           args.require_complete)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,FileNotFoundError,ValueError) as e:
        print('ERROR:',e,file=sys.stderr,flush=True)
        raise SystemExit(2)
