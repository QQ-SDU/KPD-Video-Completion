#!/usr/bin/env python3
"""Single-command DAVIS 2017 nine-method launcher.

One fixed seed. Select one / multiple / all videos. Each selected video runs all
nine methods at SR=0.2,0.3,0.4. Optional independent-video parallel jobs.
"""
from __future__ import annotations
import argparse, csv, os, subprocess, sys, concurrent.futures
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREPROCESS=ROOT/'experiments/08_davis2017_generalization/prepare_davis.py'
ENGINE=HERE/'davis9.py'
TEX=HERE/'make_davis9_tex.py'
DEFAULTS=('blackswan','bmx-trees','breakdance','camel','car-roundabout','drift-chicane','kite-surf','parkour')


def names_of(s):
    names=list(DEFAULTS) if s.strip().lower()=='all' else [n.strip() for n in s.split(',') if n.strip()]
    if not names or len(set(names))!=len(names):
        raise ValueError('Choose at least one video; duplicates are not allowed')
    unknown=[x for x in names if x not in DEFAULTS]
    if unknown:raise ValueError(f'Unknown video(s): {unknown}. Choices: {", ".join(DEFAULTS)}')
    return names


def call(cmd, *, cwd=None):
    print('CMD',' '.join(map(str,cmd)),flush=True)
    return subprocess.run(list(map(str,cmd)),cwd=cwd).returncode


def launch_video(video, args, output):
    cmd=[sys.executable,str(ENGINE),'run','--output',str(output),'--videos',video,'--methods','all']
    if args.overwrite:cmd+=['--overwrite']
    if args.dry_run:cmd+=['--dry-run']
    if args.stop_on_error:cmd+=['--stop-on-error']
    logs=output/'video_logs';logs.mkdir(exist_ok=True,parents=True)
    logfile=logs/f'{video}.log'
    print('START',video,'all 9 methods, three rates; output:',logfile,flush=True)
    with logfile.open('a',encoding='utf-8') as fh:
        fh.write('\n=== VIDEO '+video+' ===\n');fh.flush()
        result=subprocess.run(cmd,stdout=fh,stderr=subprocess.STDOUT,cwd=ROOT)
    print(('SUCCESS' if result.returncode==0 else 'FAILED'),video,'rc=',result.returncode,'log=',logfile,flush=True)
    return video,result.returncode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',default='all',help='blackswan OR blackswan,camel OR all')
    p.add_argument('--list-videos',action='store_true')
    p.add_argument('--jobs',type=int,default=1,help='number of videos running concurrently (default 1)')
    p.add_argument('--seed',type=int,default=20261014,help='exactly ONE seed for all videos and methods')
    p.add_argument('--davis-root',default=os.environ.get('DAVIS_ROOT'),help='raw DAVIS JPEGImages/480p folder or parent')
    p.add_argument('--data',default=str(ROOT/'benchmark_suite/data/davis2017_new'),help='processed HWT arrays')
    p.add_argument('--output',default=str(ROOT/'results/davis2017_all9_single_seed'),help='one shared result directory')
    p.add_argument('--skip-preprocess',action='store_true',help='use existing processed manifest; DAVIS_ROOT not needed')
    p.add_argument('--prepare-only',action='store_true',help='create / append case files and stop before nine methods')
    p.add_argument('--dry-run',action='store_true',help='print all nine model commands; no model executed')
    p.add_argument('--overwrite',action='store_true',help='rerun already successful models in selected videos')
    p.add_argument('--stop-on-error',action='store_true',help='stop the current video at its first failed method')
    p.add_argument('--no-evaluate',action='store_true',help='do not produce a selected-subset report')
    args=p.parse_args()
    if args.list_videos:
        print('\n'.join(DEFAULTS));return 0
    try: names=names_of(args.videos)
    except ValueError as e:p.error(str(e))
    if args.jobs<1:p.error('--jobs must be >=1')
    if args.seed<0 or args.seed>2**32-1:p.error('--seed must fit in unsigned 32-bit integer')
    rates=(0.2,0.3,0.4)  # All three paper sampling rates for EVERY chosen video.
    output=Path(args.output).expanduser().resolve();data=Path(args.data).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    print('VIDEOS:',','.join(names),'SEED:',args.seed,'SR: 0.2,0.3,0.4','JOBS:',args.jobs,flush=True)
    print('EXPERIMENT SIZE:',len(names)*len(rates),'cases x 9 methods =',len(names)*len(rates)*9,'model runs',flush=True)
    if args.jobs>1:
        print('WARNING: concurrent videos launch concurrent GPU models and MATLAB sessions. Confirm available VRAM/RAM/licenses.',flush=True)
    if not args.skip_preprocess:
        if not args.davis_root:p.error('DAVIS_ROOT / --davis-root is required for preprocessing')
        cmd=[sys.executable,str(PREPROCESS),'--davis-root',args.davis_root,'--out',data,
             '--sequences',','.join(names),'--append']
        if call(cmd,cwd=ROOT):return 2
    else:
        if not (data/'manifest.csv').exists():p.error('Processed manifest not found; remove --skip-preprocess')
    # Preparing cases is always sequential, preventing index races with other videos.
    cmd=[sys.executable,str(ENGINE),'prepare','--data',data,'--output',output,
         '--videos',','.join(names),'--rates','0.2,0.3,0.4','--seed',args.seed]
    if call(cmd,cwd=ROOT):return 2
    if args.prepare_only:
        print('PREPARED ONLY; no models run.');return 0
    if args.jobs==1:
        statuses=[launch_video(video,args,output) for video in names]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs,len(names))) as pool:
            statuses=list(pool.map(lambda video: launch_video(video,args,output),names))
    failed=[video for video,rc in statuses if rc!=0]
    if args.dry_run:
        print('DRY RUN COMPLETE. No model outputs were produced; see video_logs/*.log.');return 2 if failed else 0
    if not args.no_evaluate:
        # Explicit selection puts reports in reports/selected__... and preserves
        # the all-video official summary for later full runs.
        chosen='all' if args.videos.strip().lower()=='all' else ','.join(names)
        cmd=[sys.executable,str(ENGINE),'evaluate','--output',output,'--videos',chosen]
        if failed:cmd+=['--allow-partial']
        rc=call(cmd,cwd=ROOT)
        if rc==0 and not failed:
            report=(output if chosen=='all' else output/'reports'/('selected__'+'__'.join(sorted(names))))
            call([sys.executable,str(TEX),'--output',report],cwd=ROOT)
            print('SELECTED REPORT:',report,flush=True)
        elif rc!=0:
            print('EVALUATION INCOMPLETE: check missing_or_failed.csv and task logs.',flush=True)
    if failed:
        print('FAILURES:',','.join(failed),'-- rerun the same command; valid outputs will be skipped.',flush=True)
        return 2
    print('ALL NINE METHODS FINISHED for selected videos; outputs retained for future incremental runs.',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
