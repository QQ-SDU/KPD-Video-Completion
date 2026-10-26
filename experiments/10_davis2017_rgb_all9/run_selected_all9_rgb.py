#!/usr/bin/env python3
"""DAVIS RGB all-nine launcher: choose 1/many/all, parallel videos, one seed.

For each video and SR, all nine methods run. Each method internally handles R,G,B
serially with the original 3D algorithm and original frozen parameters.
"""
from __future__ import annotations
import argparse, concurrent.futures, os, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREP=HERE/'prepare_davis_rgb.py'; ENGINE=HERE/'rgb9.py'
TEX=ROOT/'experiments/09_davis2017_all9/make_davis9_tex.py'
VIDEOS=('blackswan','bmx-trees','breakdance','camel','car-roundabout','drift-chicane','kite-surf','parkour')

def names_of(s):
    result=list(VIDEOS) if s.lower()=='all' else [x.strip() for x in s.split(',') if x.strip()]
    if not result or len(set(result))!=len(result):raise ValueError('Empty or duplicate video selection')
    wrong=[x for x in result if x not in VIDEOS]
    if wrong:raise ValueError('Unknown video(s): '+str(wrong))
    return result

def command(cmd):
    print('CMD',' '.join(map(str,cmd)),flush=True)
    return subprocess.run(list(map(str,cmd)),cwd=ROOT).returncode

def run_one(video,a,output):
    cmd=[sys.executable,str(ENGINE),'run','--videos',video,'--output',str(output),'--methods','all']
    if a.overwrite:cmd.append('--overwrite')
    if a.stop_on_error:cmd.append('--stop-on-error')
    if a.dry_run:cmd.append('--dry-run')
    logdir=output/'video_logs';logdir.mkdir(parents=True,exist_ok=True)
    file=logdir/(video+'.log')
    print('RGB START',video,'(9 methods × 3 channels × 3 SR), log:',file,flush=True)
    with file.open('a',encoding='utf-8') as f:
        f.write('\n=== RGB VIDEO '+video+' ===\n');f.flush()
        ret=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print(('RGB SUCCESS' if ret.returncode==0 else 'RGB FAILED'),video,'rc=',ret.returncode,flush=True)
    return video,ret.returncode

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',default='all')
    p.add_argument('--list-videos',action='store_true')
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--davis-root',default=os.environ.get('DAVIS_ROOT'))
    p.add_argument('--data',default=str(ROOT/'benchmark_suite/data/davis2017_rgb_new'))
    p.add_argument('--output',default=str(ROOT/'results/davis2017_rgb_all9_single_seed'))
    p.add_argument('--skip-preprocess',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--stop-on-error',action='store_true')
    p.add_argument('--no-evaluate',action='store_true')
    a=p.parse_args()
    if a.list_videos:print('\n'.join(VIDEOS));return 0
    try:names=names_of(a.videos)
    except ValueError as exc:p.error(str(exc))
    if a.jobs<1:p.error('jobs must be >=1')
    if a.seed!=20261014:p.error('This protocol fixes a single seed: 20261014')
    output=Path(a.output).resolve();data=Path(a.data).resolve()
    # Additional guard even when user overrides paths.
    if 'davis2017_all9_single_seed'==output.name or data.name=='davis2017_new':
        p.error('RGB outputs must be separate from prior grayscale outputs')
    output.mkdir(parents=True,exist_ok=True)
    print(f'RGB VIDEOS: {names}; SEED: {a.seed}; SR: 0.2,0.3,0.4; JOBS: {a.jobs}',flush=True)
    print(f'TASKS: {len(names)*3*9} completed RGB method outputs; up to {len(names)*3*9*3} original-channel algorithm runs',flush=True)
    if a.jobs>1:print('WARNING: each video can launch GPU and MATLAB jobs; GTX 1650: --jobs 1 recommended',flush=True)
    if not a.skip_preprocess:
        if not a.davis_root:p.error('Provide --davis-root or DAVIS_ROOT for preprocessing')
        cmd=[sys.executable,str(PREP),'--davis-root',a.davis_root,'--out',str(data),
             '--sequences',','.join(names),'--append']
        if command(cmd):return 2
    elif not (data/'manifest.csv').exists():p.error('RGB manifest missing; remove --skip-preprocess')
    cmd=[sys.executable,str(ENGINE),'prepare','--data',str(data),'--output',str(output),
         '--videos',','.join(names),'--rates','0.2,0.3,0.4','--seed',a.seed]
    if command(cmd):return 2
    if a.prepare_only:print('PREPARED ONLY, no RGB methods run');return 0
    if a.jobs==1:statuses=[run_one(name,a,output) for name in names]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(a.jobs,len(names))) as pool:
            statuses=list(pool.map(lambda n:run_one(n,a,output),names))
    failures=[name for name,rc in statuses if rc]
    if a.dry_run:
        print('DRY-RUN ONLY: commands in video_logs; no model outputs');return 2 if failures else 0
    if not a.no_evaluate:
        choose='all' if a.videos.strip().lower()=='all' else ','.join(names)
        cmd=[sys.executable,str(ENGINE),'evaluate','--output',str(output),'--videos',choose]
        if failures:cmd.append('--allow-partial')
        rc=command(cmd)
        if not rc and not failures:
            report=output if choose=='all' else output/'reports'/('selected__'+'__'.join(sorted(names)))
            command([sys.executable,str(TEX),'--output',str(report)])
    if failures:
        print('ERROR VIDEOS:',failures,'rerun same command to skip successful per-channel outputs');return 2
    print('ALL NINE RGB METHODS COMPLETE for selected videos');return 0
if __name__=='__main__':sys.exit(main())
