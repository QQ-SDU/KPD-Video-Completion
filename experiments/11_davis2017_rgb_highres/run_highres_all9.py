#!/usr/bin/env python3
"""Run unchanged 9 baseline algorithms on higher resolution RGB DAVIS videos.

Preserves frozen seed, SRs, 9-method all-or-nothing selection, and independent RGB
channels.  Separate data/results folders for each spatial/temporal setting.
Use --size native and optionally --all-frames for original DAVIS JPEG video.
"""
from __future__ import annotations
import argparse, os, sys, subprocess, concurrent.futures
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREP=ROOT/'experiments/10_davis2017_rgb_all9/prepare_davis_rgb.py'
ENGINE=ROOT/'experiments/10_davis2017_rgb_all9/rgb9.py'
VIDEOS=('blackswan','bmx-trees','breakdance','camel','car-roundabout','drift-chicane','kite-surf','parkour')
SIZES={'144p':(144,256),'360p':(360,640),'480p':(480,854),'native':None}
SEED=20261014

def names(s):
    a=list(VIDEOS) if s.lower()=='all' else [k.strip() for k in s.split(',') if k.strip()]
    if not a or len(set(a))!=len(a) or any(x not in VIDEOS for x in a):
        raise ValueError('Invalid --videos: '+s)
    return a

def command(cmd, log=None, capture=False):
    cmd=list(map(str,cmd))
    print('CMD',subprocess.list2cmdline(cmd),flush=True)
    if capture:
        return 0
    if log is None:return subprocess.run(cmd,cwd=ROOT).returncode
    with Path(log).open('a',encoding='utf-8') as f:
        f.write('CMD '+subprocess.list2cmdline(cmd)+'\n');f.flush()
        return subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',default='blackswan',help='one name, comma-separated names or all')
    p.add_argument('--size',choices=SIZES,default='360p',help='native = original DAVIS JPEG spatial resolution')
    p.add_argument('--frames',type=int,default=50,help='number of first frames per video')
    p.add_argument('--all-frames',action='store_true',help='use full original clip, no temporal downsampling')
    p.add_argument('--jobs',type=int,default=1,help='simultaneous VIDEOS (recommend 1 for 9-method RGB benchmark)')
    p.add_argument('--davis-root',default=os.environ.get('DAVIS_ROOT'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--skip-preprocess',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stop-on-error',action='store_true')
    p.add_argument('--overwrite',action='store_true',help='force recomputation. Avoid for normal resuming.')
    p.add_argument('--no-evaluate',action='store_true')
    p.add_argument('--check-gpu',action='store_true',help='run CUDA / Conv3D compatibility probe and exit')
    a=p.parse_args(argv)
    if a.check_gpu:
        return subprocess.call([sys.executable,str(HERE/'check_rtx5080.py')],cwd=ROOT)
    try:chosen=names(a.videos)
    except ValueError as e:p.error(str(e))
    if a.jobs<1:p.error('--jobs must be >=1')
    if a.frames<1:p.error('--frames must be positive')
    if a.size=='native' and a.jobs>1:
        print('WARNING: native RGB videos are memory-intensive. Prefer --jobs 1.',flush=True)
    if a.all_frames and a.size!='native':
        print('WARNING: all original frames but spatially resized to',a.size,flush=True)
    tag=f'{a.size}_'+('allframes' if a.all_frames else f'f{a.frames:03d}')
    data=ROOT/'benchmark_suite'/'data'/('davis2017_rgb_'+tag)
    output=ROOT/'results'/('davis2017_rgb_all9_seed'+str(SEED)+'_'+tag)
    output.mkdir(parents=True,exist_ok=True)
    print('RGB HIGH-RES | RESOLUTION',a.size,'| FRAMES',('all' if a.all_frames else a.frames),flush=True)
    print('VIDEOS',chosen,'| SEED',SEED,'| SR 0.2,0.3,0.4 | methods 9 | jobs',a.jobs,flush=True)
    print('DATA DIR',data,'\nRESULT DIR',output,flush=True)
    print('PLANNED:',len(chosen)*3*9,'RGB results / up to',len(chosen)*3*9*3,'channel fits',flush=True)
    print('WARNING: native-resolution processing for all 9 frozen baselines has NOT been validated on RTX 5080.',flush=True)
    if not a.skip_preprocess:
        if not a.davis_root:p.error('Specify --davis-root or environment DAVIS_ROOT')
        cmd=[sys.executable,str(PREP),'--davis-root',a.davis_root,'--out',data,
             '--sequences',','.join(chosen),'--append','--frames',a.frames]
        if a.size=='native':cmd.append('--native-resolution')
        else:
            hh,ww=SIZES[a.size];cmd+=['--height',hh,'--width',ww]
        if a.all_frames:cmd.append('--all-frames')
        ret=command(cmd)
        if ret:return ret
    elif not (data/'manifest.csv').exists():p.error('Missing '+str(data/'manifest.csv')+'; remove --skip-preprocess')
    cmd=[sys.executable,str(ENGINE),'prepare','--data',data,'--output',output,
         '--videos',','.join(chosen),'--rates','0.2,0.3,0.4','--seed',SEED]
    ret=command(cmd)
    if ret:return ret
    if a.prepare_only:return 0
    if a.dry_run:
        for video in chosen:
            cmd=[sys.executable,str(ENGINE),'run','--output',output,'--videos',video,'--methods','all','--dry-run']
            if a.stop_on_error:cmd.append('--stop-on-error')
            ret=command(cmd)
            if ret:return ret
        print('DRY RUN complete; no model trained.',flush=True)
        return 0
    def run_one(video):
        cmd=[sys.executable,str(ENGINE),'run','--videos',video,'--output',output,'--methods','all']
        if a.overwrite:cmd.append('--overwrite')
        if a.stop_on_error:cmd.append('--stop-on-error')
        folder=output/'video_logs';folder.mkdir(exist_ok=True)
        log=folder/(video+'.log')
        print('START',video,'see',log,flush=True)
        rc=command(cmd,log=log)
        print('DONE' if rc==0 else 'FAILED',video,'return code',rc,flush=True)
        return (video,rc)
    if a.jobs==1: results=[run_one(v) for v in chosen]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            results=list(pool.map(run_one,chosen))
    errors=[v for v,rc in results if rc]
    if not a.no_evaluate:
        cmd=[sys.executable,str(ENGINE),'evaluate','--output',output,'--videos',','.join(chosen)]
        if errors:cmd.append('--allow-partial')
        rc=command(cmd)
        if rc and not errors:return rc
    if errors:
        print('FAILED VIDEOS:',errors,'| Rerun identical command to resume.',flush=True)
        return 2
    print('DONE ALL SELECTED RGB RUNS. Independent output:',output,flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
