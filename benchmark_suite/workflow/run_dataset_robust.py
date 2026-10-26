from __future__ import annotations
import argparse,subprocess,sys,threading,os,json
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'workflow'))
from common import sync_profile,profile_id,logs_dir,result_root

def run_checked(cmd,title):
    print('\n'+'='*88); print(title); print('='*88); print('>',' '.join(map(str,cmd)),flush=True); r=subprocess.run(cmd,cwd=str(ROOT));
    if r.returncode: raise SystemExit(r.returncode)
def pump(label,proc,logfile):
    logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('w',encoding='utf-8',errors='replace') as f:
        for line in iter(proc.stdout.readline,''):
            if not line: break
            f.write(line); f.flush(); print(f'[{label}] {line}',end='',flush=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--prepare-force',action='store_true'); a=ap.parse_args(); prof=sync_profile()
    run_checked([sys.executable,str(ROOT/'workflow/verify_package.py')],'1/6 Environment/source check')
    run_checked([sys.executable,str(ROOT/'workflow/dataset_preflight.py'),'--dataset',a.dataset],'2/6 Dataset preflight')
    prep=[sys.executable,str(ROOT/'workflow/prepare_cases.py'),'--dataset',a.dataset] + (['--force'] if a.prepare_force else [])
    run_checked(prep,'3/6 Prepare common cases for all five sampling rates')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); ldir=logs_dir(a.dataset); ldir.mkdir(parents=True,exist_ok=True)
    mcmd=[sys.executable,'-u',str(ROOT/'workflow/run_matlab_robust.py'),'--dataset',a.dataset] + (['--overwrite'] if a.overwrite else [])
    pcmd=[sys.executable,'-u',str(ROOT/'workflow/run_python_robust.py'),'--dataset',a.dataset] + (['--overwrite'] if a.overwrite else [])
    env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'
    print('\n'+'='*88); print('4/6 Launching ONLY two parallel lanes: MATLAB || Python'); print('MATLAB uses a fresh MATLAB process for every task; Python uses a fresh child process for every task.'); print('='*88,flush=True)
    mp=subprocess.Popen(mcmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env,bufsize=1)
    pp=subprocess.Popen(pcmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env,bufsize=1)
    tm=threading.Thread(target=pump,args=('MATLAB',mp,ldir/f'matlab_lane_{stamp}.log'),daemon=True); tp=threading.Thread(target=pump,args=('PY',pp,ldir/f'python_lane_{stamp}.log'),daemon=True); tm.start(); tp.start(); mrc=mp.wait(); prc=pp.wait(); tm.join(); tp.join(); print(f'Lane exits: MATLAB={mrc}, Python={prc}',flush=True)
    print('\n'+'='*88); print('5/6 Common evaluation + reports'); print('='*88)
    subprocess.run([sys.executable,str(ROOT/'workflow/evaluate_dataset.py'),'--dataset',a.dataset],cwd=str(ROOT))
    subprocess.run([sys.executable,str(ROOT/'workflow/build_reports.py'),'--dataset',a.dataset],cwd=str(ROOT))
    subprocess.run([sys.executable,str(ROOT/'workflow/build_paper_summaries.py'),'--dataset',a.dataset],cwd=str(ROOT))
    rdir=result_root(a.dataset); rdir.mkdir(parents=True,exist_ok=True); rec={'timestamp':datetime.now().isoformat(timespec='seconds'),'package_version':'V2.1-5SR-robust-multidataset-reuse-safe','dataset':a.dataset,'profile_id':profile_id(),'matlab_lane_returncode':mrc,'python_lane_returncode':prc};
    with (rdir/'run_history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    print('\n'+'='*88); print('6/6 Final audit'); print('='*88)
    arc=subprocess.run([sys.executable,str(ROOT/'workflow/audit_dataset.py'),'--dataset',a.dataset],cwd=str(ROOT)).returncode
    if arc: raise SystemExit(arc)
if __name__=='__main__': main()
