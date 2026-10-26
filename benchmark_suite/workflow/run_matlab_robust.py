from __future__ import annotations
import argparse,csv,subprocess,shutil,time,sys,os
from pathlib import Path
from datetime import datetime
from common import ROOT,load_experiment,case_dir,recon_dir,logs_dir,sync_profile
from output_validation import validate_output,quarantine_output

def qmat(s): return str(s).replace("'","''")
def rc_hex(rc): return f"0x{(int(rc)&0xffffffff):08X}"
def append_csv(path,row,header):
    new=not path.exists(); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); 
        if new: w.writerow(header)
        w.writerow(row)
def run_wait(cmd,log_path,heartbeat,label):
    log_path.parent.mkdir(parents=True,exist_ok=True); t0=time.perf_counter()
    with log_path.open('w',encoding='utf-8',errors='replace') as log:
        p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
        try:
            while True:
                rc=p.poll()
                if rc is not None: return rc,time.perf_counter()-t0
                time.sleep(min(heartbeat,10))
                elapsed=time.perf_counter()-t0
                # Print heartbeat only at requested intervals.
                if int(elapsed)//heartbeat != int(max(0,elapsed-10))//heartbeat:
                    print(f"[MATLAB heartbeat] {label} elapsed={elapsed/60:.1f} min",flush=True)
        except KeyboardInterrupt:
            try: p.terminate(); time.sleep(2)
            except Exception: pass
            try:
                if p.poll() is None: p.kill()
            except Exception: pass
            raise

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--methods',nargs='+'); a=ap.parse_args()
    sync_profile(); exp=load_experiment(); cfg=exp.get('execution',{}); methods=a.methods or exp['methods']['matlab']; cases=sorted(case_dir(a.dataset).glob('*.mat'))
    if not cases: raise SystemExit('No cases. Run prepare_cases.py first.')
    matlab=shutil.which('matlab');
    if not matlab: raise SystemExit('MATLAB not found in PATH')
    max_attempts=int(cfg.get('matlab_max_attempts',3)); retry=float(cfg.get('retry_delay_seconds',5)); heartbeat=int(cfg.get('heartbeat_seconds',300)); accept_nonzero=bool(cfg.get('accept_valid_output_after_nonzero_exit',True))
    ldir=logs_dir(a.dataset); attempts_csv=ldir/'matlab_attempts.csv'; failures=[]; total=len(methods)*len(cases); job=0
    header=['timestamp','job','method','case','attempt','returncode','returncode_hex','wall_seconds','validation','status','output','log']
    for method in methods:
        outdir=recon_dir(a.dataset,method); outdir.mkdir(parents=True,exist_ok=True)
        for case in cases:
            job+=1; out=outdir/f'{method}__{case.stem}.mat'
            ok,reason=validate_output(case,out)
            if ok and not a.overwrite:
                print(f"[MATLAB {job}/{total}] SKIP {method} :: {case.stem}",flush=True); continue
            if out.exists():
                q=quarantine_output(a.dataset,method,out,reason); print(f"[MATLAB] quarantined invalid output: {q}",flush=True)
            success=False
            for attempt in range(1,max_attempts+1):
                print(f"[MATLAB {job}/{total}] RUN {method} :: {case.stem} | attempt {attempt}/{max_attempts}",flush=True)
                log=ldir/'matlab_tasks'/method/f'{case.stem}__attempt{attempt:02d}.log'
                expr=f"addpath('workflow'); run_matlab_task('{qmat(a.dataset)}','{qmat(method)}','{qmat(case.stem)}',{str(bool(a.overwrite)).lower()})"
                rc,wall=run_wait([matlab,'-batch',expr],log,heartbeat,f'{method}::{case.stem}')
                ok,reason=validate_output(case,out)
                if ok and (rc==0 or accept_nonzero): status='DONE' if rc==0 else 'VALID_OUTPUT_AFTER_CRASH'; success=True
                else: status='FAILED_ATTEMPT'
                append_csv(attempts_csv,[datetime.now().isoformat(timespec='seconds'),job,method,case.stem,attempt,rc,rc_hex(rc),f'{wall:.3f}',reason,status,str(out),str(log)],header)
                print(f"[MATLAB] {status} rc={rc_hex(rc)} wall={wall:.1f}s validation={reason}",flush=True)
                if success: break
                if out.exists(): quarantine_output(a.dataset,method,out,reason)
                partial=outdir/f'{method}__{case.stem}.partial.mat'
                if partial.exists():
                    try: partial.unlink()
                    except Exception: pass
                if attempt<max_attempts: time.sleep(retry)
            if not success:
                failures.append((method,case.stem)); print(f"[MATLAB FINAL FAILURE] {method} :: {case.stem}; continuing.",flush=True)
    failfile=ldir/'matlab_final_failures.csv'
    with failfile.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['method','case']); w.writerows(failures)
    print(f"MATLAB robust lane finished. failures={len(failures)}",flush=True)
    # Return zero so the other lane/evaluation/audit can continue; audit is authoritative.
if __name__=='__main__': main()
