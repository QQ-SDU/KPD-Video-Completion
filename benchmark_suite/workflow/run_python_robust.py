from __future__ import annotations
import argparse,csv,subprocess,sys,time,re,os
from datetime import datetime
from pathlib import Path
from common import ROOT,load_method_defaults,load_experiment,case_dir,recon_dir,logs_dir,sync_profile
from output_validation import validate_output,quarantine_output
METHODS=['S2NTNN','HLRTF','LRTFR','DRO-TFF','OTLRM','Ours']
def parse_seed(stem):
    m=re.search(r'__seed(\d+)$',stem); return int(m.group(1)) if m else 1
def build_cmd(method,case,out):
    C=load_method_defaults(); seed=parse_seed(case.stem)
    if method=='S2NTNN':
        c=C[method]; return [sys.executable,str(ROOT/'Methods/S2NTNN/benchmark_runner_s2ntnn.py'),'--case',str(case),'--out',str(out),'--seed',str(seed),'--max-iter',str(c['max_iter']),'--lr',str(c['lr']),'--lambda-nuc',str(c['lambda_nuc']),'--n4-mult',str(c['n4_mult'])]
    if method=='HLRTF':
        c=C[method]; return [sys.executable,str(ROOT/'Methods/HLRTF/benchmark_runner_hlrtf.py'),'--case',str(case),'--out',str(out),'--seed',str(seed),'--max-iter',str(c['max_iter']),'--lr',str(c['lr']),'--gamma',str(c['gamma']),'--shrink',str(c['shrink'])]
    if method=='LRTFR':
        c=C[method]; return [sys.executable,str(ROOT/'Methods/LRTFR/run_same_mask.py'),'--case',str(case),'--out',str(out),'--seed',str(seed),'--max-iter',str(c['max_iter']),'--lr',str(c['lr']),'--weight-decay',str(c['weight_decay']),'--omega',str(c['omega']),'--down',','.join(map(str,c['down']))]
    if method=='DRO-TFF':
        c=C[method]; return [sys.executable,str(ROOT/'Benchmark/adapters/dro_tff/run_drotff_same_mask.py'),'--repo',str(ROOT/'Methods/DRO-TFF'),'--case',str(case),'--out',str(out),'--seed',str(seed),'--max-iter',str(c['max_iter']),'--lr',str(c['lr']),'--beta',str(c['beta']),'--omega',str(c['omega']),'--layers',str(c['layers']),'--gamma',str(c['gamma'])]
    if method=='OTLRM':
        c=C[method]; return [sys.executable,str(ROOT/'Methods/OTLRM/run_same_mask.py'),'--case',str(case),'--out',str(out),'--seed',str(seed),'--max-iter',str(c['max_iter']),'--lr',str(c['lr']),'--shrink',str(c['shrink']),'--gamma',str(c['gamma']),'--weight-decay',str(c['weight_decay'])]
    if method=='Ours':
        c=C[method]; return [sys.executable,str(ROOT/'Methods/Ours/run_same_mask.py'),'--case',str(case),'--out',str(out),'--seed',str(seed),'--rank',str(c['rank']),'--masked-sweeps',str(c['masked_sweeps']),'--lam-space',str(c['lambda_s']),'--lam-time',str(c['lambda_t']),'--ridge',str(c['lambda_0']),'--crossfit-folds',str(c['crossfit_folds']),'--hidden',str(c['hidden']),'--max-residual',str(c['max_residual']),'--patch-t',str(c['patch_t']),'--patch-h',str(c['patch_h']),'--patch-w',str(c['patch_w']),'--batch-size',str(c['batch_size']),'--steps',str(c['steps']),'--lr',str(c['lr']),'--smooth-l1-beta',str(c['smooth_l1_beta']),'--detail-weight',str(c['detail_weight']),'--magnitude-scale',str(c['magnitude_scale']),'--magnitude-cap',str(c['magnitude_cap']),'--residual-penalty',str(c['residual_penalty']),'--device','auto']
    raise KeyError(method)
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
            last=0
            while True:
                rc=p.poll()
                if rc is not None: return rc,time.perf_counter()-t0
                time.sleep(min(heartbeat,10)); elapsed=time.perf_counter()-t0
                if elapsed-last>=heartbeat: print(f"[PY heartbeat] {label} elapsed={elapsed/60:.1f} min",flush=True); last=elapsed
        except KeyboardInterrupt:
            try: p.terminate(); time.sleep(2)
            except Exception: pass
            try:
                if p.poll() is None: p.kill()
            except Exception: pass
            raise

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--methods',nargs='+',choices=METHODS); ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args()
    sync_profile(); exp=load_experiment(); cfg=exp.get('execution',{}); methods=a.methods or exp['methods']['python']; cases=sorted(case_dir(a.dataset).glob('*.mat'))
    if not cases: raise SystemExit('No cases. Run prepare first.')
    max_attempts=int(cfg.get('python_max_attempts',2)); retry=float(cfg.get('retry_delay_seconds',5)); heartbeat=int(cfg.get('heartbeat_seconds',300)); accept_nonzero=bool(cfg.get('accept_valid_output_after_nonzero_exit',True))
    ldir=logs_dir(a.dataset); status=ldir/'python_attempts.csv'; header=['timestamp','job','method','case','attempt','returncode','wall_seconds','validation','status','output','log']; failures=[]; total=len(cases)*len(methods); job=0
    for method in methods:
        outdir=recon_dir(a.dataset,method); outdir.mkdir(parents=True,exist_ok=True)
        for case in cases:
            job+=1; out=outdir/f'{method}__{case.stem}.mat'; ok,reason=validate_output(case,out)
            if ok and not a.overwrite: print(f"[PY {job}/{total}] SKIP {method} :: {case.stem}",flush=True); continue
            if out.exists(): print('[PY] quarantined:',quarantine_output(a.dataset,method,out,reason),flush=True)
            success=False
            for attempt in range(1,max_attempts+1):
                partial=outdir/f'{method}__{case.stem}.partial.mat'
                if partial.exists():
                    try: partial.unlink()
                    except Exception: pass
                print(f"[PY {job}/{total}] RUN {method} :: {case.stem} | attempt {attempt}/{max_attempts}",flush=True)
                log=ldir/'python_tasks'/method/f'{case.stem}__attempt{attempt:02d}.log'; rc,wall=run_wait(build_cmd(method,case,partial),log,heartbeat,f'{method}::{case.stem}')
                ok,reason=validate_output(case,partial,check_stale=False)
                if ok and (rc==0 or accept_nonzero):
                    os.replace(partial,out); status_name='DONE' if rc==0 else 'VALID_OUTPUT_AFTER_CRASH'; success=True
                else: status_name='FAILED_ATTEMPT'
                append_csv(status,[datetime.now().isoformat(timespec='seconds'),job,method,case.stem,attempt,rc,f'{wall:.3f}',reason,status_name,str(out),str(log)],header)
                print(f"[PY] {status_name} rc={rc} wall={wall:.1f}s validation={reason}",flush=True)
                if success: break
                if partial.exists():
                    try: partial.unlink()
                    except Exception: pass
                if attempt<max_attempts: time.sleep(retry)
            if not success:
                failures.append((method,case.stem)); print(f"[PY FINAL FAILURE] {method} :: {case.stem}; continuing.",flush=True)
    with (ldir/'python_final_failures.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['method','case']); w.writerows(failures)
    print(f"Python robust lane finished. failures={len(failures)}",flush=True)
if __name__=='__main__': main()
