from __future__ import annotations
import argparse,csv,subprocess,sys,time,re
from datetime import datetime
from pathlib import Path
from common import ROOT,load_method_defaults,case_dir,recon_dir,logs_dir,sync_profile
METHODS=["S2NTNN","HLRTF","LRTFR","DRO-TFF","OTLRM","Ours"]

def parse_seed(stem):
    m=re.search(r"__seed(\d+)$",stem); return int(m.group(1)) if m else 1

def build_cmd(method,case,out):
    C=load_method_defaults(); seed=parse_seed(case.stem)
    if method=="S2NTNN":
        c=C[method]; return [sys.executable,str(ROOT/"Methods/S2NTNN/benchmark_runner_s2ntnn.py"),"--case",str(case),"--out",str(out),"--seed",str(seed),"--max-iter",str(c["max_iter"]),"--lr",str(c["lr"]),"--lambda-nuc",str(c["lambda_nuc"]),"--n4-mult",str(c["n4_mult"])]
    if method=="HLRTF":
        c=C[method]; return [sys.executable,str(ROOT/"Methods/HLRTF/benchmark_runner_hlrtf.py"),"--case",str(case),"--out",str(out),"--seed",str(seed),"--max-iter",str(c["max_iter"]),"--lr",str(c["lr"]),"--gamma",str(c["gamma"]),"--shrink",str(c["shrink"])]
    if method=="LRTFR":
        c=C[method]; return [sys.executable,str(ROOT/"Methods/LRTFR/run_same_mask.py"),"--case",str(case),"--out",str(out),"--seed",str(seed),"--max-iter",str(c["max_iter"]),"--lr",str(c["lr"]),"--weight-decay",str(c["weight_decay"]),"--omega",str(c["omega"]),"--down",",".join(map(str,c["down"]))]
    if method=="DRO-TFF":
        c=C[method]; return [sys.executable,str(ROOT/"Benchmark/adapters/dro_tff/run_drotff_same_mask.py"),"--repo",str(ROOT/"Methods/DRO-TFF"),"--case",str(case),"--out",str(out),"--seed",str(seed),"--max-iter",str(c["max_iter"]),"--lr",str(c["lr"]),"--beta",str(c["beta"]),"--omega",str(c["omega"]),"--layers",str(c["layers"]),"--gamma",str(c["gamma"])]
    if method=="OTLRM":
        c=C[method]; return [sys.executable,str(ROOT/"Methods/OTLRM/run_same_mask.py"),"--case",str(case),"--out",str(out),"--seed",str(seed),"--max-iter",str(c["max_iter"]),"--lr",str(c["lr"]),"--shrink",str(c["shrink"]),"--gamma",str(c["gamma"]),"--weight-decay",str(c["weight_decay"])]
    if method=="Ours":
        c=C[method]; return [sys.executable,str(ROOT/"Methods/Ours/run_same_mask.py"),"--case",str(case),"--out",str(out),"--seed",str(seed),"--rank",str(c["rank"]),"--masked-sweeps",str(c["masked_sweeps"]),"--lam-space",str(c["lambda_s"]),"--lam-time",str(c["lambda_t"]),"--ridge",str(c["lambda_0"]),"--crossfit-folds",str(c["crossfit_folds"]),"--hidden",str(c["hidden"]),"--max-residual",str(c["max_residual"]),"--patch-t",str(c["patch_t"]),"--patch-h",str(c["patch_h"]),"--patch-w",str(c["patch_w"]),"--batch-size",str(c["batch_size"]),"--steps",str(c["steps"]),"--lr",str(c["lr"]),"--smooth-l1-beta",str(c["smooth_l1_beta"]),"--detail-weight",str(c["detail_weight"]),"--magnitude-scale",str(c["magnitude_scale"]),"--magnitude-cap",str(c["magnitude_cap"]),"--residual-penalty",str(c["residual_penalty"]),"--device","auto"]
    raise KeyError(method)

def valid(p): return p.exists() and p.stat().st_size>256

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--methods",nargs="+",choices=METHODS,default=METHODS); ap.add_argument("--overwrite",action="store_true"); ap.add_argument("--stop-on-error",action="store_true"); a=ap.parse_args()
    sync_profile(); cases=sorted(case_dir(a.dataset).glob("*.mat"))
    if not cases: raise SystemExit("No cases. Run prepare first.")
    ldir=logs_dir(a.dataset); ldir.mkdir(parents=True,exist_ok=True); status=ldir/"python_status.csv"; new=not status.exists()
    fails=[]; total=len(cases)*len(a.methods); job=0
    with status.open("a",newline="",encoding="utf-8-sig") as fh:
        w=csv.writer(fh)
        if new: w.writerow(["timestamp","method","case","status","returncode","wall_seconds","output"])
        for method in a.methods:
            outdir=recon_dir(a.dataset,method); outdir.mkdir(parents=True,exist_ok=True)
            for case in cases:
                job+=1; out=outdir/f"{method}__{case.stem}.mat"
                if valid(out) and not a.overwrite:
                    print(f"[{job}/{total}] SKIP {method} :: {case.stem}"); w.writerow([datetime.now().isoformat(timespec="seconds"),method,case.stem,"SKIP",0,0,str(out)]); fh.flush(); continue
                print("\n"+"="*78); print(f"[{job}/{total}] RUN {method} :: {case.stem}"); print("="*78)
                t0=time.perf_counter(); r=subprocess.run(build_cmd(method,case,out),cwd=str(ROOT)); wall=time.perf_counter()-t0
                st="DONE" if r.returncode==0 else "FAILED"; w.writerow([datetime.now().isoformat(timespec="seconds"),method,case.stem,st,r.returncode,f"{wall:.3f}",str(out)]); fh.flush()
                if r.returncode:
                    fails.append((method,case.stem,r.returncode))
                    if a.stop_on_error: raise SystemExit(r.returncode)
    if fails: print("[WARN] failures:",fails)
    print("Python suite finished.")

if __name__=="__main__": main()
