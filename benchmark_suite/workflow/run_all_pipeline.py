from __future__ import annotations
import argparse,subprocess,sys,shutil,json
from pathlib import Path
from datetime import datetime
from common import ROOT,enabled_datasets,logs_dir,sync_profile,profile_id

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",action="append"); ap.add_argument("--overwrite",action="store_true"); ap.add_argument("--prepare-force",action="store_true"); a=ap.parse_args()
    prof=sync_profile(); names=a.dataset or [d["name"] for d in enabled_datasets()]
    if not names: raise SystemExit("No enabled datasets.")
    prep=[sys.executable,str(ROOT/"workflow/prepare_cases.py")]
    for n in names: prep += ["--dataset",n]
    if a.prepare_force: prep.append("--force")
    if subprocess.run(prep,cwd=str(ROOT)).returncode: raise SystemExit("Case preparation failed.")
    matlab=shutil.which("matlab")
    if not matlab: raise SystemExit("MATLAB not found in PATH.")
    for dataset in names:
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); ldir=logs_dir(dataset); ldir.mkdir(parents=True,exist_ok=True)
        ml=ldir/f"matlab_{stamp}.log"; pl=ldir/f"python_{stamp}.log"
        print("\n"+"#"*80); print("DATASET:",dataset); print("PROFILE:",prof); print("Launching MATLAB and Python in parallel..."); print("#"*80)
        mcmd=[matlab,"-batch",f"addpath('workflow'); run_matlab_suite('{dataset.replace(chr(39),chr(39)*2)}',{'true' if a.overwrite else 'false'})"]
        pcmd=[sys.executable,str(ROOT/"workflow/run_python_suite.py"),"--dataset",dataset] + (["--overwrite"] if a.overwrite else [])
        mf=ml.open("w",encoding="utf-8",errors="replace"); pf=pl.open("w",encoding="utf-8",errors="replace")
        mp=subprocess.Popen(mcmd,cwd=str(ROOT),stdout=mf,stderr=subprocess.STDOUT); pp=subprocess.Popen(pcmd,cwd=str(ROOT),stdout=pf,stderr=subprocess.STDOUT)
        mrc=mp.wait(); prc=pp.wait(); mf.close(); pf.close()
        print("MATLAB rc=",mrc,"Python rc=",prc); print("Logs:",ml,pl)
        subprocess.run([sys.executable,str(ROOT/"workflow/evaluate_dataset.py"),"--dataset",dataset],cwd=str(ROOT))
        subprocess.run([sys.executable,str(ROOT/"workflow/build_reports.py"),"--dataset",dataset],cwd=str(ROOT))
        rdir=ROOT/"artifacts/results"/dataset/profile_id(); rdir.mkdir(parents=True,exist_ok=True)
        (rdir/"run_manifest.json").write_text(json.dumps({"timestamp":datetime.now().isoformat(timespec="seconds"),"dataset":dataset,"profile_id":profile_id(),"matlab_returncode":mrc,"python_returncode":prc,"matlab_log":str(ml),"python_log":str(pl)},indent=2),encoding="utf-8")
    print("All requested datasets finished.")

if __name__=="__main__": main()
