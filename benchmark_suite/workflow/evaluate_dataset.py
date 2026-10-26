from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Benchmark"))
from benchmark.evaluate_output import load_case,load_recon,align_shape
from benchmark.metrics import psnr,temporal_gradient_mae
from common import load_experiment,case_dir,recon_dir,metrics_dir,profile_id,sync_profile

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); a=ap.parse_args(); sync_profile()
    exp=load_experiment(); methods=exp["methods"]["all"]; mdir=metrics_dir(a.dataset); mdir.mkdir(parents=True,exist_ok=True)
    runs=[]; frames=[]; missing=[]
    for case in sorted(case_dir(a.dataset).glob("*.npz")):
        meta=json.loads(Path(str(case.with_suffix(""))+".json").read_text(encoding="utf-8")); gt,obs,mask=load_case(case); miss=~mask
        for method in methods:
            out=recon_dir(a.dataset,method)/f"{method}__{case.stem}.mat"
            if not out.exists(): missing.append({"dataset":a.dataset,"method":method,"case":case.stem,"output":str(out)}); continue
            try:
                rec,runtime=load_recon(out); rec=align_shape(rec,gt.shape)
                if exp["evaluation"]["clip_to_0_1"]: rec=np.clip(rec,0,1)
                dc_mae=float(np.abs(rec[mask]-obs[mask]).mean()) if np.any(mask) else 0.0
                dc_max=float(np.abs(rec[mask]-obs[mask]).max()) if np.any(mask) else 0.0
                if exp["evaluation"]["force_data_consistency_for_all_methods"]:
                    rec=rec.copy(); rec[mask]=obs[mask]
                ss=[structural_similarity(gt[:,:,k],rec[:,:,k],data_range=1.0) for k in range(gt.shape[2])]
                row={"dataset":a.dataset,"method":method,"case":case.stem,"video":meta["video_name"],"sampling_rate":meta["sampling_rate"],"seed":meta["seed"],
                     "height":gt.shape[0],"width":gt.shape[1],"frames":gt.shape[2],"psnr_full":psnr(gt,rec,1.0),"ssim_frame_mean":float(np.mean(ss)),"ssim_frame_std":float(np.std(ss)),
                     "psnr_missing":psnr(gt,rec,1.0,mask=miss),"mae_missing":float(np.abs(gt[miss]-rec[miss]).mean()) if np.any(miss) else np.nan,
                     "temporal_gradient_mae_missing":temporal_gradient_mae(gt,rec,mask=miss),"observed_dc_mae_before_force":dc_mae,"observed_dc_max_before_force":dc_max,
                     "runtime_seconds":runtime,"case_file":str(case),"reconstruction_file":str(out),"profile_id":profile_id()}
                runs.append(row)
                for k in range(gt.shape[2]):
                    mk=~mask[:,:,k]
                    frames.append({"dataset":a.dataset,"method":method,"case":case.stem,"video":meta["video_name"],"sampling_rate":meta["sampling_rate"],"seed":meta["seed"],"frame":k,
                                   "psnr_full":psnr(gt[:,:,k],rec[:,:,k],1.0),"ssim":float(structural_similarity(gt[:,:,k],rec[:,:,k],data_range=1.0)),
                                   "mae_missing":float(np.abs(gt[:,:,k][mk]-rec[:,:,k][mk]).mean()) if np.any(mk) else np.nan})
                print(f"[OK] {method:8s} {case.stem}: {row['psnr_full']:.3f} / {row['ssim_frame_mean']:.4f}")
            except Exception as e:
                missing.append({"dataset":a.dataset,"method":method,"case":case.stem,"output":str(out),"error":repr(e)})
    def save(path,rows):
        if not rows: path.write_text("",encoding="utf-8"); return
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields: fields.append(k)
        with path.open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    save(mdir/"per_run_metrics.csv",runs); save(mdir/"per_frame_metrics.csv",frames); save(mdir/"missing_or_failed_outputs.csv",missing)
    print("Saved:",mdir)

if __name__=="__main__": main()
