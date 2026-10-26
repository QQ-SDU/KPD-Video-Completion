from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'workflow'))
from common import load_experiment,get_dataset,count_enabled_sources,case_dir,recon_dir,metrics_dir,result_root,sync_profile,profile_id
from output_validation import validate_output

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); a=ap.parse_args(); sync_profile(); exp=load_experiment(); dcfg=get_dataset(a.dataset)
    nsrc=count_enabled_sources(dcfg); nsr=len(exp['sampling']['sampling_rates']); nseed=len(exp['sampling']['seeds']); methods=exp['methods']['all']; expected_cases=nsrc*nsr*nseed; expected_recons=expected_cases*len(methods)
    cases=sorted(case_dir(a.dataset).glob('*.mat')); rows=[]; valid_total=0
    for method in methods:
        found=0; valid=0
        for c in cases:
            out=recon_dir(a.dataset,method)/f'{method}__{c.stem}.mat'
            if out.exists(): found+=1
            ok,_=validate_output(c,out)
            if ok: valid+=1
        valid_total+=valid; rows.append({'method':method,'expected':expected_cases,'files_found':found,'valid_outputs':valid,'complete':valid==expected_cases})
    mf=metrics_dir(a.dataset)/'per_run_metrics.csv'; missingf=metrics_dir(a.dataset)/'missing_or_failed_outputs.csv'; metric_rows=0; runtime_missing=0; nonfinite=0
    if mf.exists() and mf.stat().st_size>0:
        df=pd.read_csv(mf); metric_rows=len(df)
        for col in ['psnr_full','ssim_frame_mean']:
            if col in df.columns: nonfinite += int(pd.to_numeric(df[col],errors='coerce').isna().sum())
        if 'runtime_seconds' in df.columns: runtime_missing=int(pd.to_numeric(df['runtime_seconds'],errors='coerce').isna().sum())
    missing_rows=0
    if missingf.exists() and missingf.stat().st_size>3:
        try: missing_rows=len(pd.read_csv(missingf))
        except Exception: missing_rows=1
    passed=(len(cases)==expected_cases and valid_total==expected_recons and metric_rows==expected_recons and missing_rows==0 and nonfinite==0 and runtime_missing==0)
    outdir=result_root(a.dataset); outdir.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(outdir/'FINAL_METHOD_COMPLETENESS.csv',index=False,encoding='utf-8-sig')
    obj={'dataset':a.dataset,'profile_id':profile_id(),'pass':passed,'source_videos':nsrc,'sampling_rates':exp['sampling']['sampling_rates'],'seeds':exp['sampling']['seeds'],'cases_found':len(cases),'cases_expected':expected_cases,'valid_reconstructions':valid_total,'reconstructions_expected':expected_recons,'per_run_metric_rows':metric_rows,'missing_or_failed_rows':missing_rows,'runtime_missing_rows':runtime_missing,'nonfinite_primary_metric_values':nonfinite}
    (outdir/'FINAL_AUDIT.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    lines=[f"DATASET FINAL AUDIT: {a.dataset}",'='*76,f"Profile: {profile_id()}",f"Source videos: {nsrc}",f"SRs: {exp['sampling']['sampling_rates']}",f"Cases: {len(cases)}/{expected_cases}",f"Valid reconstructions: {valid_total}/{expected_recons}",f"Metric rows: {metric_rows}/{expected_recons}",f"Missing/failed metric rows: {missing_rows}",f"Missing runtime rows: {runtime_missing}",f"Non-finite primary metrics: {nonfinite}",'']
    for r in rows: lines.append(f"{r['method']:8s}: {r['valid_outputs']:4d}/{expected_cases}  {'PASS' if r['complete'] else 'INCOMPLETE'}")
    lines += ['', 'FINAL STATUS: '+('PASS' if passed else 'CHECK REQUIRED')]; text='\n'.join(lines)+'\n'; (outdir/'FINAL_AUDIT.txt').write_text(text,encoding='utf-8'); print(text)
    if not passed: raise SystemExit(4)
if __name__=='__main__': main()
