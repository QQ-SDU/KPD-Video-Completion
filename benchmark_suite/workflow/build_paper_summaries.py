from __future__ import annotations
import argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'workflow'))
from common import metrics_dir,tables_dir,sync_profile
DEV_TOKENS=('akiyo','foreman','mobile')
def aggregate(df,tag,out):
    agg=(df.groupby(['method','sampling_rate'],as_index=False).agg(n=('video','count'),psnr_mean=('psnr_full','mean'),psnr_std=('psnr_full','std'),ssim_mean=('ssim_frame_mean','mean'),ssim_std=('ssim_frame_mean','std'),missing_psnr_mean=('psnr_missing','mean'),temporal_mae_mean=('temporal_gradient_mae_missing','mean'),runtime_mean_seconds=('runtime_seconds','mean'),runtime_sum_seconds=('runtime_seconds','sum')))
    agg.to_csv(out/f'{tag}_summary_by_method_sr.csv',index=False,encoding='utf-8-sig')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); a=ap.parse_args(); sync_profile(); src=metrics_dir(a.dataset)/'per_run_metrics.csv'
    if not src.exists(): raise SystemExit('per_run_metrics.csv not found')
    df=pd.read_csv(src); out=tables_dir(a.dataset); out.mkdir(parents=True,exist_ok=True); aggregate(df,'all',out)
    if a.dataset.lower()=='formal23':
        vids=sorted(df.video.astype(str).unique()); dev=[v for v in vids if any(tok in v.lower() for tok in DEV_TOKENS)]
        if len(dev)==3:
            aggregate(df[~df.video.isin(dev)].copy(),'heldout20',out); pd.DataFrame({'development_video':dev}).to_csv(out/'heldout20_excluded_development_videos.csv',index=False,encoding='utf-8-sig'); print('Held-out-20 excluded:',dev)
        else:
            (out/'HELDOUT20_WARNING.txt').write_text('Expected exactly Akiyo/Foreman/Mobile; detected: '+', '.join(dev),encoding='utf-8')
if __name__=='__main__': main()
