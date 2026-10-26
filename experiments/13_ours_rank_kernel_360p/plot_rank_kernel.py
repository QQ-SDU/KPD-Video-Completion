#!/usr/bin/env python3
"""Publication-ready vector PDF and PNG illustrating Rank × spatial kernel diagnostics."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[2]/'results/rank_kernel_rgb360p_seed20261014')
    ap.add_argument('--metric',choices=['psnr_missing_mean','ssim_full_mean'],default='psnr_missing_mean')
    a=ap.parse_args()
    src=a.results/'summary_rank_kernel.csv'
    with src.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise RuntimeError('Empty summary: '+str(src))
    rates=sorted({float(row['sampling_rate']) for row in rows})
    a.results.mkdir(parents=True,exist_ok=True)
    for sr in rates:
        fig,ax=plt.subplots(figsize=(5.4,3.25))
        for k in sorted({int(row['spatial_kernel']) for row in rows}):
            vals=sorted([row for row in rows if float(row['sampling_rate'])==sr and int(row['spatial_kernel'])==k],key=lambda r:int(r['rank']))
            if not vals:continue
            xs=[int(r['rank']) for r in vals]
            ys=[float(r[a.metric]) for r in vals]
            ax.plot(xs,ys,marker='o',linewidth=1.7,label=f'3x{k}x{k} kernel')
        ax.set_xlabel('KPD rank K')
        ax.set_ylabel('Missing-region PSNR (dB)' if a.metric=='psnr_missing_mean' else 'Full-frame RGB SSIM')
        ax.set_xticks(sorted({int(row['rank']) for row in rows if float(row['sampling_rate'])==sr}))
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        base=a.results/f'rank_kernel_SR{int(sr*100):03d}_{a.metric}'
        fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight')
        fig.savefig(base.with_suffix('.png'),dpi=250,bbox_inches='tight')
        plt.close(fig)
        print('Saved',base.with_suffix('.pdf'))
if __name__=='__main__':main()
