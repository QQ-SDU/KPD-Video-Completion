from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import metrics_dir,tables_dir,figures_dir,sync_profile
DISPLAY={"S2NTNN":"S$^2$NTNN","DRO-TFF":"DRO-TFF"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); a=ap.parse_args(); sync_profile()
    src=metrics_dir(a.dataset)/"per_run_metrics.csv"
    if not src.exists() or src.stat().st_size==0: raise SystemExit("No metrics.")
    df=pd.read_csv(src); tdir=tables_dir(a.dataset); fdir=figures_dir(a.dataset); tdir.mkdir(parents=True,exist_ok=True); fdir.mkdir(parents=True,exist_ok=True)
    agg=(df.groupby(["method","sampling_rate"],as_index=False).agg(n=("video","count"),psnr_mean=("psnr_full","mean"),psnr_std=("psnr_full","std"),
         ssim_mean=("ssim_frame_mean","mean"),ssim_std=("ssim_frame_mean","std"),missing_psnr_mean=("psnr_missing","mean"),missing_mae_mean=("mae_missing","mean"),
         temporal_mae_mean=("temporal_gradient_mae_missing","mean"),runtime_mean_seconds=("runtime_seconds","mean"),runtime_sum_seconds=("runtime_seconds","sum")))
    agg.to_csv(tdir/"summary_by_method_sr.csv",index=False,encoding="utf-8-sig"); df.to_csv(tdir/"per_video_method_sr.csv",index=False,encoding="utf-8-sig")
    methods=list(dict.fromkeys(df.method.tolist())); srs=sorted(df.sampling_rate.unique()); wide=[]
    for m in methods:
        row={"Method":m}; dm=agg[agg.method==m]
        for sr in srs:
            r=dm[np.isclose(dm.sampling_rate,sr)]; tag=f"{int(round(sr*100)):02d}"
            if not r.empty:
                row[f"SR{tag}_PSNR"]=float(r.iloc[0].psnr_mean); row[f"SR{tag}_SSIM"]=float(r.iloc[0].ssim_mean); row[f"SR{tag}_Runtime_s"]=float(r.iloc[0].runtime_mean_seconds)
        wide.append(row)
    wide=pd.DataFrame(wide); wide.to_csv(tdir/"paper_psnr_ssim_runtime_table.csv",index=False,encoding="utf-8-sig")
    ranks=[]
    for sr in srs:
        d=agg[np.isclose(agg.sampling_rate,sr)].copy(); d["psnr_rank"]=d.psnr_mean.rank(ascending=False,method="min"); d["ssim_rank"]=d.ssim_mean.rank(ascending=False,method="min"); ranks.append(d)
    pd.concat(ranks,ignore_index=True).to_csv(tdir/"ranking_by_sr.csv",index=False,encoding="utf-8-sig")
    tex=[r"\begin{table*}[!t]",r"\centering",r"\caption{Strict same-mask comparison on %s.}"%a.dataset.replace("_",r"\_"),r"\begin{tabular}{%s}"%("l"+"cc"*len(srs)),r"\hline"]
    header=["Method"]; [header.extend([f"PSNR@{sr:.1f}",f"SSIM@{sr:.1f}"]) for sr in srs]; tex.append(" & ".join(header)+r" \\"); tex.append(r"\hline")
    for _,r in wide.iterrows():
        vals=[DISPLAY.get(r["Method"],r["Method"])]
        for sr in srs:
            tag=f"{int(round(sr*100)):02d}"; p=r.get(f"SR{tag}_PSNR",np.nan); s=r.get(f"SR{tag}_SSIM",np.nan)
            vals += ["--" if pd.isna(p) else f"{p:.3f}","--" if pd.isna(s) else f"{s:.3f}"]
        tex.append(" & ".join(vals)+r" \\")
    tex += [r"\hline",r"\end{tabular}",r"\end{table*}"]; (tdir/"paper_psnr_ssim_table.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    for metric,label,fn in [("psnr_mean","PSNR (dB)","psnr_vs_sampling_rate"),("ssim_mean","SSIM","ssim_vs_sampling_rate"),("temporal_mae_mean","Temporal-gradient MAE","temporal_mae_vs_sampling_rate")]:
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        for m in methods:
            d=agg[agg.method==m].sort_values("sampling_rate")
            if not d.empty: ax.plot(d.sampling_rate,d[metric],marker="o",label=DISPLAY.get(m,m))
        ax.set_xlabel("Sampling rate"); ax.set_ylabel(label); ax.grid(True,alpha=.25); ax.legend(fontsize=7,ncol=2); fig.tight_layout()
        fig.savefig(fdir/f"{fn}.png",dpi=300); fig.savefig(fdir/f"{fn}.pdf"); plt.close(fig)
    for sr in srs:
        d=agg[np.isclose(agg.sampling_rate,sr)].sort_values("runtime_mean_seconds"); fig,ax=plt.subplots(figsize=(8,4.8))
        ax.bar([DISPLAY.get(x,x) for x in d.method],d.runtime_mean_seconds); ax.set_ylabel("Mean runtime (s)"); ax.set_xlabel("Method"); ax.tick_params(axis="x",rotation=45); ax.grid(True,axis="y",alpha=.25); fig.tight_layout()
        tag=f"{int(round(sr*100)):02d}"; fig.savefig(fdir/f"runtime_sr{tag}.png",dpi=300); fig.savefig(fdir/f"runtime_sr{tag}.pdf"); plt.close(fig)
    print("Tables:",tdir); print("Figures:",fdir)

if __name__=="__main__": main()
