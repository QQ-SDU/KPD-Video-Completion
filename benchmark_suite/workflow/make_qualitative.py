from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Benchmark"))
from benchmark.evaluate_output import load_case,load_recon,align_shape
from common import case_dir,recon_dir,figures_dir,load_experiment,sr_tag,sync_profile

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--video",required=True); ap.add_argument("--sr",type=float,default=.2); ap.add_argument("--seed",type=int,default=1); ap.add_argument("--frame",type=int,default=None); ap.add_argument("--roi",default=None); ap.add_argument("--methods",default=None); a=ap.parse_args(); sync_profile()
    xs=sorted(case_dir(a.dataset).glob(f"*{a.video}*__sr{sr_tag(a.sr)}__seed{a.seed}.npz"))
    if not xs: raise SystemExit("No matching case.")
    case=xs[0]; gt,obs,mask=load_case(case); h,w,t=gt.shape; fi=(t-1)//2 if a.frame is None else max(0,min(t-1,a.frame))
    if a.roi: x0,y0,rw,rh=map(int,a.roi.split(","))
    else: s=max(16,int(round(min(h,w)*.25))); x0=(w-s)//2; y0=(h-s)//2; rw=rh=s
    methods=a.methods.split(",") if a.methods else load_experiment()["methods"]["all"]; items=[("Observed",obs)]
    for m in methods:
        p=recon_dir(a.dataset,m)/f"{m}__{case.stem}.mat"
        if not p.exists(): continue
        rec,_=load_recon(p); rec=align_shape(rec,gt.shape); rec=np.clip(rec,0,1); rec[mask]=obs[mask]; items.append((m,rec))
    items.append(("GT",gt)); n=len(items); fig,axs=plt.subplots(2,n,figsize=(1.8*n,4),gridspec_kw={"height_ratios":[1,.7]},squeeze=False)
    for j,(m,v) in enumerate(items):
        ax=axs[0,j]; ax.imshow(v[:,:,fi],cmap="gray",vmin=0,vmax=1); ax.add_patch(Rectangle((x0,y0),rw,rh,fill=False,edgecolor="red",linewidth=1.4)); ax.set_title(m,fontsize=9,fontweight="bold"); ax.axis("off")
        az=axs[1,j]; az.imshow(v[y0:y0+rh,x0:x0+rw,fi],cmap="gray",vmin=0,vmax=1); az.axis("off")
    plt.subplots_adjust(left=.004,right=.996,top=.93,bottom=.01,wspace=.025,hspace=.05); out=figures_dir(a.dataset)/"qualitative"; out.mkdir(parents=True,exist_ok=True); stem=f"{case.stem}__frame{fi}"
    fig.savefig(out/f"{stem}.png",dpi=300,bbox_inches="tight",pad_inches=.02); fig.savefig(out/f"{stem}.pdf",bbox_inches="tight",pad_inches=.02); plt.close(fig); print("Saved:",out)

if __name__=="__main__": main()
