#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V7 theory-aligned one-case tube-wise completion with cross-fitted residual learning.

Compares:
    Masked-CLSA
    KPD + simple residual interpolation
    KPD + cross-fitted tube residual CNN

The central diagnostic is whether cross-fitted pseudo-missing KPD residuals
match strict-validation KPD residuals better than old in-sample residuals do.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter, sobel

from kpd_video_core import (
    load_input, parse_resize, psnr, masked_psnr, mean_ssim, smooth_fill
)
from masked_kpd import run_masked_clsa, reconstruct as reconstruct_kpd
from detail_refiner_v41 import DetailResidualCNN, infer_detail
from detail_refiner_tube import (
    set_seed, exact_tube_mask, split_observed_tubes, split_tubes
)
from crossfit_tube_refiner import (
    build_crossfit_folds, train_crossfit_tube_cnn
)


def normalized_spatial_gaussian(values, mask, sigma_xy):
    sy, sx = sigma_xy
    m = mask.astype(np.float64)
    sigma = (sy, sx, 0.0)
    num = gaussian_filter(values*m, sigma=sigma, mode="nearest")
    den = gaussian_filter(m, sigma=sigma, mode="nearest")
    return num / np.maximum(den, 1e-8)


def masked_mae(gt, rec, mask):
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(gt[mask]-rec[mask])))


def calibrate_simple_residual(gt, raw_kpd, train_mask, seed):
    fit_mask, calib_mask = split_tubes(train_mask, 0.85, seed)
    residual = np.zeros_like(gt)
    residual[fit_mask] = gt[fit_mask] - raw_kpd[fit_mask]

    sigma_candidates = [
        (0.75,0.75),(1.0,1.0),(1.5,1.5),(2.0,2.0),(3.0,3.0)
    ]
    alpha_candidates = [0.5,0.75,1.0,1.25]

    target = gt[calib_mask]-raw_kpd[calib_mask]
    best=None
    rows=[]
    for sig in sigma_candidates:
        interp=normalized_spatial_gaussian(residual,fit_mask,sig)
        for alpha in alpha_candidates:
            mae=float(np.mean(np.abs(target-alpha*interp[calib_mask])))
            row={"sigma_y":sig[0],"sigma_x":sig[1],
                 "alpha":alpha,"calib_mae":mae}
            rows.append(row)
            if best is None or mae<best["calib_mae"]:
                best=row
    return (best["sigma_y"],best["sigma_x"]),best["alpha"],rows


def project_observed(rec,gt,mask):
    x=np.clip(rec,0,1).copy()
    x[mask]=gt[mask]
    return x


def gradient_masks(gt, missing_mask):
    gx=sobel(gt,axis=1,mode="nearest")
    gy=sobel(gt,axis=0,mode="nearest")
    mag=np.sqrt(gx*gx+gy*gy)
    vals=mag[missing_mask]
    q50=float(np.percentile(vals,50))
    q80=float(np.percentile(vals,80))
    return {
        "low":missing_mask & (mag<=q50),
        "mid":missing_mask & (mag>q50) & (mag<=q80),
        "high":missing_mask & (mag>q80),
    }


def metrics(gt,rec,obs_mask,regions):
    miss=~obs_mask
    return {
        "psnr_full":psnr(gt,rec),
        "psnr_missing":masked_psnr(gt,rec,miss),
        "ssim_full":mean_ssim(gt,rec),
        "psnr_low_gradient":masked_psnr(gt,rec,regions["low"]),
        "psnr_mid_gradient":masked_psnr(gt,rec,regions["mid"]),
        "psnr_high_gradient":masked_psnr(gt,rec,regions["high"]),
        "mae_missing":masked_mae(gt,rec,miss),
    }


def save_bar(path, rows, title):
    fig,ax=plt.subplots(figsize=(9.5,4.9))
    names=[r["method"] for r in rows]
    vals=[r["psnr_full"] for r in rows]
    bars=ax.bar(names,vals)
    ax.set_ylabel("Full-video PSNR (dB)")
    ax.set_title(title)
    ax.tick_params(axis="x",rotation=12)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{v:.2f}",
                ha="center",va="bottom")
    fig.tight_layout()
    fig.savefig(path,dpi=190,bbox_inches="tight")
    plt.close(fig)


def save_frames(path,gt,observed,vols,frames):
    items=[("GT",gt),("Observed",observed)]+vols
    fig,axes=plt.subplots(len(frames),len(items),
                          figsize=(3.0*len(items),3.0*len(frames)),
                          squeeze=False)
    for rr,t in enumerate(frames):
        t=int(np.clip(t,0,gt.shape[2]-1))
        for cc,(name,v) in enumerate(items):
            axes[rr,cc].imshow(v[:,:,t],cmap="gray",vmin=0,vmax=1)
            axes[rr,cc].set_title(name if rr==0 else "")
            axes[rr,cc].set_ylabel(f"Frame {t+1}" if cc==0 else "")
            axes[rr,cc].set_xticks([]); axes[rr,cc].set_yticks([])
    fig.tight_layout()
    fig.savefig(path,dpi=190,bbox_inches="tight")
    plt.close(fig)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--video-name",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=None,help="Optional reproducibility seed; omitted => generate a fresh seed.")
    p.add_argument("--sampling-rate",type=float,required=True)
    p.add_argument("--resize",default="144x176")
    p.add_argument("--max-frames",type=int,default=100)

    p.add_argument("--rank",type=int,default=12)
    p.add_argument("--masked-sweeps",type=int,default=12)
    p.add_argument("--lam-space",type=float,default=0.005)
    p.add_argument("--lam-time",type=float,default=0.05)
    p.add_argument("--ridge",type=float,default=1e-5)

    p.add_argument(
        "--protocol", choices=["development","final"], default="final",
        help=("development: reserve strict validation tubes for diagnostics; "
              "final: after hyperparameters are fixed, cross-fit on all observed tubes")
    )
    p.add_argument("--train-fraction",type=float,default=0.80)
    p.add_argument("--crossfit-folds",type=int,default=5)
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--patch-t",type=int,default=16)
    p.add_argument("--patch-h",type=int,default=48)
    p.add_argument("--patch-w",type=int,default=48)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--hidden",type=int,default=12)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--detail-weight",type=float,default=2.0)
    p.add_argument("--max-residual",type=float,default=0.25)
    p.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    p.add_argument("--show-frames",default="10,49,89")
    args=p.parse_args()

    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    print(f"Run seed: {args.seed}")
    set_seed(args.seed)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)

    if args.device=="auto":
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    gt,fps=load_input(
        Path(args.input),
        max_frames=args.max_frames,
        resize_hw=parse_resize(args.resize),
        start_frame=0
    )
    gt=np.asarray(gt,dtype=np.float64)

    obs_mask=exact_tube_mask(gt.shape,args.sampling_rate,args.seed)
    observed=np.zeros_like(gt); observed[obs_mask]=gt[obs_mask]

    if args.protocol == "development":
        train_mask,strict_val_mask=split_observed_tubes(
            obs_mask,args.train_fraction,args.seed+77
        )
    else:
        train_mask=obs_mask.copy()
        strict_val_mask=np.zeros_like(obs_mask,dtype=bool)

    regions=gradient_masks(gt,~obs_mask)

    print("============================================================")
    print(f"V7 THEORY-ALIGNED CROSS-FIT | {args.video_name} | SR={args.sampling_rate} | seed={args.seed}")
    print("============================================================")
    print("Protocol:",args.protocol)
    print("Device:",device)
    if device.type=="cuda":
        print("GPU:",torch.cuda.get_device_name(0))
    print("shape:",gt.shape)
    print(f"full observed tube fraction : {obs_mask[:,:,0].mean():.6f}")
    print(f"cross-fit pool fraction     : {train_mask[:,:,0].mean():.6f}")
    print(f"strict-val tube fraction    : {strict_val_mask[:,:,0].mean():.6f}")
    expected_context = train_mask[:,:,0].mean() * (1.0 - 1.0/args.crossfit_folds)
    print(f"expected fold context frac  : {expected_context:.6f}")

    # ------------------------------------------------------------
    # 1. Cross-fit-pool KPD backbone.  This is used for development
    # diagnostics and the simple residual baseline only.
    # ------------------------------------------------------------
    print("\n[1/6] Cross-fit-pool KPD backbone...")
    obs_train=np.zeros_like(observed); obs_train[train_mask]=observed[train_mask]
    init_train=smooth_fill(obs_train,train_mask)
    train_res=run_masked_clsa(
        observed=obs_train,mask=train_mask,initial_tensor=init_train,
        rank=args.rank,outer_sweeps=args.masked_sweeps,
        lam_space=args.lam_space,lam_time=args.lam_time,ridge=args.ridge,
        seed=args.seed,init="warm",project_observed=False,verbose=False,
        evaluation_gt=None,balance_scales=True,check_monotonic=True
    )
    train_backbone=train_res.backbone
    print("  KPD objective monotone:", all(h["monotone_ok"] for h in train_res.history))

    insample_r=observed[train_mask]-train_backbone[train_mask]
    insample_mae=float(np.mean(np.abs(insample_r)))
    has_strict=bool(np.any(strict_val_mask))
    if has_strict:
        strict_r=observed[strict_val_mask]-train_backbone[strict_val_mask]
        strict_resid_mae=float(np.mean(np.abs(strict_r)))
        strict_resid_rms=float(np.sqrt(np.mean(strict_r*strict_r)))
        print(f"  in-sample residual MAE       : {insample_mae:.6f}")
        print(f"  strict out-of-sample MAE     : {strict_resid_mae:.6f}")
        print(f"  strict/in-sample ratio       : {strict_resid_mae/max(insample_mae,1e-12):.3f}x")
    else:
        strict_resid_mae=float("nan")
        strict_resid_rms=float("nan")
        print(f"  in-sample residual MAE       : {insample_mae:.6f}")
        print("  strict validation            : omitted in final protocol")

    # ------------------------------------------------------------
    # 2. Cross-fitted KPD targets.  Only `observed` is passed, never GT.
    # ------------------------------------------------------------
    print("\n[2/6] Building cross-fitted pseudo-missing KPD states...")
    states=build_crossfit_folds(
        observed=observed,
        train_mask=train_mask,
        rank=args.rank,
        masked_sweeps=args.masked_sweeps,
        lam_space=args.lam_space,
        lam_time=args.lam_time,
        ridge=args.ridge,
        n_folds=args.crossfit_folds,
        seed=args.seed+500
    )

    cf_mae=float(np.mean([s.target_residual_mae for s in states]))
    cf_rms=float(np.mean([s.target_residual_rms for s in states]))
    new_ratio=(strict_resid_mae/max(cf_mae,1e-12)) if has_strict else float("nan")

    print("\nResidual-distribution diagnostic:")
    print(f"  in-sample target MAE        : {insample_mae:.6f}")
    print(f"  cross-fit pseudo target MAE : {cf_mae:.6f}")
    if has_strict:
        print(f"  strict-val target MAE       : {strict_resid_mae:.6f}")
        print(f"  strict / cross-fit ratio    : {new_ratio:.3f}x")

    # ------------------------------------------------------------
    # 3. Simple residual baseline.
    # ------------------------------------------------------------
    print("\n[3/6] Calibrating simple residual interpolation...")
    sigma,alpha,calib_rows=calibrate_simple_residual(
        observed,train_backbone,train_mask,args.seed+101
    )
    train_residual=np.zeros_like(observed)
    train_residual[train_mask]=observed[train_mask]-train_backbone[train_mask]
    simple_val_r=alpha*normalized_spatial_gaussian(
        train_residual,train_mask,sigma
    )
    simple_val=np.clip(train_backbone+simple_val_r,0,1)

    if has_strict:
        strict_kpd_mae=masked_mae(observed,train_backbone,strict_val_mask)
        strict_simple_mae=masked_mae(observed,simple_val,strict_val_mask)
        print("  selected sigma/alpha:",sigma,alpha)
        print(f"  strict-val MAE KPD/simple: {strict_kpd_mae:.6f}/{strict_simple_mae:.6f}")
    else:
        strict_kpd_mae=strict_simple_mae=float("nan")
        print("  selected sigma/alpha:",sigma,alpha)

    # ------------------------------------------------------------
    # 4. Cross-fitted tube CNN.  The training function has no GT argument.
    # ------------------------------------------------------------
    print("\n[4/6] Training cross-fitted tube residual CNN...")
    set_seed(args.seed+1000)
    model=DetailResidualCNN(
        hidden=args.hidden,max_residual=args.max_residual
    ).to(device)

    history=train_crossfit_tube_cnn(
        model=model,
        states=states,
        device=device,
        steps=args.steps,
        patch_t=args.patch_t,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        batch_size=args.batch_size,
        lr=args.lr,
        detail_weight=args.detail_weight,
        amp=True,
        print_every=max(1,args.steps//6),
        seed=args.seed+2000,
        smooth_l1_beta=0.02,
        magnitude_scale=0.05,
        magnitude_cap=3.0,
        residual_penalty=1e-4,
    )

    if has_strict:
        pred_val=infer_detail(model,train_backbone,obs_train,train_mask,device)
        cf_val=np.clip(train_backbone+pred_val,0,1)
        strict_cf_mae=masked_mae(observed,cf_val,strict_val_mask)
        print(f"  strict-val MAE KPD+CrossFitCNN: {strict_cf_mae:.6f}")
    else:
        strict_cf_mae=float("nan")

    # ------------------------------------------------------------
    # 5. Final KPD from all observed tubes.
    # ------------------------------------------------------------
    print("\n[5/6] Full observed-tube KPD backbone...")
    init_full=smooth_fill(observed,obs_mask)
    full_res=run_masked_clsa(
        observed=observed,mask=obs_mask,initial_tensor=init_full,
        rank=args.rank,outer_sweeps=args.masked_sweeps,
        lam_space=args.lam_space,lam_time=args.lam_time,ridge=args.ridge,
        seed=args.seed,init="warm",project_observed=True,verbose=False,
        evaluation_gt=None,balance_scales=True,check_monotonic=True
    )
    kpd_projected=full_res.reconstruction
    kpd_backbone=full_res.backbone
    print("  Full-KPD objective monotone:", all(h["monotone_ok"] for h in full_res.history))

    # ------------------------------------------------------------
    # 6. Final inference and data consistency.
    # ------------------------------------------------------------
    print("[6/6] Final inference...")
    full_residual=np.zeros_like(observed)
    full_residual[obs_mask]=observed[obs_mask]-kpd_backbone[obs_mask]
    simple_r=alpha*normalized_spatial_gaussian(
        full_residual,obs_mask,sigma
    )
    simple_final=project_observed(kpd_backbone+simple_r,observed,obs_mask)

    pred_cf=infer_detail(model,kpd_backbone,observed,obs_mask,device)
    cf_final=project_observed(kpd_backbone+pred_cf,observed,obs_mask)

    rows=[]
    for name,rec in [
        ("Masked-CLSA",kpd_projected),
        ("KPD + simple residual interpolation",simple_final),
        ("KPD + cross-fitted tube residual CNN",cf_final),
    ]:
        rows.append({"method":name,**metrics(gt,rec,obs_mask,regions)})

    _write_csv(out/"METHODS.csv",rows)
    _write_csv(out/"TRAIN_HISTORY.csv",history)
    _write_csv(out/"CALIBRATION.csv",calib_rows)
    _write_csv(out/"KPD_TRAIN_HISTORY.csv",train_res.history)
    _write_csv(out/"KPD_FULL_HISTORY.csv",full_res.history)

    fold_rows=[
        {
            "fold":s.fold,
            "context_tube_fraction":float(s.context_mask[:,:,0].mean()),
            "pseudo_target_tube_fraction":float(s.target_mask[:,:,0].mean()),
            "target_residual_mae":s.target_residual_mae,
            "target_residual_rms":s.target_residual_rms,
            "kpd_objective_final":s.kpd_objective_final,
            "kpd_monotone":s.kpd_monotone,
        } for s in states
    ]
    _write_csv(out/"CROSSFIT_FOLDS.csv",fold_rows)

    rmap={r["method"]:r for r in rows}
    kpd=rmap["Masked-CLSA"]
    simple=rmap["KPD + simple residual interpolation"]
    ours=rmap["KPD + cross-fitted tube residual CNN"]

    summary={
        "video":args.video_name,
        "seed":args.seed,
        "sampling_rate":args.sampling_rate,
        "protocol":{
            "mode":args.protocol,
            "shape_HWT":list(gt.shape),
            "mask_type":"exact mode-3 tube-wise",
            "crossfit_folds":args.crossfit_folds,
            "crossfit_pool_fraction_of_observed":float(train_mask.sum()/max(obs_mask.sum(),1)),
            "expected_fold_context_fraction_of_observed":float(
                (train_mask.sum()/max(obs_mask.sum(),1))*(1.0-1.0/args.crossfit_folds)
            ),
            "strict_validation_excluded_from_all_training": bool(has_strict),
            "true_missing_gt_used_in_kpd_or_cnn_training":False,
            "bounded_backbone_used_for_residual_learning":True,
            "regularizer_optimal_scale_balancing":True,
            "kpd_objective_monotonicity_checked":True,
        },
        "residual_distribution":{
            "in_sample_mae":insample_mae,
            "crossfit_pseudo_missing_mae":cf_mae,
            "crossfit_pseudo_missing_rms":cf_rms,
            "strict_val_out_of_sample_mae":strict_resid_mae,
            "strict_val_out_of_sample_rms":strict_resid_rms,
            "strict_over_insample":(strict_resid_mae/max(insample_mae,1e-12)) if has_strict else float("nan"),
            "strict_over_crossfit":new_ratio,
        },
        "strict_validation_mae":{
            "kpd":strict_kpd_mae,
            "simple":strict_simple_mae,
            "crossfit_cnn":strict_cf_mae,
        },
        "methods":rows,
        "gains_db":{
            "crossfit_vs_kpd":ours["psnr_full"]-kpd["psnr_full"],
            "crossfit_vs_simple":ours["psnr_full"]-simple["psnr_full"],
            "high_gradient_vs_kpd":ours["psnr_high_gradient"]-kpd["psnr_high_gradient"],
        }
    }
    (out/"SUMMARY.json").write_text(
        json.dumps(summary,indent=2,allow_nan=True),encoding="utf-8"
    )

    np.save(out/"tube_mask.npy",obs_mask[:,:,0])
    np.save(out/"kpd.npy",kpd_projected.astype(np.float32))
    np.save(out/"simple.npy",simple_final.astype(np.float32))
    np.save(out/"crossfit_cnn.npy",cf_final.astype(np.float32))

    save_bar(
        out/"FIG_PSNR.png",rows,
        f"{args.video_name}, tube SR={args.sampling_rate}, seed={args.seed}"
    )
    frames=[int(x.strip()) for x in args.show_frames.split(",") if x.strip()]
    save_frames(
        out/"VIS_FRAMES.png",gt,observed,
        [
            ("Masked-CLSA",kpd_projected),
            ("Simple residual",simple_final),
            ("CrossFit-CNN",cf_final),
        ],frames
    )

    print("\n================ V7 ONE CASE ================")
    print(f"{'Method':44s} {'PSNR':>8s} {'SSIM':>8s} {'High':>8s}")
    print("-"*74)
    for r in rows:
        print(
            f"{r['method']:44s} "
            f"{r['psnr_full']:8.3f} {r['ssim_full']:8.4f} "
            f"{r['psnr_high_gradient']:8.3f}"
        )

    print("\nTheory checks:")
    print("  train KPD monotone :", all(h["monotone_ok"] for h in train_res.history))
    print("  fold KPDs monotone :", all(s.kpd_monotone for s in states))
    print("  full KPD monotone  :", all(h["monotone_ok"] for h in full_res.history))
    print(f"  cross-fit context/full-observed ratio ≈ {summary['protocol']['expected_fold_context_fraction_of_observed']:.3f}")
    if has_strict:
        print("\nResidual mismatch:")
        print(f"  strict/in-sample : {summary['residual_distribution']['strict_over_insample']:.3f}x")
        print(f"  strict/cross-fit : {summary['residual_distribution']['strict_over_crossfit']:.3f}x")

    print("\nGains:")
    print(f"  vs KPD    : {summary['gains_db']['crossfit_vs_kpd']:+.3f} dB")
    print(f"  vs simple : {summary['gains_db']['crossfit_vs_simple']:+.3f} dB")
    print(f"  high-grad : {summary['gains_db']['high_gradient_vs_kpd']:+.3f} dB")


if __name__=="__main__":
    main()
