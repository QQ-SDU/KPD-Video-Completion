#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V7 controlled ablation and parameter-sensitivity suite.

Scientific protocol
-------------------
* Hyperparameter/model selection uses ONLY a strict validation subset carved
  from the genuinely observed tubes (development protocol).
* True missing ground truth is evaluated only after a candidate has been run;
  it is saved as a diagnostic/test metric and is NEVER used to rank candidates.
* Three representative videos may be used for robust selection. The default is
  Akiyo (low motion), Foreman (motion), Mobile (high texture/high resolution).
* Final-paper wrappers use the frozen Table-I base configuration (K=4).

Studies
-------
methods   : KPD / simple residual / direct in-sample residual CNN / CrossFit CNN
rank      : K in {4,8,12,16,20}
folds     : F in {2,3,5,8}
inputs    : B / BR / BRM / FULL, with the same 4-channel CNN architecture
lam_space : spatial smoothness sensitivity
lam_time  : temporal smoothness sensitivity
steps     : one training trajectory evaluated at 300/600/900/1200 steps

The best setting reported by this script means "best among the tested
candidates by mean strict-validation MAE"; it is not a global optimum claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import torch
import sys

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parents[1]
SRC_OURS = PAPER_ROOT / "src" / "ours_v7"
sys.path.insert(0, str(SRC_OURS))

from kpd_video_core import load_input, parse_resize, smooth_fill, psnr, mean_ssim, masked_psnr
from masked_kpd import run_masked_clsa
from detail_refiner_v41 import DetailResidualCNN
from detail_refiner_tube import set_seed, exact_tube_mask, split_observed_tubes, split_tubes
from crossfit_tube_refiner import build_crossfit_folds
from ablation_refiner import (
    train_crossfit_ablation,
    train_direct_tube_ablation,
    infer_ablation,
    VALID_INPUT_MODES,
)
from run_v51_one_case import normalized_spatial_gaussian


@dataclass(frozen=True)
class Config:
    rank: int = 4
    sweeps: int = 12
    lam_space: float = 0.005
    lam_time: float = 0.05
    ridge: float = 1e-5
    folds: int = 5
    steps: int = 1200
    patch_t: int = 16
    patch_h: int = 48
    patch_w: int = 48
    batch_size: int = 2
    hidden: int = 12
    lr: float = 1e-3
    detail_weight: float = 2.0
    max_residual: float = 0.25
    input_mode: str = "FULL"


def _write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _read_manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _device(spec: str):
    if spec == "auto":
        d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        d = torch.device(spec)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return d


def _project_observed(rec, observed, mask):
    x = np.clip(rec, 0.0, 1.0).copy()
    x[mask] = observed[mask]
    return x


def _masked_mae(reference, rec, mask):
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(reference[mask] - rec[mask])))


def _metrics(gt, rec, obs_mask):
    miss = ~obs_mask
    return {
        "test_psnr_full": float(psnr(gt, rec)),
        "test_ssim_full": float(mean_ssim(gt, rec)),
        "test_psnr_missing": float(masked_psnr(gt, rec, miss)),
        "test_mae_missing": float(np.mean(np.abs(gt[miss] - rec[miss]))),
    }


def _safe_simple_calibration(observed, backbone, train_mask, seed):
    fit_mask, calib_mask = split_tubes(train_mask, 0.85, seed)
    residual = np.zeros_like(observed)
    residual[fit_mask] = observed[fit_mask] - backbone[fit_mask]
    sigmas = [(0.75,0.75),(1.0,1.0),(1.5,1.5),(2.0,2.0),(3.0,3.0)]
    alphas = [0.5,0.75,1.0,1.25]
    target = observed[calib_mask] - backbone[calib_mask]
    best = None
    for sig in sigmas:
        interp = normalized_spatial_gaussian(residual, fit_mask, sig)
        for alpha in alphas:
            mae = float(np.mean(np.abs(target - alpha * interp[calib_mask])))
            row = {"sigma_y":sig[0], "sigma_x":sig[1], "alpha":alpha, "mae":mae}
            if best is None or mae < best["mae"]:
                best = row
    return (best["sigma_y"], best["sigma_x"]), best["alpha"]


class VideoContext:
    def __init__(self, row, root: Path, sr: float, seed: int, train_fraction: float, device):
        self.name = row["name"]
        self.file = root / row["file"]
        self.frames = int(row["frames"])
        self.device = device
        if not self.file.exists():
            raise FileNotFoundError(f"Missing video: {self.file}")
        self.gt, self.fps = load_input(
            self.file,
            max_frames=self.frames,
            resize_hw=None,
            start_frame=0,
        )
        self.gt = np.asarray(self.gt, dtype=np.float64)
        self.obs_mask = exact_tube_mask(self.gt.shape, sr, seed)
        self.observed = np.zeros_like(self.gt)
        self.observed[self.obs_mask] = self.gt[self.obs_mask]
        self.train_mask, self.strict_mask = split_observed_tubes(
            self.obs_mask, train_fraction=train_fraction, seed=seed+77
        )
        self.obs_train = np.zeros_like(self.observed)
        self.obs_train[self.train_mask] = self.observed[self.train_mask]
        self.kpd_cache = {}
        self.fold_cache = {}
        self.simple_cache = {}

    def get_kpd(self, cfg: Config, seed: int):
        key = (cfg.rank, cfg.sweeps, cfg.lam_space, cfg.lam_time, cfg.ridge)
        if key in self.kpd_cache:
            return self.kpd_cache[key]
        print(f"  [{self.name}] fitting training/full KPD for key={key}")
        init_train = smooth_fill(self.obs_train, self.train_mask)
        train_res = run_masked_clsa(
            observed=self.obs_train, mask=self.train_mask, initial_tensor=init_train,
            rank=cfg.rank, outer_sweeps=cfg.sweeps,
            lam_space=cfg.lam_space, lam_time=cfg.lam_time, ridge=cfg.ridge,
            seed=seed, init="warm", project_observed=False, verbose=False,
            evaluation_gt=None, balance_scales=True, check_monotonic=True,
        )
        init_full = smooth_fill(self.observed, self.obs_mask)
        full_res = run_masked_clsa(
            observed=self.observed, mask=self.obs_mask, initial_tensor=init_full,
            rank=cfg.rank, outer_sweeps=cfg.sweeps,
            lam_space=cfg.lam_space, lam_time=cfg.lam_time, ridge=cfg.ridge,
            seed=seed, init="warm", project_observed=True, verbose=False,
            evaluation_gt=None, balance_scales=True, check_monotonic=True,
        )
        out = {
            "train_backbone": train_res.backbone,
            "full_backbone": full_res.backbone,
            "full_projected": full_res.reconstruction,
            "train_monotone": all(bool(h["monotone_ok"]) for h in train_res.history),
            "full_monotone": all(bool(h["monotone_ok"]) for h in full_res.history),
        }
        self.kpd_cache[key] = out
        return out

    def get_folds(self, cfg: Config, seed: int):
        kpd_key = (cfg.rank, cfg.sweeps, cfg.lam_space, cfg.lam_time, cfg.ridge)
        key = kpd_key + (cfg.folds,)
        if key in self.fold_cache:
            return self.fold_cache[key]
        print(f"  [{self.name}] building {cfg.folds} cross-fit states")
        states = build_crossfit_folds(
            observed=self.observed,
            train_mask=self.train_mask,
            rank=cfg.rank,
            masked_sweeps=cfg.sweeps,
            lam_space=cfg.lam_space,
            lam_time=cfg.lam_time,
            ridge=cfg.ridge,
            n_folds=cfg.folds,
            seed=seed+123,
        )
        self.fold_cache[key] = states
        return states

    def get_simple(self, cfg: Config, seed: int):
        key = (cfg.rank, cfg.sweeps, cfg.lam_space, cfg.lam_time, cfg.ridge)
        if key in self.simple_cache:
            return self.simple_cache[key]
        kpd = self.get_kpd(cfg, seed)
        sigma, alpha = _safe_simple_calibration(
            self.observed, kpd["train_backbone"], self.train_mask, seed+301
        )
        self.simple_cache[key] = (sigma, alpha)
        return sigma, alpha


def _evaluate_model(ctx: VideoContext, model, cfg: Config, kpd, device):
    # Strict validation: only training observations are exposed to the CNN.
    pred_val = infer_ablation(
        model, kpd["train_backbone"], ctx.obs_train, ctx.train_mask,
        device, input_mode=cfg.input_mode
    )
    val_rec = np.clip(kpd["train_backbone"] + pred_val, 0.0, 1.0)
    val_mae = _masked_mae(ctx.observed, val_rec, ctx.strict_mask)

    # True-missing metrics are diagnostic only and are not used for selection.
    pred_full = infer_ablation(
        model, kpd["full_backbone"], ctx.observed, ctx.obs_mask,
        device, input_mode=cfg.input_mode
    )
    final = _project_observed(kpd["full_backbone"] + pred_full, ctx.observed, ctx.obs_mask)
    return val_mae, _metrics(ctx.gt, final, ctx.obs_mask)


def _run_crossfit(ctx, cfg, seed, device):
    kpd = ctx.get_kpd(cfg, seed)
    states = ctx.get_folds(cfg, seed)
    set_seed(seed+1000)
    model = DetailResidualCNN(hidden=cfg.hidden, max_residual=cfg.max_residual).to(device)
    t0 = time.perf_counter()
    train_crossfit_ablation(
        model, states, device,
        input_mode=cfg.input_mode,
        steps=cfg.steps,
        patch_t=cfg.patch_t, patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        batch_size=cfg.batch_size, lr=cfg.lr,
        detail_weight=cfg.detail_weight,
        amp=True, print_every=max(1, cfg.steps//6), seed=seed+2000,
    )
    train_seconds = time.perf_counter() - t0
    val_mae, met = _evaluate_model(ctx, model, cfg, kpd, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return val_mae, met, train_seconds, kpd, states


def _run_direct(ctx, cfg, seed, device):
    kpd = ctx.get_kpd(cfg, seed)
    set_seed(seed+1000)
    model = DetailResidualCNN(hidden=cfg.hidden, max_residual=cfg.max_residual).to(device)
    t0 = time.perf_counter()
    train_direct_tube_ablation(
        model, kpd["train_backbone"], ctx.observed, ctx.train_mask, device,
        input_mode=cfg.input_mode,
        steps=cfg.steps,
        patch_t=cfg.patch_t, patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        batch_size=cfg.batch_size, lr=cfg.lr,
        detail_weight=cfg.detail_weight,
        amp=True, print_every=max(1, cfg.steps//6), seed=seed+2000,
    )
    train_seconds = time.perf_counter() - t0
    val_mae, met = _evaluate_model(ctx, model, cfg, kpd, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return val_mae, met, train_seconds, kpd


def _run_kpd_only(ctx, cfg, seed):
    kpd = ctx.get_kpd(cfg, seed)
    val_mae = _masked_mae(ctx.observed, kpd["train_backbone"], ctx.strict_mask)
    return val_mae, _metrics(ctx.gt, kpd["full_projected"], ctx.obs_mask), kpd


def _run_simple(ctx, cfg, seed):
    kpd = ctx.get_kpd(cfg, seed)
    sigma, alpha = ctx.get_simple(cfg, seed)
    tr_resid = np.zeros_like(ctx.observed)
    tr_resid[ctx.train_mask] = ctx.observed[ctx.train_mask] - kpd["train_backbone"][ctx.train_mask]
    val_r = alpha * normalized_spatial_gaussian(tr_resid, ctx.train_mask, sigma)
    val_rec = np.clip(kpd["train_backbone"] + val_r, 0, 1)
    val_mae = _masked_mae(ctx.observed, val_rec, ctx.strict_mask)

    full_resid = np.zeros_like(ctx.observed)
    full_resid[ctx.obs_mask] = ctx.observed[ctx.obs_mask] - kpd["full_backbone"][ctx.obs_mask]
    full_r = alpha * normalized_spatial_gaussian(full_resid, ctx.obs_mask, sigma)
    final = _project_observed(kpd["full_backbone"] + full_r, ctx.observed, ctx.obs_mask)
    met = _metrics(ctx.gt, final, ctx.obs_mask)
    met.update({"simple_sigma_y":sigma[0], "simple_sigma_x":sigma[1], "simple_alpha":alpha})
    return val_mae, met, kpd


def _candidate_specs(study, raw_candidates=None):
    if raw_candidates:
        vals = [x.strip() for x in raw_candidates.split(",") if x.strip()]
    else:
        vals = None
    if study == "methods":
        return vals or ["KPD", "SIMPLE", "DIRECT", "CROSSFIT"]
    if study == "rank":
        return [int(x) for x in (vals or [4,8,12,16,20])]
    if study == "folds":
        return [int(x) for x in (vals or [2,3,5,8])]
    if study == "inputs":
        out = [str(x).upper() for x in (vals or list(VALID_INPUT_MODES))]
        for x in out:
            if x not in VALID_INPUT_MODES:
                raise ValueError(f"Bad input candidate {x}; choose from {VALID_INPUT_MODES}")
        return out
    if study == "lam_space":
        return [float(x) for x in (vals or [0.001,0.005,0.01,0.05])]
    if study == "lam_time":
        return [float(x) for x in (vals or [0.01,0.05,0.1,0.5])]
    if study == "steps":
        return [int(x) for x in (vals or [300,600,900,1200])]
    raise ValueError(study)


def _cfg_for_candidate(base: Config, study: str, candidate):
    if study == "rank":
        return replace(base, rank=int(candidate))
    if study == "folds":
        return replace(base, folds=int(candidate))
    if study == "inputs":
        return replace(base, input_mode=str(candidate).upper())
    if study == "lam_space":
        return replace(base, lam_space=float(candidate))
    if study == "lam_time":
        return replace(base, lam_time=float(candidate))
    if study == "steps":
        return replace(base, steps=int(candidate))
    return base


def _aggregate(rows, study):
    candidates = []
    for r in rows:
        c = str(r["candidate"])
        if c not in candidates:
            candidates.append(c)
    out = []
    for c in candidates:
        rr = [r for r in rows if str(r["candidate"]) == c]
        vals = np.array([float(r["strict_val_mae"]) for r in rr], dtype=float)
        ps = np.array([float(r["test_psnr_full"]) for r in rr], dtype=float)
        ss = np.array([float(r["test_ssim_full"]) for r in rr], dtype=float)
        times = np.array([float(r.get("cnn_train_seconds", 0.0)) for r in rr], dtype=float)
        out.append({
            "study":study,
            "candidate":c,
            "n_videos":len(rr),
            "mean_strict_val_mae_SELECTION_SCORE":float(np.mean(vals)),
            "std_strict_val_mae":float(np.std(vals)),
            "mean_test_psnr_DIAGNOSTIC_ONLY":float(np.mean(ps)),
            "mean_test_ssim_DIAGNOSTIC_ONLY":float(np.mean(ss)),
            "mean_cnn_train_seconds":float(np.mean(times)),
        })
    out.sort(key=lambda r: r["mean_strict_val_mae_SELECTION_SCORE"])
    for i, r in enumerate(out, start=1):
        r["validation_rank"] = i
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True, choices=["methods","rank","folds","inputs","lam_space","lam_time","steps"])
    p.add_argument("--root", default=".")
    p.add_argument("--manifest", default="delta_tableiv_working_manifest.csv")
    p.add_argument("--videos", default="Akiyo,Foreman,Mobile")
    p.add_argument("--sampling-rate", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=None, help="Optional reproducibility seed; omitted => generate a fresh seed.")
    p.add_argument("--train-fraction", type=float, default=0.80)
    p.add_argument("--output", default=None)
    p.add_argument("--candidates", default=None, help="Comma-separated override")
    p.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")

    # Base configuration; update these after each selected stage if desired.
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--sweeps", type=int, default=12)
    p.add_argument("--lam-space", type=float, default=0.005)
    p.add_argument("--lam-time", type=float, default=0.05)
    p.add_argument("--ridge", type=float, default=1e-5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--patch-t", type=int, default=16)
    p.add_argument("--patch-h", type=int, default=48)
    p.add_argument("--patch-w", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--hidden", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--detail-weight", type=float, default=2.0)
    p.add_argument("--max-residual", type=float, default=0.25)
    p.add_argument("--input-mode", choices=list(VALID_INPUT_MODES), default="FULL")
    args = p.parse_args()

    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    print(f"Run seed: {args.seed}")

    root = Path(args.root).resolve()
    manifest_path = root / args.manifest
    manifest = _read_manifest(manifest_path)
    wanted = [x.strip() for x in args.videos.split(",") if x.strip()]
    by_name = {r["name"]:r for r in manifest}
    missing_names = [x for x in wanted if x not in by_name]
    if missing_names:
        raise KeyError(f"Videos not in manifest: {missing_names}")
    device = _device(args.device)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    base = Config(
        rank=args.rank, sweeps=args.sweeps,
        lam_space=args.lam_space, lam_time=args.lam_time, ridge=args.ridge,
        folds=args.folds, steps=args.steps,
        patch_t=args.patch_t, patch_h=args.patch_h, patch_w=args.patch_w,
        batch_size=args.batch_size, hidden=args.hidden, lr=args.lr,
        detail_weight=args.detail_weight, max_residual=args.max_residual,
        input_mode=args.input_mode,
    )
    candidates = _candidate_specs(args.study, args.candidates)
    out = Path(args.output or f"ablation_{args.study}_sr{int(round(args.sampling_rate*100)):02d}")
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "study":args.study,
        "selection_rule":"minimum mean strict-validation MAE across selected videos",
        "true_missing_metrics_used_for_selection":False,
        "sampling_rate":args.sampling_rate,
        "train_fraction":args.train_fraction,
        "videos":wanted,
        "candidates":candidates,
        "base_config":asdict(base),
        "note":"Best means best among tested candidates, not global optimum.",
    }
    (out/"STUDY_PROTOCOL.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    contexts = {}
    for name in wanted:
        print("\n" + "="*78)
        print(f"Loading {name}")
        print("="*78)
        contexts[name] = VideoContext(
            by_name[name], root, args.sampling_rate, args.seed,
            args.train_fraction, device
        )
        c = contexts[name]
        print("shape:", c.gt.shape,
              "obs=", float(c.obs_mask[:,:,0].mean()),
              "train=", float(c.train_mask[:,:,0].mean()),
              "strict=", float(c.strict_mask[:,:,0].mean()))

    rows = []

    # Training-step sensitivity is one shared trajectory per video.
    if args.study == "steps":
        max_steps = max(int(x) for x in candidates)
        for name in wanted:
            ctx = contexts[name]
            cfg = replace(base, steps=max_steps)
            kpd = ctx.get_kpd(cfg, args.seed)
            states = ctx.get_folds(cfg, args.seed)
            set_seed(args.seed+1000)
            model = DetailResidualCNN(hidden=cfg.hidden, max_residual=cfg.max_residual).to(device)

            def checkpoint(step, mdl):
                cfg_step = replace(cfg, steps=int(step))
                t_eval = time.perf_counter()
                val_mae, met = _evaluate_model(ctx, mdl, cfg_step, kpd, device)
                rows.append({
                    "study":args.study, "candidate":str(step), "video":name,
                    "strict_val_mae":val_mae,
                    **met,
                    "cnn_train_seconds":float("nan"),
                    "kpd_train_monotone":kpd["train_monotone"],
                    "kpd_full_monotone":kpd["full_monotone"],
                    "test_metrics_used_for_selection":False,
                    "eval_seconds":time.perf_counter()-t_eval,
                    **asdict(cfg_step),
                })
                _write_csv(out/"RESULTS_PARTIAL.csv", rows)
                print(f"  CHECKPOINT {step}: strict MAE={val_mae:.6f}, test PSNR={met['test_psnr_full']:.3f}")

            train_crossfit_ablation(
                model, states, device, input_mode=cfg.input_mode,
                steps=max_steps,
                patch_t=cfg.patch_t, patch_h=cfg.patch_h, patch_w=cfg.patch_w,
                batch_size=cfg.batch_size, lr=cfg.lr, detail_weight=cfg.detail_weight,
                amp=True, print_every=max(1,max_steps//6), seed=args.seed+2000,
                checkpoint_steps=[int(x) for x in candidates], checkpoint_callback=checkpoint,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        for cand in candidates:
            print("\n" + "#"*78)
            print(f"STUDY={args.study} | CANDIDATE={cand}")
            print("#"*78)
            cfg = _cfg_for_candidate(base, args.study, cand)
            for name in wanted:
                ctx = contexts[name]
                print(f"\n--- {name} | candidate={cand} ---")
                wall0 = time.perf_counter()
                train_sec = 0.0
                if args.study == "methods":
                    method = str(cand).upper()
                    if method == "KPD":
                        val_mae, met, kpd = _run_kpd_only(ctx, cfg, args.seed)
                    elif method == "SIMPLE":
                        val_mae, met, kpd = _run_simple(ctx, cfg, args.seed)
                    elif method == "DIRECT":
                        val_mae, met, train_sec, kpd = _run_direct(ctx, cfg, args.seed, device)
                    elif method == "CROSSFIT":
                        val_mae, met, train_sec, kpd, states = _run_crossfit(ctx, cfg, args.seed, device)
                    else:
                        raise ValueError(f"Unknown method candidate {cand}")
                else:
                    val_mae, met, train_sec, kpd, states = _run_crossfit(ctx, cfg, args.seed, device)

                row = {
                    "study":args.study,
                    "candidate":str(cand),
                    "video":name,
                    "strict_val_mae":val_mae,
                    **met,
                    "cnn_train_seconds":train_sec,
                    "wall_seconds":time.perf_counter()-wall0,
                    "kpd_train_monotone":kpd["train_monotone"],
                    "kpd_full_monotone":kpd["full_monotone"],
                    "test_metrics_used_for_selection":False,
                    **asdict(cfg),
                }
                rows.append(row)
                _write_csv(out/"RESULTS_PARTIAL.csv", rows)
                print(
                    f"RESULT {name} {cand}: strict MAE={val_mae:.6f}; "
                    f"test PSNR={met['test_psnr_full']:.3f}; SSIM={met['test_ssim_full']:.4f}"
                )

    _write_csv(out/"RESULTS.csv", rows)
    agg = _aggregate(rows, args.study)
    _write_csv(out/"AGGREGATE_BY_VALIDATION.csv", agg)
    best = agg[0] if agg else None
    (out/"BEST_BY_VALIDATION.json").write_text(
        json.dumps({"best":best, "all":agg}, indent=2, allow_nan=True), encoding="utf-8"
    )

    print("\n" + "="*78)
    print("VALIDATION RANKING (selection uses strict validation ONLY)")
    print("="*78)
    for r in agg:
        print(
            f"#{r['validation_rank']:02d} {r['candidate']:>10s} | "
            f"strict MAE={r['mean_strict_val_mae_SELECTION_SCORE']:.6f} | "
            f"test PSNR(diag)={r['mean_test_psnr_DIAGNOSTIC_ONLY']:.3f} | "
            f"SSIM(diag)={r['mean_test_ssim_DIAGNOSTIC_ONLY']:.4f}"
        )
    if best:
        print("\nBEST AMONG TESTED CANDIDATES BY STRICT VALIDATION:", best["candidate"])
    print("Outputs:", out)


if __name__ == "__main__":
    main()
