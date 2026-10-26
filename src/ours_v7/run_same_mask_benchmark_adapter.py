#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Same-mask runner for the V7 theory-aligned proposed method.

Method:
  Kronecker-structured masked KPD backbone
  + 5-fold cross-fitted self-supervised residual CNN
  + exact data consistency.

Fairness / no-leakage guarantee:
  - The benchmark case's exact supplied mask is used.
  - This runner loads ONLY `observed` and `mask` from the MAT file.
    `gt` is not loaded by the method process.
  - Cross-fit fold KPDs see context observations only.
  - No GT-based checkpoint selection or early stopping is used.
  - Runtime includes cross-fit KPD fitting, CNN training, final KPD fitting,
    and final inference, but excludes case-file I/O and output saving.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

HERE = Path(__file__).resolve().parent
REF = HERE / "reference_v7"
sys.path.insert(0, str(REF))

from kpd_video_core import smooth_fill
from masked_kpd import run_masked_clsa
from detail_refiner_v41 import DetailResidualCNN, infer_detail
from detail_refiner_tube import set_seed
from crossfit_tube_refiner import build_crossfit_folds, train_crossfit_tube_cnn


def load_observation_only(case_path: Path):
    # variable_names is deliberate: the method process does not load `gt`.
    d = sio.loadmat(case_path, variable_names=["observed", "mask"])
    if "observed" not in d or "mask" not in d:
        raise KeyError(f"Case must contain observed and mask: {case_path}")
    observed = np.asarray(d["observed"], dtype=np.float64)
    mask = np.asarray(d["mask"]).astype(bool)
    if observed.ndim != 3 or mask.ndim != 3 or observed.shape != mask.shape:
        raise ValueError(
            f"Expected matching HxWxT observed/mask arrays, got "
            f"{observed.shape} and {mask.shape}"
        )
    # The benchmark protocol is mode-3 tube-wise. Verify rather than assume.
    if not np.all(mask == mask[:, :, :1]):
        raise ValueError("Supplied mask is not mode-3 tube-wise.")
    # Ignore anything outside the supplied observation mask.
    clean = np.zeros_like(observed, dtype=np.float64)
    clean[mask] = observed[mask]
    return clean, mask


def project_observed(rec, observed, mask):
    x = np.clip(np.asarray(rec, dtype=np.float64), 0.0, 1.0)
    x = x.copy()
    x[mask] = observed[mask]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None, help="Optional reproducibility seed; omitted => generate a fresh seed.")

    # Frozen final paper configuration. K=4 was selected by the sensitivity study.
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--masked-sweeps", type=int, default=12)
    ap.add_argument("--lam-space", type=float, default=5e-3)
    ap.add_argument("--lam-time", type=float, default=5e-2)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--crossfit-folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--patch-t", type=int, default=16)
    ap.add_argument("--patch-h", type=int, default=48)
    ap.add_argument("--patch-w", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--detail-weight", type=float, default=2.0)
    ap.add_argument("--max-residual", type=float, default=0.25)
    ap.add_argument("--smooth-l1-beta", type=float, default=0.02)
    ap.add_argument("--magnitude-scale", type=float, default=0.05)
    ap.add_argument("--magnitude-cap", type=float, default=3.0)
    ap.add_argument("--residual-penalty", type=float, default=1e-4)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    print(f"Run seed: {args.seed}")

    case_path = Path(args.case).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    observed, obs_mask = load_observation_only(case_path)
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    print("=" * 76)
    print("OURS V7 theory-aligned same-mask runner")
    print("case:", case_path.name)
    print("shape:", observed.shape)
    print("realized tube SR:", f"{obs_mask[:, :, 0].mean():.6f}")
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print(
        f"K={args.rank}, sweeps={args.masked_sweeps}, "
        f"folds={args.crossfit_folds}, CNN steps={args.steps}"
    )
    print("GT loaded by method process: False")
    print("=" * 76)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # -------------------------------------------------------------
    # Algorithm 2, Steps 1-8: cross-fitted pseudo-missing states.
    # Final protocol: all observed tubes enter the cross-fit pool.
    # -------------------------------------------------------------
    states = build_crossfit_folds(
        observed=observed,
        train_mask=obs_mask,
        rank=args.rank,
        masked_sweeps=args.masked_sweeps,
        lam_space=args.lam_space,
        lam_time=args.lam_time,
        ridge=args.ridge,
        n_folds=args.crossfit_folds,
        seed=args.seed + 500,
    )

    # -------------------------------------------------------------
    # Algorithm 2, Step 9: self-supervised residual CNN.
    # -------------------------------------------------------------
    set_seed(args.seed + 1000)
    model = DetailResidualCNN(
        hidden=args.hidden, max_residual=args.max_residual
    ).to(device)

    history = train_crossfit_tube_cnn(
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
        print_every=max(1, args.steps // 6),
        seed=args.seed + 2000,
        smooth_l1_beta=args.smooth_l1_beta,
        magnitude_scale=args.magnitude_scale,
        magnitude_cap=args.magnitude_cap,
        residual_penalty=args.residual_penalty,
    )

    # -------------------------------------------------------------
    # Algorithm 2, Step 10: final KPD fit on ALL observed tubes.
    # Initialization uses observed data only.
    # -------------------------------------------------------------
    print("\nFinal all-observed KPD fit...")
    init_full = smooth_fill(observed, obs_mask)
    final_kpd = run_masked_clsa(
        observed=observed,
        mask=obs_mask,
        initial_tensor=init_full,
        rank=args.rank,
        outer_sweeps=args.masked_sweeps,
        lam_space=args.lam_space,
        lam_time=args.lam_time,
        ridge=args.ridge,
        seed=args.seed,
        init="warm",
        project_observed=True,
        verbose=False,
        evaluation_gt=None,
        balance_scales=True,
        check_monotonic=True,
    )
    backbone = final_kpd.backbone
    final_monotone = all(bool(h["monotone_ok"]) for h in final_kpd.history)
    print("Final KPD objective monotone:", final_monotone)

    # -------------------------------------------------------------
    # Algorithm 2, Steps 11-13: residual inference + data consistency.
    # -------------------------------------------------------------
    print("Final residual inference...")
    pred = infer_detail(model, backbone, observed, obs_mask, device)
    recon = project_observed(backbone + pred, observed, obs_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - t0

    fold_monotone = all(bool(s.kpd_monotone) for s in states)
    diag = {
        "method": "Ours",
        "version": "V7 theory-aligned",
        "case": case_path.stem,
        "shape_hwt": list(map(int, observed.shape)),
        "seed": int(args.seed),
        "realized_sampling_rate": float(obs_mask[:, :, 0].mean()),
        "protocol": "final",
        "gt_loaded_by_method_process": False,
        "crossfit_fold_kpd_monotone": bool(fold_monotone),
        "final_kpd_monotone": bool(final_monotone),
        "runtime_seconds": float(runtime_seconds),
        "config": {
            "rank": args.rank,
            "masked_sweeps": args.masked_sweeps,
            "lambda_s": args.lam_space,
            "lambda_t": args.lam_time,
            "lambda_0": args.ridge,
            "crossfit_folds": args.crossfit_folds,
            "hidden": args.hidden,
            "max_residual": args.max_residual,
            "patch": [args.patch_t, args.patch_h, args.patch_w],
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "smooth_l1_beta": args.smooth_l1_beta,
            "detail_weight": args.detail_weight,
            "magnitude_scale": args.magnitude_scale,
            "magnitude_cap": args.magnitude_cap,
            "residual_penalty": args.residual_penalty,
        },
        "folds": [
            {
                "fold": int(s.fold),
                "context_tube_fraction": float(s.context_mask[:, :, 0].mean()),
                "pseudo_target_tube_fraction": float(s.target_mask[:, :, 0].mean()),
                "target_residual_mae": float(s.target_residual_mae),
                "target_residual_rms": float(s.target_residual_rms),
                "kpd_objective_final": float(s.kpd_objective_final),
                "kpd_monotone": bool(s.kpd_monotone),
            }
            for s in states
        ],
        "final_kpd_history": final_kpd.history,
        "cnn_last": history[-1] if history else None,
    }

    # Use standard MAT (not -v7.3) so the shared scipy evaluator can read it.
    sio.savemat(
        out_path,
        {
            "recon": recon.astype(np.float32),
            "runtime_seconds": np.array([[runtime_seconds]], dtype=np.float64),
            "rank": np.array([[args.rank]], dtype=np.int32),
            "masked_sweeps": np.array([[args.masked_sweeps]], dtype=np.int32),
            "crossfit_folds": np.array([[args.crossfit_folds]], dtype=np.int32),
            "cnn_steps": np.array([[args.steps]], dtype=np.int32),
        },
        do_compression=True,
    )
    out_path.with_suffix(".json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nOurs done: {runtime_seconds:.3f} s")
    print("Fold KPD objectives monotone:", fold_monotone)
    print("Final KPD objective monotone:", final_monotone)
    print("Saved:", out_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
