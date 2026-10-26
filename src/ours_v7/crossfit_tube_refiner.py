#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Theory-aligned cross-fitted tube residual learning.

For each fold f:
    context_f = training observed tubes minus pseudo_missing_f
    raw_f     = masked KPD fitted from context_f only
    B_f       = clip(raw_f, 0, 1)
    target_f  = observed[pseudo_missing_f] - B_f[pseudo_missing_f]

Important implementation guarantee
----------------------------------
This module never requires the complete ground-truth video.  It receives only
an observed tensor whose unavailable entries are zero.  Therefore neither
truly missing values nor strict-validation values can enter a fold-specific
KPD fit or the CNN loss accidentally.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from kpd_video_core import smooth_fill
from masked_kpd import run_masked_clsa
from detail_refiner_v41 import highpass_torch


@dataclass
class CrossFitState:
    fold: int
    backbone: np.ndarray
    context_mask: np.ndarray
    target_mask: np.ndarray
    residual_context: np.ndarray
    target_residual: np.ndarray
    target_residual_mae: float
    target_residual_rms: float
    kpd_objective_final: float
    kpd_monotone: bool


def _tube_indices(mask):
    return np.flatnonzero(mask[:, :, 0].ravel())


def build_crossfit_folds(
    observed,
    train_mask,
    rank=12,
    masked_sweeps=12,
    lam_space=0.005,
    lam_time=0.05,
    ridge=1e-5,
    n_folds=5,
    seed=1234,
):
    """
    Partition observed training tubes into folds.  The `observed` tensor must
    contain measurement values only on the globally observed set; values
    outside that set are ignored and should normally be zero.
    """
    observed = np.asarray(observed, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)
    H, W, T = observed.shape
    idx = _tube_indices(train_mask)
    if len(idx) < n_folds * 4:
        raise ValueError("Too few observed tubes for requested cross-fitting.")

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    fold_ids = np.array_split(idx, n_folds)
    states = []

    print(f"Building {n_folds} cross-fitted KPD states...")
    for f, target_idx in enumerate(fold_ids, start=1):
        target2 = np.zeros(H * W, dtype=bool)
        target2[target_idx] = True
        target2 = target2.reshape(H, W)
        target_mask = np.repeat(target2[:, :, None], T, axis=2)
        target_mask &= train_mask

        context_mask = train_mask & (~target_mask)

        # Fold-specific KPD pipeline: context values only.
        obs_context = np.zeros_like(observed)
        obs_context[context_mask] = observed[context_mask]
        init_context = smooth_fill(obs_context, context_mask)

        res = run_masked_clsa(
            observed=obs_context,
            mask=context_mask,
            initial_tensor=init_context,
            rank=rank,
            outer_sweeps=masked_sweeps,
            lam_space=lam_space,
            lam_time=lam_time,
            ridge=ridge,
            seed=seed + f,
            init="warm",
            project_observed=False,
            verbose=False,
            evaluation_gt=None,
            balance_scales=True,
            check_monotonic=True,
        )
        backbone = res.backbone

        residual_context = np.zeros_like(observed)
        residual_context[context_mask] = (
            observed[context_mask] - backbone[context_mask]
        )

        # Store only the allowed pseudo-missing supervision; zero elsewhere.
        target_residual = np.zeros_like(observed)
        target_residual[target_mask] = (
            observed[target_mask] - backbone[target_mask]
        )
        tr = target_residual[target_mask]
        mae = float(np.mean(np.abs(tr)))
        rms = float(np.sqrt(np.mean(tr * tr)))
        monotone = all(bool(h["monotone_ok"]) for h in res.history)

        states.append(
            CrossFitState(
                fold=f,
                backbone=backbone,
                context_mask=context_mask,
                target_mask=target_mask,
                residual_context=residual_context,
                target_residual=target_residual,
                target_residual_mae=mae,
                target_residual_rms=rms,
                kpd_objective_final=float(res.history[-1]["objective"]),
                kpd_monotone=monotone,
            )
        )
        print(
            f"  fold {f}/{n_folds}: "
            f"context tubes={context_mask[:,:,0].mean():.5f}, "
            f"pseudo-missing tubes={target_mask[:,:,0].mean():.5f}, "
            f"target MAE/RMS={mae:.6f}/{rms:.6f}, "
            f"KPD monotone={monotone}"
        )

    return states


def _sample_patch_start(n, p, rng):
    if p >= n:
        return 0
    return int(rng.integers(0, n - p + 1))


def train_crossfit_tube_cnn(
    model,
    states,
    device,
    steps=1200,
    patch_t=16,
    patch_h=48,
    patch_w=48,
    batch_size=2,
    lr=1e-3,
    detail_weight=2.0,
    amp=True,
    print_every=100,
    seed=1234,
    smooth_l1_beta=0.02,
    magnitude_scale=0.05,
    magnitude_cap=3.0,
    residual_penalty=1e-4,
):
    """
    Train only on stored cross-fitted pseudo-missing residual targets.

    No complete ground-truth video is an argument to this function.  The
    target-dependent fitting term is evaluated only on target_mask; the weak
    residual-magnitude penalty acts on the full predicted patch.
    """
    if not states:
        raise ValueError("states must be non-empty")
    H, W, T = states[0].backbone.shape
    pt, ph, pw = min(patch_t, T), min(patch_h, H), min(patch_w, W)

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = []

    model.train()

    for step in range(1, steps + 1):
        x_batch, target_batch, target_mask_batch = [], [], []
        attempts = 0

        while len(x_batch) < batch_size:
            attempts += 1
            if attempts > batch_size * 100:
                raise RuntimeError(
                    "Unable to sample patches containing cross-fit target tubes."
                )

            state = states[int(rng.integers(0, len(states)))]

            t0 = _sample_patch_start(T, pt, rng)
            h0 = _sample_patch_start(H, ph, rng)
            w0 = _sample_patch_start(W, pw, rng)
            sl = np.s_[h0:h0+ph, w0:w0+pw, t0:t0+pt]

            target_mask = state.target_mask[sl]
            if int(target_mask.sum()) < max(8, pt * 2):
                continue

            b = state.backbone[sl]
            rvis = state.residual_context[sl]
            cmask = state.context_mask[sl]
            target_local = state.target_residual[sl]

            bt = np.transpose(b.astype(np.float32), (2,0,1))[None, ...]
            rt = np.transpose(rvis.astype(np.float32), (2,0,1))[None, ...]
            mt = np.transpose(cmask.astype(np.float32), (2,0,1))[None, ...]
            inp3 = np.concatenate([bt, rt, mt], axis=0)

            target = np.transpose(
                target_local.astype(np.float32), (2,0,1)
            )[None, ...]
            tmask = np.transpose(
                target_mask.astype(np.bool_), (2,0,1)
            )[None, ...]

            x_batch.append(inp3)
            target_batch.append(target)
            target_mask_batch.append(tmask)

        x3 = torch.from_numpy(np.stack(x_batch)).to(device)
        target = torch.from_numpy(np.stack(target_batch)).to(device)
        tmask = torch.from_numpy(np.stack(target_mask_batch)).to(device).bool()

        base = x3[:, 0:1]
        hp = highpass_torch(base)
        x = torch.cat([x3, hp], dim=1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            pr = pred[tmask]
            tg = target[tmask]

            weight = 1.0 + detail_weight * torch.clamp(
                torch.abs(tg) / magnitude_scale,
                min=0.0,
                max=magnitude_cap,
            )
            point_loss = F.smooth_l1_loss(
                pr, tg, reduction="none", beta=smooth_l1_beta
            )
            holdout_loss = (weight * point_loss).mean()

            magnitude_loss = pred.pow(2).mean()
            loss = holdout_loss + residual_penalty * magnitude_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "crossfit_target_loss": float(holdout_loss.detach().cpu()),
            "magnitude_loss": float(magnitude_loss.detach().cpu()),
        }
        history.append(row)

        if step == 1 or step % print_every == 0 or step == steps:
            print(
                f"[CrossFit step {step:04d}/{steps}] "
                f"loss={row['loss']:.6f}, "
                f"target={row['crossfit_target_loss']:.6f}"
            )

    return history
