#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation-only residual-training helpers for V7.

This file deliberately does not modify the main V7 implementation.
It provides two controlled baselines:

1) Direct residual CNN:
   the KPD backbone is fitted with all training-observed tubes, so the
   residual targets are in-sample with respect to KPD. Whole observed tubes
   are hidden only from the CNN input. This is the correct ablation for
   testing whether KPD-level cross-fitting is necessary.

2) Network-input ablation:
   all variants use the same 4-channel CNN architecture. Removed channels are
   set to zero rather than changing the number of network parameters.

Input modes:
    B      = [backbone, 0, 0, 0]
    BR     = [backbone, visible residual, 0, 0]
    BRM    = [backbone, visible residual, mask, 0]
    FULL   = [backbone, visible residual, mask, spatial high-pass]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from detail_refiner_v41 import highpass_torch, hwt_to_torch, torch_to_hwt


VALID_INPUT_MODES = ("B", "BR", "BRM", "FULL")


def _check_mode(mode: str) -> str:
    mode = str(mode).upper()
    if mode not in VALID_INPUT_MODES:
        raise ValueError(f"Unknown input mode {mode!r}; choose from {VALID_INPUT_MODES}")
    return mode


def _assemble_input(base, residual_visible, mask_visible, mode: str):
    """Assemble a 4-channel tensor while keeping architecture fixed."""
    mode = _check_mode(mode)
    z = torch.zeros_like(base)
    hp = highpass_torch(base)
    if mode == "B":
        return torch.cat([base, z, z, z], dim=1)
    if mode == "BR":
        return torch.cat([base, residual_visible, z, z], dim=1)
    if mode == "BRM":
        return torch.cat([base, residual_visible, mask_visible, z], dim=1)
    return torch.cat([base, residual_visible, mask_visible, hp], dim=1)


def _sample_patch_start(n, p, rng):
    if p >= n:
        return 0
    return int(rng.integers(0, n - p + 1))


def train_crossfit_ablation(
    model,
    states,
    device,
    *,
    input_mode="FULL",
    steps=1200,
    patch_t=16,
    patch_h=48,
    patch_w=48,
    batch_size=2,
    lr=1e-3,
    detail_weight=2.0,
    amp=True,
    print_every=200,
    seed=1234,
    smooth_l1_beta=0.02,
    magnitude_scale=0.05,
    magnitude_cap=3.0,
    residual_penalty=1e-4,
    checkpoint_steps=None,
    checkpoint_callback=None,
):
    """Train from true cross-fitted pseudo-missing KPD residuals.

    ``checkpoint_callback(step, model)`` may be supplied for the training-step
    sensitivity study. It is called after optimizer updates at requested steps.
    """
    mode = _check_mode(input_mode)
    if not states:
        raise ValueError("states must be non-empty")
    H, W, T = states[0].backbone.shape
    pt, ph, pw = min(patch_t, T), min(patch_h, H), min(patch_w, W)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = []
    checkpoints = set(int(x) for x in (checkpoint_steps or []))
    model.train()

    for step in range(1, int(steps) + 1):
        x_batch, target_batch, target_mask_batch = [], [], []
        attempts = 0
        while len(x_batch) < batch_size:
            attempts += 1
            if attempts > batch_size * 100:
                raise RuntimeError("Unable to sample patches containing cross-fit target tubes.")
            state = states[int(rng.integers(0, len(states)))]
            t0 = _sample_patch_start(T, pt, rng)
            h0 = _sample_patch_start(H, ph, rng)
            w0 = _sample_patch_start(W, pw, rng)
            sl = np.s_[h0:h0+ph, w0:w0+pw, t0:t0+pt]
            tmask_np = state.target_mask[sl]
            if int(tmask_np.sum()) < max(8, pt * 2):
                continue

            b = state.backbone[sl]
            rvis = state.residual_context[sl]
            cmask = state.context_mask[sl]
            target_local = state.target_residual[sl]

            bt = np.transpose(b.astype(np.float32), (2, 0, 1))[None, ...]
            rt = np.transpose(rvis.astype(np.float32), (2, 0, 1))[None, ...]
            mt = np.transpose(cmask.astype(np.float32), (2, 0, 1))[None, ...]
            target = np.transpose(target_local.astype(np.float32), (2, 0, 1))[None, ...]
            tmask = np.transpose(tmask_np.astype(np.bool_), (2, 0, 1))[None, ...]
            x_batch.append(np.concatenate([bt, rt, mt], axis=0))
            target_batch.append(target)
            target_mask_batch.append(tmask)

        x3 = torch.from_numpy(np.stack(x_batch)).to(device)
        target = torch.from_numpy(np.stack(target_batch)).to(device)
        tmask = torch.from_numpy(np.stack(target_mask_batch)).to(device).bool()
        base, rvis, mask = x3[:, 0:1], x3[:, 1:2], x3[:, 2:3]
        x = _assemble_input(base, rvis, mask, mode)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            pr, tg = pred[tmask], target[tmask]
            weight = 1.0 + detail_weight * torch.clamp(
                torch.abs(tg) / magnitude_scale, min=0.0, max=magnitude_cap
            )
            point_loss = F.smooth_l1_loss(
                pr, tg, reduction="none", beta=smooth_l1_beta
            )
            target_loss = (weight * point_loss).mean()
            magnitude_loss = pred.pow(2).mean()
            loss = target_loss + residual_penalty * magnitude_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "target_loss": float(target_loss.detach().cpu()),
            "magnitude_loss": float(magnitude_loss.detach().cpu()),
        }
        history.append(row)
        if step == 1 or step % max(1, print_every) == 0 or step == steps:
            print(f"[CrossFit/{mode} {step:04d}/{steps}] loss={row['loss']:.6f}")

        if step in checkpoints and checkpoint_callback is not None:
            checkpoint_callback(step, model)
            model.train()

    return history


def train_direct_tube_ablation(
    model,
    baseline_train,
    observed,
    train_mask,
    device,
    *,
    input_mode="FULL",
    steps=1200,
    patch_t=16,
    patch_h=48,
    patch_w=48,
    batch_size=2,
    pseudo_holdout=0.20,
    lr=1e-3,
    detail_weight=2.0,
    amp=True,
    print_every=200,
    seed=1234,
    smooth_l1_beta=0.02,
    magnitude_scale=0.05,
    magnitude_cap=3.0,
    residual_penalty=1e-4,
):
    """Direct in-sample-residual baseline, with tube-wise CNN holdout.

    Crucial distinction from cross-fitting: ``baseline_train`` has already been
    fitted using *all* ``train_mask`` tubes, including the tubes used as CNN
    pseudo-targets. Thus these targets are in-sample KPD residuals.

    The function receives only the observed tensor; true missing values are not
    needed for optimization.
    """
    mode = _check_mode(input_mode)
    observed = np.asarray(observed, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)
    H, W, T = observed.shape
    pt, ph, pw = min(patch_t, T), min(patch_h, H), min(patch_w, W)

    residual_known = np.zeros_like(observed)
    residual_known[train_mask] = observed[train_mask] - baseline_train[train_mask]
    rr = residual_known[train_mask]
    if rr.size == 0:
        raise RuntimeError("No training observations.")
    print(
        "Direct in-sample KPD residual MAE/RMS: "
        f"{np.mean(np.abs(rr)):.6f}/{np.sqrt(np.mean(rr*rr)):.6f}"
    )

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = []
    model.train()

    for step in range(1, int(steps) + 1):
        x_batch, target_batch, hold_batch = [], [], []
        attempts = 0
        while len(x_batch) < batch_size:
            attempts += 1
            if attempts > batch_size * 100:
                raise RuntimeError("Could not sample enough observed tubes for direct ablation.")
            t0 = _sample_patch_start(T, pt, rng)
            h0 = _sample_patch_start(H, ph, rng)
            w0 = _sample_patch_start(W, pw, rng)
            sl = np.s_[h0:h0+ph, w0:w0+pw, t0:t0+pt]

            # Tube-wise mask: use one time slice to identify observed spatial tubes.
            m2 = train_mask[h0:h0+ph, w0:w0+pw, t0]
            tube_idx = np.flatnonzero(m2.ravel())
            if len(tube_idx) < 8:
                continue
            rng.shuffle(tube_idx)
            n_hold = max(2, int(round(len(tube_idx) * pseudo_holdout)))
            n_hold = min(n_hold, len(tube_idx) - 1)
            hold2 = np.zeros(m2.size, dtype=bool)
            hold2[tube_idx[:n_hold]] = True
            hold2 = hold2.reshape(m2.shape)
            vis2 = m2 & (~hold2)
            hold = np.repeat(hold2[:, :, None], pt, axis=2)
            visible = np.repeat(vis2[:, :, None], pt, axis=2)

            b = baseline_train[sl]
            rpatch = residual_known[sl]
            rvis = np.zeros_like(b)
            rvis[visible] = rpatch[visible]
            bt = np.transpose(b.astype(np.float32), (2, 0, 1))[None, ...]
            rt = np.transpose(rvis.astype(np.float32), (2, 0, 1))[None, ...]
            mt = np.transpose(visible.astype(np.float32), (2, 0, 1))[None, ...]
            target = np.transpose(rpatch.astype(np.float32), (2, 0, 1))[None, ...]
            hold_t = np.transpose(hold.astype(np.bool_), (2, 0, 1))[None, ...]
            x_batch.append(np.concatenate([bt, rt, mt], axis=0))
            target_batch.append(target)
            hold_batch.append(hold_t)

        x3 = torch.from_numpy(np.stack(x_batch)).to(device)
        target = torch.from_numpy(np.stack(target_batch)).to(device)
        hold = torch.from_numpy(np.stack(hold_batch)).to(device).bool()
        base, rvis, mask = x3[:, 0:1], x3[:, 1:2], x3[:, 2:3]
        x = _assemble_input(base, rvis, mask, mode)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            pr, tg = pred[hold], target[hold]
            weight = 1.0 + detail_weight * torch.clamp(
                torch.abs(tg) / magnitude_scale, min=0.0, max=magnitude_cap
            )
            point_loss = F.smooth_l1_loss(
                pr, tg, reduction="none", beta=smooth_l1_beta
            )
            target_loss = (weight * point_loss).mean()
            magnitude_loss = pred.pow(2).mean()
            loss = target_loss + residual_penalty * magnitude_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()
        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "target_loss": float(target_loss.detach().cpu()),
            "magnitude_loss": float(magnitude_loss.detach().cpu()),
        }
        history.append(row)
        if step == 1 or step % max(1, print_every) == 0 or step == steps:
            print(f"[Direct/{mode} {step:04d}/{steps}] loss={row['loss']:.6f}")
    return history


@torch.no_grad()
def infer_ablation(model, baseline_hwt, observed_hwt, visible_mask_hwt, device, input_mode="FULL"):
    """Inference with the same zero-channel ablation convention used in training."""
    mode = _check_mode(input_mode)
    model.eval()

    def _one(b_np, o_np, m_np):
        b = hwt_to_torch(b_np, device)
        o = hwt_to_torch(o_np, device)
        m = hwt_to_torch(m_np.astype(np.float32), device)
        r = (o - b) * m
        x = _assemble_input(b, r, m, mode)
        return torch_to_hwt(model(x))

    try:
        return _one(baseline_hwt, observed_hwt, visible_mask_hwt)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or device.type != "cuda":
            raise
        torch.cuda.empty_cache()
        print("CUDA OOM during ablation inference; using temporal chunks.")

    H, W, T = baseline_hwt.shape
    chunk = min(20, T)
    overlap = min(4, max(0, chunk // 4))
    stride = max(1, chunk - overlap)
    acc = np.zeros_like(baseline_hwt, dtype=np.float64)
    wt = np.zeros_like(baseline_hwt, dtype=np.float64)
    starts = list(range(0, max(1, T - chunk + 1), stride))
    if starts[-1] != max(0, T - chunk):
        starts.append(max(0, T - chunk))
    for t0 in starts:
        sl = np.s_[:, :, t0:t0+chunk]
        r = _one(baseline_hwt[sl], observed_hwt[sl], visible_mask_hwt[sl])
        acc[sl] += r
        wt[sl] += 1.0
    return acc / np.maximum(wt, 1.0)
