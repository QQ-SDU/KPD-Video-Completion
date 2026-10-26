#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tube-wise self-supervised residual refinement.

For tube-wise completion, the observation mask is constant along time:
a spatial location (h,w) is either observed for ALL frames or missing
for ALL frames.

Accordingly, self-supervised masking must also hide whole mode-3 tubes,
not random individual voxels. This keeps the training task aligned with
the actual test degradation.
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detail_refiner_v41 import (
    DetailResidualCNN,
    hwt_to_torch,
    torch_to_hwt,
    highpass_torch,
    infer_detail,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_tube_mask(shape, sampling_rate, seed):
    """
    Choose exactly round(SR * H * W) observed spatial tubes.
    Returns H x W x T bool mask constant along time.
    """
    H, W, T = shape
    n_tubes = H * W
    k = int(round(float(sampling_rate) * n_tubes))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_tubes, size=k, replace=False)
    m2 = np.zeros(n_tubes, dtype=bool)
    m2[idx] = True
    m2 = m2.reshape(H, W)
    return np.repeat(m2[:, :, None], T, axis=2)


def split_observed_tubes(mask, train_fraction=0.8, seed=1234):
    """
    Split the observed spatial tubes into disjoint train/strict-val tube sets.
    """
    H, W, T = mask.shape
    tube_mask = mask[:, :, 0].copy()
    idx = np.flatnonzero(tube_mask.ravel())
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(len(idx) * train_fraction))

    tr2 = np.zeros(H * W, dtype=bool)
    va2 = np.zeros(H * W, dtype=bool)
    tr2[idx[:n_train]] = True
    va2[idx[n_train:]] = True
    tr2 = tr2.reshape(H, W)
    va2 = va2.reshape(H, W)

    return (
        np.repeat(tr2[:, :, None], T, axis=2),
        np.repeat(va2[:, :, None], T, axis=2),
    )


def split_tubes(mask, keep_fraction, seed):
    """
    Nested split inside an already tube-wise mask.
    """
    return split_observed_tubes(mask, train_fraction=keep_fraction, seed=seed)


def _sample_patch_start(n, p, rng):
    if p >= n:
        return 0
    return int(rng.integers(0, n - p + 1))


def train_detail_cnn_tubewise(
    model,
    baseline_train,
    gt,
    train_obs_mask,
    strict_val_mask,
    device,
    steps=1200,
    patch_t=16,
    patch_h=48,
    patch_w=48,
    batch_size=2,
    pseudo_holdout=0.25,
    lr=1e-3,
    detail_weight=2.0,
    amp=True,
    print_every=100,
    seed=1234,
):
    """
    Dynamic pseudo-holdout training where entire spatial tubes are hidden.

    In a sampled HxWxT patch, a pseudo-held spatial location is withheld
    across the full temporal extent of that patch.

    Truly missing GT and strict validation tubes never enter the loss.
    """
    H, W, T = gt.shape
    pt, ph, pw = min(patch_t, T), min(patch_h, H), min(patch_w, W)

    residual_known = np.zeros_like(gt)
    residual_known[train_obs_mask] = gt[train_obs_mask] - baseline_train[train_obs_mask]

    rr = residual_known[train_obs_mask]
    residual_rms = float(np.sqrt(np.mean(rr * rr))) if rr.size else 0.0
    residual_mae = float(np.mean(np.abs(rr))) if rr.size else 0.0
    print(f"Training tube residual target MAE/RMS: {residual_mae:.6f} / {residual_rms:.6f}")
    if residual_rms < 1e-6:
        raise RuntimeError(
            "Tube residual targets are effectively zero. "
            "Use raw KPD factor prediction, not observed-projected reconstruction."
        )

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = []

    for step in range(1, steps + 1):
        x_batch, target_batch, hold_batch = [], [], []
        attempts = 0

        while len(x_batch) < batch_size:
            attempts += 1
            if attempts > batch_size * 80:
                raise RuntimeError("Could not sample enough observed tubes in patches.")

            t0 = _sample_patch_start(T, pt, rng)
            h0 = _sample_patch_start(H, ph, rng)
            w0 = _sample_patch_start(W, pw, rng)
            sl = np.s_[h0:h0+ph, w0:w0+pw, t0:t0+pt]

            # Because mask is tube-wise, one temporal slice is sufficient
            # to identify observed spatial tubes in this patch.
            m2 = train_obs_mask[h0:h0+ph, w0:w0+pw, t0]
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
            r_visible = np.zeros_like(b)
            r_patch = residual_known[sl]
            r_visible[visible] = r_patch[visible]

            bt = np.transpose(b.astype(np.float32), (2, 0, 1))[None, ...]
            rt = np.transpose(r_visible.astype(np.float32), (2, 0, 1))[None, ...]
            mt = np.transpose(visible.astype(np.float32), (2, 0, 1))[None, ...]
            inp3 = np.concatenate([bt, rt, mt], axis=0)

            target = np.transpose(
                (gt[sl] - b).astype(np.float32), (2, 0, 1)
            )[None, ...]
            hold_t = np.transpose(hold, (2, 0, 1))[None, ...]

            x_batch.append(inp3)
            target_batch.append(target)
            hold_batch.append(hold_t)

        x3 = torch.from_numpy(np.stack(x_batch)).to(device)
        target = torch.from_numpy(np.stack(target_batch)).to(device)
        hold = torch.from_numpy(np.stack(hold_batch)).to(device).bool()

        base = x3[:, 0:1]
        hp = highpass_torch(base)
        x = torch.cat([x3, hp], dim=1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            pr = pred[hold]
            tg = target[hold]

            weight = 1.0 + detail_weight * torch.clamp(
                torch.abs(tg) / 0.05, min=0.0, max=3.0
            )
            point_loss = F.smooth_l1_loss(
                pr, tg, reduction="none", beta=0.02
            )
            holdout_loss = (weight * point_loss).mean()
            magnitude_loss = pred.pow(2).mean()
            loss = holdout_loss + 1e-4 * magnitude_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "holdout_loss": float(holdout_loss.detach().cpu()),
            "magnitude_loss": float(magnitude_loss.detach().cpu()),
        }
        history.append(row)

        if step == 1 or step % print_every == 0 or step == steps:
            print(
                f"[Tube detail step {step:04d}/{steps}] "
                f"loss={row['loss']:.6f}, holdout={row['holdout_loss']:.6f}"
            )

    return history
