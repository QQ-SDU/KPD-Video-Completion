#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V4.1 self-supervised detail residual refinement.

The KPD backbone already reconstructs global spatiotemporal structure well.
This module learns only the residual detail that KPD misses.

Training principle
------------------
Let B be a Masked-CLSA reconstruction. On observed entries, the true residual

    R = O - B

is known. We dynamically hide part of those known residual samples and train
a small local 3D CNN to predict them from:
    1) the KPD baseline B,
    2) the visible residual samples,
    3) the visible observation mask,
    4) a spatial high-pass map of B.

No truly missing ground-truth voxel is used in the training loss.

For stricter validation, the backbone used during training is fitted only on
a fixed training subset of the observed set. A held-out observed subset is
never used to fit that training backbone or optimize the CNN. After fixed-step
training, the same CNN is applied to a full-observation Masked-CLSA baseline.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DetailResidualCNN(nn.Module):
    """
    Small local 3D residual predictor.
    Input channels:
      0: KPD baseline
      1: visible observed residual (O - baseline)
      2: visible observation mask
      3: baseline spatial high-pass map
    """
    def __init__(self, hidden=12, max_residual=0.25, spatial_kernel=3):
        super().__init__()
        self.max_residual = float(max_residual)
        if spatial_kernel not in (3, 5):
            raise ValueError("spatial_kernel must be 3 or 5")
        # k=3 EXACTLY reproduces the published four-convolution architecture.
        # k=5 changes each kernel from (3,3,3) to (3,5,5), while keeping
        # temporal support/dilation identical. Stride=1, output size unchanged.
        k = int(spatial_kernel)
        ks = (3, k, k)
        p = (1, k // 2, k // 2)
        d2 = (2, 2, 2)
        p2 = (2, k - 1, k - 1)
        self.spatial_kernel = k
        self.c1 = nn.Conv3d(4, hidden, ks, padding=p)
        self.c2 = nn.Conv3d(hidden, hidden, ks, padding=p2, dilation=d2)
        self.c3 = nn.Conv3d(hidden, hidden, ks, padding=p)
        self.out = nn.Conv3d(hidden, 1, ks, padding=p)

        # Start from "no detail correction", so epoch 0 equals KPD baseline.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        z = F.gelu(self.c1(x))
        z = F.gelu(self.c2(z))
        z = F.gelu(self.c3(z))
        z = self.out(z)
        return self.max_residual * torch.tanh(z)


def hwt_to_torch(x: np.ndarray, device):
    """H,W,T -> 1,1,T,H,W"""
    return torch.from_numpy(
        np.transpose(x.astype(np.float32), (2, 0, 1))
    ).unsqueeze(0).unsqueeze(0).to(device)


def torch_to_hwt(x: torch.Tensor):
    """1,1,T,H,W -> H,W,T"""
    a = x.detach().float().cpu().numpy()[0, 0]
    return np.transpose(a, (1, 2, 0))


def highpass_torch(base_5d: torch.Tensor):
    """
    Spatial high-pass only. We intentionally do not remove temporal low
    frequencies here because the KPD baseline already models temporal structure.
    """
    low = F.avg_pool3d(
        base_5d, kernel_size=(1, 5, 5),
        stride=1, padding=(0, 2, 2)
    )
    return base_5d - low


def build_full_input(
    baseline_hwt: np.ndarray,
    observed_hwt: np.ndarray,
    visible_mask_hwt: np.ndarray,
    device,
):
    b = hwt_to_torch(baseline_hwt, device)
    o = hwt_to_torch(observed_hwt, device)
    m = hwt_to_torch(visible_mask_hwt.astype(np.float32), device)
    r = (o - b) * m
    hp = highpass_torch(b)
    return torch.cat([b, r, m, hp], dim=1)


def split_observed_mask(mask: np.ndarray, train_fraction=0.8, seed=4321):
    """
    Fixed train/validation split INSIDE the truly observed set.
    """
    idx = np.flatnonzero(mask.ravel())
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(len(idx) * train_fraction))

    tr = np.zeros(mask.size, dtype=bool)
    va = np.zeros(mask.size, dtype=bool)
    tr[idx[:n_train]] = True
    va[idx[n_train:]] = True
    return tr.reshape(mask.shape), va.reshape(mask.shape)


def _sample_patch_start(n, p, rng):
    if p >= n:
        return 0
    return int(rng.integers(0, n - p + 1))


def train_detail_cnn(
    model: nn.Module,
    baseline_train: np.ndarray,
    gt: np.ndarray,
    train_obs_mask: np.ndarray,
    val_obs_mask: np.ndarray,
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
    Dynamic self-supervised holdout training.

    gt is accessed only at positions where train_obs_mask==True to construct
    residual targets. val_obs_mask is used only for post-training validation,
    not to select a checkpoint.
    """
    H, W, T = gt.shape
    pt, ph, pw = min(patch_t, T), min(patch_h, H), min(patch_w, W)

    observed_train = np.zeros_like(gt)
    observed_train[train_obs_mask] = gt[train_obs_mask]
    residual_known = np.zeros_like(gt)
    residual_known[train_obs_mask] = gt[train_obs_mask] - baseline_train[train_obs_mask]

    # Fail loudly if the caller accidentally passed an observation-clamped
    # KPD reconstruction. This was the failure mode in V4.1: the targets
    # became identically zero and the zero-initialized output layer had
    # exactly zero gradient for all training steps.
    rr = residual_known[train_obs_mask]
    residual_rms = float(np.sqrt(np.mean(rr * rr))) if rr.size else 0.0
    residual_mae = float(np.mean(np.abs(rr))) if rr.size else 0.0
    print(f"Training residual target MAE/RMS: {residual_mae:.6f} / {residual_rms:.6f}")
    if residual_rms < 1e-6:
        raise RuntimeError(
            "Training residual targets are effectively zero. "
            "Use the RAW KPD factor reconstruction, not a reconstruction "
            "with observed entries projected back to ground truth."
        )

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    model.train()

    for step in range(1, steps + 1):
        x_batch = []
        target_batch = []
        hold_batch = []

        attempts = 0
        while len(x_batch) < batch_size:
            attempts += 1
            if attempts > batch_size * 50:
                raise RuntimeError("Could not sample patches with enough observed targets.")

            t0 = _sample_patch_start(T, pt, rng)
            h0 = _sample_patch_start(H, ph, rng)
            w0 = _sample_patch_start(W, pw, rng)
            sl = np.s_[h0:h0+ph, w0:w0+pw, t0:t0+pt]

            m_obs = train_obs_mask[sl]
            idx = np.flatnonzero(m_obs.ravel())
            if len(idx) < 32:
                continue

            rng.shuffle(idx)
            n_hold = max(16, int(round(len(idx) * pseudo_holdout)))
            n_hold = min(n_hold, len(idx) - 1)

            hold = np.zeros(m_obs.size, dtype=bool)
            hold[idx[:n_hold]] = True
            hold = hold.reshape(m_obs.shape)
            visible = m_obs & (~hold)

            b = baseline_train[sl]
            r_visible = np.zeros_like(b)
            r_visible[visible] = residual_known[sl][visible]

            # HWT -> 1,T,H,W for each channel, then C,T,H,W
            bt = np.transpose(b.astype(np.float32), (2, 0, 1))[None, ...]
            rt = np.transpose(r_visible.astype(np.float32), (2, 0, 1))[None, ...]
            mt = np.transpose(visible.astype(np.float32), (2, 0, 1))[None, ...]

            # Spatial high-pass computed in numpy for patch training.
            # Use a simple 5x5 box via torch after batching, so put placeholder now.
            inp3 = np.concatenate([bt, rt, mt], axis=0)
            target = np.transpose(
                (gt[sl] - b).astype(np.float32), (2, 0, 1)
            )[None, ...]
            hold_t = np.transpose(hold, (2, 0, 1))[None, ...]

            x_batch.append(inp3)
            target_batch.append(target)
            hold_batch.append(hold_t)

        x3 = torch.from_numpy(np.stack(x_batch)).to(device)  # B,3,T,H,W
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

            # Emphasize locations where the KPD backbone is wrong by a larger amount.
            # These locations correspond more strongly to the missing-detail problem.
            weight = 1.0 + detail_weight * torch.clamp(
                torch.abs(tg) / 0.05, min=0.0, max=3.0
            )
            point_loss = F.smooth_l1_loss(pr, tg, reduction="none", beta=0.02)
            holdout_loss = (weight * point_loss).mean()

            # Weak stabilizer only; do not force the residual to be smooth.
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
                f"[Detail step {step:04d}/{steps}] "
                f"loss={row['loss']:.6f}, holdout={row['holdout_loss']:.6f}"
            )

    return history


@torch.no_grad()
def infer_detail(
    model: nn.Module,
    baseline_hwt: np.ndarray,
    observed_hwt: np.ndarray,
    visible_mask_hwt: np.ndarray,
    device,
):
    """
    Full-volume inference. If CUDA memory is tight, automatically falls back
    to temporal chunks with overlap.
    """
    model.eval()
    try:
        x = build_full_input(baseline_hwt, observed_hwt, visible_mask_hwt, device)
        pred = model(x)
        return torch_to_hwt(pred)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or device.type != "cuda":
            raise
        torch.cuda.empty_cache()
        print("CUDA OOM during full-volume inference; using temporal chunks.")

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
        x = build_full_input(
            baseline_hwt[sl], observed_hwt[sl], visible_mask_hwt[sl], device
        )
        pred = model(x)
        r = torch_to_hwt(pred)
        acc[sl] += r
        wt[sl] += 1.0

    return acc / np.maximum(wt, 1.0)
