#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from kpd_video_core import (
    EPS, psnr, masked_psnr, mean_ssim
)


@dataclass
class Rank1Result2D:
    a: np.ndarray
    b: np.ndarray
    error: float
    sweeps: int


@dataclass
class KPDModel2D:
    components: List[Rank1Result2D]
    reconstruction: np.ndarray
    residual: np.ndarray
    residual_norms: List[float]


def outer2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.outer(a, b)


def clsa_rank1_2d(
    matrix: np.ndarray,
    max_sweeps: int = 40,
    tol: float = 1e-5,
    restarts: int = 2,
    seed: int = 0,
) -> Rank1Result2D:
    """
    Frame-wise d=2 CLSA-style rank-1 approximation:
        X ~= a \\otimes b == a b^T.

    Cyclic least-squares:
        a <- X b / ||b||^2
        b <- X^T a / ||a||^2

    This is intentionally the d=2 analogue of the joint d=3 solver,
    so the comparison isolates the value of joint temporal modeling.
    """
    X = np.asarray(matrix, dtype=np.float64)
    H, W = X.shape
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(max(1, restarts)):
        b = rng.normal(size=W)
        b /= np.linalg.norm(b) + EPS
        a = X @ b / (b @ b + EPS)
        prev = outer2(a, b)
        sweeps_done = 0

        for sweep in range(1, max_sweeps + 1):
            sweeps_done = sweep

            a = (X @ b) / (b @ b + EPS)
            b = (X.T @ a) / (a @ a + EPS)

            nb = np.linalg.norm(b) + EPS
            b /= nb
            a *= nb

            cur = outer2(a, b)
            change = np.linalg.norm(cur - prev) / (np.linalg.norm(prev) + EPS)
            prev = cur
            if change < tol:
                break

        err = float(np.linalg.norm(X - prev))
        rr = Rank1Result2D(a.copy(), b.copy(), err, sweeps_done)
        if best is None or rr.error < best.error:
            best = rr

    return best


def finite_sum_clsa_2d(
    matrix: np.ndarray,
    rank: int,
    max_sweeps: int = 40,
    tol: float = 1e-5,
    restarts: int = 2,
    seed: int = 0,
) -> KPDModel2D:
    X = np.asarray(matrix, dtype=np.float64)
    residual = X.copy()
    recon = np.zeros_like(X)
    components = []
    residual_norms = [float(np.linalg.norm(residual))]

    for k in range(rank):
        comp = clsa_rank1_2d(
            residual,
            max_sweeps=max_sweeps,
            tol=tol,
            restarts=restarts,
            seed=seed + 1009 * k,
        )
        term = outer2(comp.a, comp.b)
        recon += term
        residual -= term
        components.append(comp)
        residual_norms.append(float(np.linalg.norm(residual)))

    return KPDModel2D(components, recon, residual, residual_norms)


def run_framewise_projection_completion(
    gt: np.ndarray,
    mask: np.ndarray,
    initial_tensor: np.ndarray,
    rank: int = 5,
    outer_iters: int = 10,
    max_sweeps: int = 40,
    tol: float = 1e-5,
    restarts: int = 2,
    seed: int = 1234,
    verbose: bool = True,
):
    """
    Same projection-style completion as the joint 3D method, except each
    frame is decomposed independently:
        X_t ~= sum_k a_{t,k} \\otimes b_{t,k}.

    The observed mask and initialization are supplied externally, ensuring
    an exactly matched comparison with the joint 3D method.
    """
    gt = np.asarray(gt, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    x = np.asarray(initial_tensor, dtype=np.float64).copy()
    observed = np.zeros_like(gt)
    observed[mask] = gt[mask]
    miss = ~mask
    H, W, T = gt.shape
    history = []

    if verbose:
        print("\n================ Frame-wise 2D CLSA-KPD ================")
        print(f"Video shape: {gt.shape}; rank/frame={rank}")
        print(
            f"Initial PSNR(all)={psnr(gt,x):.3f} dB, "
            f"PSNR(missing)={masked_psnr(gt,x,miss):.3f} dB, "
            f"SSIM={mean_ssim(gt,x):.4f}"
        )

    for it in range(1, outer_iters + 1):
        tic = time.time()
        recon = np.zeros_like(x)
        rel_resids = []

        for t in range(T):
            model = finite_sum_clsa_2d(
                x[:, :, t],
                rank=rank,
                max_sweeps=max_sweeps,
                tol=tol,
                restarts=restarts,
                seed=seed + 100000 * it + 1000 * t,
            )
            recon[:, :, t] = model.reconstruction
            rel_resids.append(
                float(np.linalg.norm(model.residual) /
                      (np.linalg.norm(x[:, :, t]) + EPS))
            )

        low = np.clip(recon, 0.0, 1.0)
        xnew = low.copy()
        xnew[mask] = observed[mask]

        delta = float(np.linalg.norm(xnew - x) / (np.linalg.norm(x) + EPS))
        x = xnew

        row = {
            "outer_iter": it,
            "delta": delta,
            "psnr_all": psnr(gt, x),
            "psnr_missing": masked_psnr(gt, x, miss),
            "ssim_all": mean_ssim(gt, x),
            "seconds": time.time() - tic,
            "mean_frame_kpd_relative_residual": float(np.mean(rel_resids)),
        }
        history.append(row)

        if verbose:
            print(
                f"[2D outer {it:02d}/{outer_iters}] "
                f"PSNR(missing)={row['psnr_missing']:.3f} dB, "
                f"PSNR(all)={row['psnr_all']:.3f} dB, "
                f"SSIM={row['ssim_all']:.4f}, "
                f"delta={row['delta']:.3e}, "
                f"time={row['seconds']:.2f}s"
            )

        if delta < tol:
            break

    return {
        "recovered": x,
        "history": history,
    }


def temporal_gradient_rmse(gt: np.ndarray, rec: np.ndarray, mask: np.ndarray) -> float:
    """
    RMSE between temporal finite differences of GT and reconstruction.
    Evaluation is restricted to temporal edges where at least one endpoint
    is missing, reducing the trivial contribution from exactly observed data.
    Lower is better.
    """
    if gt.shape[2] < 2:
        return float("nan")

    d_gt = gt[:, :, 1:] - gt[:, :, :-1]
    d_rec = rec[:, :, 1:] - rec[:, :, :-1]
    miss_pair = (~mask[:, :, 1:]) | (~mask[:, :, :-1])

    if not np.any(miss_pair):
        return 0.0

    return float(np.sqrt(np.mean((d_gt[miss_pair] - d_rec[miss_pair]) ** 2)))


def summarize_method(gt: np.ndarray, rec: np.ndarray, mask: np.ndarray) -> dict:
    miss = ~mask
    return {
        "psnr_all": psnr(gt, rec),
        "psnr_missing": masked_psnr(gt, rec, miss),
        "ssim_all": mean_ssim(gt, rec),
        "temporal_gradient_rmse_missing": temporal_gradient_rmse(gt, rec, mask),
    }
