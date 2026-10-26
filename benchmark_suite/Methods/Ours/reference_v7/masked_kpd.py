#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Theory-aligned masked Kronecker-structured video completion.

The optimizer solves the observed-entry regularized objective

    1/2 || M * (O - sum_r a_r o b_r o c_r) ||_F^2
    + lam_space/2 * sum_r (||D a_r||^2 + ||D b_r||^2)
    + lam_time /2 * sum_r ||D c_r||^2
    + ridge    /2 * sum_r (||a_r||^2 + ||b_r||^2 + ||c_r||^2).

Two implementation details are important for the manuscript theory:
1) fold-specific calls can receive only the observed tensor (zeros elsewhere),
   so pseudo-missing / truly missing ground truth cannot enter the optimizer;
2) factor scale balancing is not heuristic.  It minimizes the regularizer
   over the equivalent scaling orbit alpha*beta*gamma=1, preserving the
   rank-one outer product and therefore never increasing the full objective.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
import numpy as np

try:
    from scipy.linalg import solve_banded
except Exception:  # pragma: no cover - dense fallback remains available
    solve_banded = None

from kpd_video_core import EPS, psnr, masked_psnr, mean_ssim


def _diff_laplacian(n: int) -> np.ndarray:
    """D^T D for first differences (dense form, used only by fallback/tests)."""
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)
    L = np.zeros((n, n), dtype=np.float64)
    diag = np.full(n, 2.0)
    diag[0] = diag[-1] = 1.0
    np.fill_diagonal(L, diag)
    idx = np.arange(n - 1)
    L[idx, idx + 1] = -1.0
    L[idx + 1, idx] = -1.0
    return L


def _solve_smoothed(diag_h: np.ndarray, rhs: np.ndarray, lam: float, ridge: float) -> np.ndarray:
    """
    Solve (diag(diag_h) + lam*D'D + ridge*I)x = rhs.

    For first differences the matrix is SPD tridiagonal.  The primary path
    uses scipy.linalg.solve_banded, giving O(n) storage/arithmetic up to the
    small fixed bandwidth.  A dense fallback is retained for portability.
    """
    diag_h = np.asarray(diag_h, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    n = len(rhs)
    main = diag_h + float(ridge)
    if lam > 0 and n > 1:
        lap_diag = np.full(n, 2.0, dtype=np.float64)
        lap_diag[0] = lap_diag[-1] = 1.0
        main = main + float(lam) * lap_diag
        off = np.full(n - 1, -float(lam), dtype=np.float64)
    else:
        off = np.zeros(max(0, n - 1), dtype=np.float64)

    if solve_banded is not None and n > 1:
        ab = np.zeros((3, n), dtype=np.float64)
        ab[0, 1:] = off
        ab[1, :] = main
        ab[2, :-1] = off
        try:
            return solve_banded((1, 1), ab, rhs, check_finite=False)
        except Exception:
            pass

    A = np.diag(main)
    if n > 1:
        idx = np.arange(n - 1)
        A[idx, idx + 1] = off
        A[idx + 1, idx] = off
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, rhs, rcond=None)[0]


def _outer3(a, b, c):
    return np.einsum("i,j,k->ijk", a, b, c, optimize=True)


def _normalize_component_for_init(a, b, c):
    """Initialization-only normalization; it is not used between exact updates."""
    nb = np.linalg.norm(b) + EPS
    nc = np.linalg.norm(c) + EPS
    return a * nb * nc, b / nb, c / nc


def _factor_regularizer_cost(x: np.ndarray, lam: float, ridge: float) -> float:
    smooth = float(np.sum(np.diff(x) ** 2)) if len(x) > 1 else 0.0
    return float(lam) * smooth + float(ridge) * float(np.dot(x, x))


def _regularizer_optimal_balance(a, b, c, lam_space, lam_time, ridge):
    """
    Preserve a o b o c while minimizing the factor regularizer over positive
    rescalings a<-alpha a, b<-beta b, c<-gamma c with alpha*beta*gamma=1.

    If q_a,q_b,q_c are the three current quadratic regularizer contributions,
    the optimum satisfies q_a alpha^2 = q_b beta^2 = q_c gamma^2.
    Therefore the full objective cannot increase under this balancing step.
    """
    qa = _factor_regularizer_cost(a, lam_space, ridge)
    qb = _factor_regularizer_cost(b, lam_space, ridge)
    qc = _factor_regularizer_cost(c, lam_time, ridge)
    if min(qa, qb, qc) <= EPS:
        return a, b, c

    logg = (np.log(qa) + np.log(qb) + np.log(qc)) / 3.0
    alpha = float(np.exp(0.5 * (logg - np.log(qa))))
    beta = float(np.exp(0.5 * (logg - np.log(qb))))
    gamma = float(np.exp(0.5 * (logg - np.log(qc))))
    # Numerically alpha*beta*gamma is one up to roundoff.
    return alpha * a, beta * b, gamma * c


def reconstruct(A, B, C):
    return np.einsum("ir,jr,kr->ijk", A, B, C, optimize=True)


def masked_objective(observed, mask, A, B, C, lam_space, lam_time, ridge) -> float:
    """Exact objective used in the manuscript, evaluated only on observed entries."""
    observed = np.asarray(observed, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    rec = reconstruct(A, B, C)
    diff = rec[mask] - observed[mask]
    val = 0.5 * float(np.dot(diff, diff))
    for r in range(A.shape[1]):
        val += 0.5 * _factor_regularizer_cost(A[:, r], lam_space, ridge)
        val += 0.5 * _factor_regularizer_cost(B[:, r], lam_space, ridge)
        val += 0.5 * _factor_regularizer_cost(C[:, r], lam_time, ridge)
    return float(val)


@dataclass
class MaskedKPDResult:
    reconstruction: np.ndarray          # data-consistent bounded output if requested
    raw_reconstruction: np.ndarray      # unconstrained sum_r a_r o b_r o c_r
    backbone: np.ndarray                # clip(raw_reconstruction,0,1)
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    history: list


def initialize_factors_from_complete_kpd(
    initial_tensor: np.ndarray,
    rank: int,
    sweeps: int = 25,
    restarts: int = 1,
    seed: int = 1234,
):
    """Warm start from the complete-tensor CLSA routine used in earlier code."""
    from kpd_video_core import finite_sum_clsa
    model = finite_sum_clsa(
        initial_tensor,
        rank=rank,
        max_sweeps=sweeps,
        tol=1e-5,
        restarts=restarts,
        seed=seed,
        verbose=False,
    )
    H, W, T = initial_tensor.shape
    A = np.zeros((H, rank))
    B = np.zeros((W, rank))
    C = np.zeros((T, rank))
    for r, comp in enumerate(model.components):
        a, b, c = _normalize_component_for_init(
            comp.a.copy(), comp.b.copy(), comp.c.copy()
        )
        A[:, r] = a
        B[:, r] = b
        C[:, r] = c
    return A, B, C


def random_factors(shape, rank, seed=1234):
    H, W, T = shape
    rng = np.random.default_rng(seed)
    A = 0.1 * rng.normal(size=(H, rank))
    B = rng.normal(size=(W, rank))
    C = rng.normal(size=(T, rank))
    for r in range(rank):
        A[:, r], B[:, r], C[:, r] = _normalize_component_for_init(
            A[:, r], B[:, r], C[:, r]
        )
    return A, B, C


def run_masked_clsa(
    observed: np.ndarray,
    mask: np.ndarray,
    initial_tensor: np.ndarray,
    rank: int = 5,
    outer_sweeps: int = 30,
    lam_space: float = 0.0,
    lam_time: float = 0.0,
    ridge: float = 1e-6,
    seed: int = 1234,
    init: str = "warm",
    project_observed: bool = True,
    verbose: bool = True,
    evaluation_gt: Optional[np.ndarray] = None,
    balance_scales: bool = True,
    check_monotonic: bool = True,
    monotonic_tol: float = 1e-9,
):
    """
    Cyclic exact block least-squares directly on observed entries.

    `observed` may contain arbitrary values outside `mask`; they are ignored.
    Cross-fitting should pass a tensor that contains only the fold context
    measurements.  `evaluation_gt` is optional and is never used by the
    optimizer; it exists only for experiment-time diagnostics.
    """
    observed = np.asarray(observed, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if observed.shape != mask.shape or observed.shape != initial_tensor.shape:
        raise ValueError("observed, mask, and initial_tensor must have identical shapes")
    if ridge <= 0:
        raise ValueError("ridge (lambda_0) must be > 0 for strict block convexity")

    M = mask.astype(np.float64)
    O = np.zeros_like(observed)
    O[mask] = observed[mask]

    if init == "warm":
        A, B, C = initialize_factors_from_complete_kpd(
            initial_tensor, rank, sweeps=20, restarts=1, seed=seed
        )
    elif init == "random":
        A, B, C = random_factors(observed.shape, rank, seed)
    else:
        raise ValueError("init must be 'warm' or 'random'")

    # An optional scale minimization before the first recorded objective is fine:
    # it merely chooses the best regularized representative of the initialized
    # rank-one components.
    if balance_scales:
        for r in range(rank):
            A[:, r], B[:, r], C[:, r] = _regularizer_optimal_balance(
                A[:, r], B[:, r], C[:, r], lam_space, lam_time, ridge
            )

    current = reconstruct(A, B, C)
    history = []
    prev_obj = masked_objective(O, mask, A, B, C, lam_space, lam_time, ridge)

    if verbose:
        print("\n================ Theory-aligned Masked KPD ================")
        print(f"shape={observed.shape}, observed={mask.mean():.4f}, rank={rank}")
        print(f"initial objective={prev_obj:.10e}")
        if evaluation_gt is not None:
            evaluation_gt = np.asarray(evaluation_gt, dtype=np.float64)
            rec_eval = np.clip(current, 0, 1)
            if project_observed:
                rec_eval = rec_eval.copy(); rec_eval[mask] = O[mask]
            miss = ~mask
            print(
                f"start PSNR(missing)={masked_psnr(evaluation_gt,rec_eval,miss):.3f} dB, "
                f"SSIM={mean_ssim(evaluation_gt,rec_eval):.4f}"
            )

    for sweep in range(1, outer_sweeps + 1):
        tic = time.time()

        for r in range(rank):
            old_term = _outer3(A[:, r], B[:, r], C[:, r])
            current_minus_r = current - old_term
            R = O - current_minus_r

            # a update
            b = B[:, r]
            c = C[:, r]
            bc2 = (b[:, None] * c[None, :]) ** 2
            h_a = np.einsum("ijk,jk->i", M, bc2, optimize=True)
            rhs_a = np.einsum("ijk,ijk,j,k->i", M, R, b, c, optimize=True)
            a = _solve_smoothed(h_a, rhs_a, lam_space, ridge)

            # b update, with the newly updated a
            ac2 = (a[:, None] * c[None, :]) ** 2
            h_b = np.einsum("ijk,ik->j", M, ac2, optimize=True)
            rhs_b = np.einsum("ijk,ijk,i,k->j", M, R, a, c, optimize=True)
            b = _solve_smoothed(h_b, rhs_b, lam_space, ridge)

            # c update, with the newly updated a and b
            ab2 = (a[:, None] * b[None, :]) ** 2
            h_c = np.einsum("ijk,ij->k", M, ab2, optimize=True)
            rhs_c = np.einsum("ijk,ijk,i,j->k", M, R, a, b, optimize=True)
            c = _solve_smoothed(h_c, rhs_c, lam_time, ridge)

            if balance_scales:
                a, b, c = _regularizer_optimal_balance(
                    a, b, c, lam_space, lam_time, ridge
                )

            A[:, r], B[:, r], C[:, r] = a, b, c
            current = current_minus_r + _outer3(a, b, c)

        obj = masked_objective(O, mask, A, B, C, lam_space, lam_time, ridge)
        delta_obj = obj - prev_obj
        scale = max(1.0, abs(prev_obj))
        monotone_ok = bool(delta_obj <= monotonic_tol * scale)
        if check_monotonic and not monotone_ok:
            raise RuntimeError(
                f"Masked KPD objective increased at sweep {sweep}: "
                f"{prev_obj:.12e} -> {obj:.12e} (delta={delta_obj:.3e})."
            )

        raw = current.copy()
        backbone = np.clip(raw, 0.0, 1.0)
        rec = backbone.copy()
        if project_observed:
            rec[mask] = O[mask]

        obs_res = float(np.sqrt(np.mean((raw[mask] - O[mask]) ** 2)))
        row = {
            "sweep": sweep,
            "objective": obj,
            "objective_change": delta_obj,
            "monotone_ok": monotone_ok,
            "observed_rmse": obs_res,
            "seconds": time.time() - tic,
        }
        if evaluation_gt is not None:
            miss = ~mask
            row.update({
                "psnr_missing": masked_psnr(evaluation_gt, rec, miss),
                "psnr_all": psnr(evaluation_gt, rec),
                "ssim_all": mean_ssim(evaluation_gt, rec),
            })
        history.append(row)

        if verbose:
            msg = (
                f"[Masked sweep {sweep:02d}/{outer_sweeps}] "
                f"J={obj:.8e}, dJ={delta_obj:+.3e}, "
                f"obsRMSE={obs_res:.6f}, time={row['seconds']:.2f}s"
            )
            if evaluation_gt is not None:
                msg += (
                    f", PSNR(miss)={row['psnr_missing']:.3f}, "
                    f"SSIM={row['ssim_all']:.4f}"
                )
            print(msg)

        prev_obj = obj

    raw = current.copy()
    backbone = np.clip(raw, 0.0, 1.0)
    rec = backbone.copy()
    if project_observed:
        rec[mask] = O[mask]

    return MaskedKPDResult(rec, raw, backbone, A, B, C, history)

def _starts(n, block, stride):
    if block >= n:
        return [0]
    vals = list(range(0, n - block + 1, stride))
    if vals[-1] != n - block:
        vals.append(n - block)
    return vals


def hierarchical_local_correction(
    gt: np.ndarray,
    mask: np.ndarray,
    global_rec: np.ndarray,
    block_hw=(24, 24),
    block_t=10,
    stride_hw=(12, 12),
    stride_t=5,
    local_rank=1,
    local_sweeps=10,
    lam_space=0.02,
    lam_time=0.05,
    ridge=1e-5,
    seed=2234,
    verbose=True,
):
    """
    Global + local masked KPD correction.

    Fit the *observed residual* O - global_rec within overlapping
    spatiotemporal blocks. The local models extrapolate residual structure
    into missing entries. Overlapping predictions are averaged.

    This targets local details/edges that a single global low-K KPD misses.
    """
    H, W, T = gt.shape
    bh = min(block_hw[0], H)
    bw = min(block_hw[1], W)
    bt = min(block_t, T)
    sh = min(stride_hw[0], bh)
    sw = min(stride_hw[1], bw)
    st = min(stride_t, bt)

    Ores = np.zeros_like(gt)
    Ores[mask] = gt[mask] - global_rec[mask]

    corr_sum = np.zeros_like(gt)
    weight = np.zeros_like(gt)

    hs = _starts(H, bh, sh)
    ws = _starts(W, bw, sw)
    ts = _starts(T, bt, st)
    total = len(hs)*len(ws)*len(ts)
    count = 0

    for h0 in hs:
        for w0 in ws:
            for t0 in ts:
                count += 1
                sl = np.s_[h0:h0+bh, w0:w0+bw, t0:t0+bt]
                m = mask[sl]
                if m.mean() < 0.02:
                    continue

                # Residual "ground truth" is only used through the mask by run_masked_clsa.
                pseudo = np.zeros((bh, bw, bt), dtype=np.float64)
                pseudo[m] = Ores[sl][m]

                # zero initial tensor for residual model
                init0 = np.zeros_like(pseudo)

                rr = run_masked_clsa(
                    observed=pseudo,
                    mask=m,
                    initial_tensor=init0,
                    rank=local_rank,
                    outer_sweeps=local_sweeps,
                    lam_space=lam_space,
                    lam_time=lam_time,
                    ridge=ridge,
                    seed=seed + count,
                    init="random",
                    project_observed=False,
                    verbose=False,
                )

                # Local residual correction is signed, so use the raw factor reconstruction.
                raw = rr.raw_reconstruction
                corr_sum[sl] += raw
                weight[sl] += 1.0

                if verbose and (count == 1 or count % max(1, total//5) == 0):
                    print(f"  local blocks {count}/{total}")

    corr = np.zeros_like(gt)
    nz = weight > 0
    corr[nz] = corr_sum[nz] / weight[nz]

    out = np.clip(global_rec + corr, 0.0, 1.0)
    out[mask] = gt[mask]
    return out
