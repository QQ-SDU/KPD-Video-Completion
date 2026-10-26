# -*- coding: utf-8 -*-
"""
Complete-data low-rank representation methods for manuscript Fig. 1.

Compared representation families
-------------------------------
1. KPD-CLSA
2. Matrix-SVD
3. t-SVD (truncated tubal-rank representation)
4. Tucker-HOSVD

Important:
- CP-ALS is deliberately not included as a separate representation family.
  For the vector-form third-order finite-sum KPD used here, the rank-K
  separable parameterization coincides algebraically with a rank-K CP model.
  CP-ALS is therefore better treated as an alternative optimizer in a
  separate solver ablation.
- t-SVD replaces TT-SVD in the main Fig. 1 because it is a more direct
  third-order tensor-spectral reference for the video tensor X in R^{H x W x T}.
"""

import numpy as np
from kpd_core import finite_sum_clsa, outer3
from metrics import psnr, mean_frame_ssim


def _row(video, method, rank_spec, params, n_data, x, xhat):
    return {
        "video": video,
        "method": method,
        "rank_spec": rank_spec,
        "param_count": int(params),
        "budget_pct": float(100.0 * params / n_data),
        "psnr": float(psnr(x, xhat)),
        "ssim": float(mean_frame_ssim(x, xhat)),
    }


def _geom_int_schedule(lo, hi, n, must_include=()):
    lo = int(lo)
    hi = int(hi)
    if hi <= lo:
        vals = [lo]
    else:
        vals = sorted(
            set(int(round(v)) for v in np.geomspace(max(1, lo), hi, n))
        )
        vals = sorted(set([lo, hi] + vals))
    vals = sorted(
        set(vals + [int(v) for v in must_include if lo <= int(v) <= hi])
    )
    return vals


# -------------------------------------------------------------------------
# 1) KPD / Cheng-style sequential CLSA on complete tensors
# -------------------------------------------------------------------------
def curve_kpd(
    x,
    video,
    max_rank=36,
    points=16,
    sweeps=40,
    restarts=2,
    seed=1,
):
    H, W, T = x.shape
    n_data = x.size
    ranks = _geom_int_schedule(1, max_rank, points, must_include=(4,))

    print(f"[KPD-CLSA] max_rank={max_rank}; recorded prefix ranks={ranks}")
    model = finite_sum_clsa(
        x,
        rank=max_rank,
        max_sweeps=sweeps,
        tol=1e-6,
        restarts=restarts,
        seed=seed,
        verbose=True,
    )

    rows = []
    recon = np.zeros_like(x, dtype=np.float64)
    wanted = set(ranks)

    for q, comp in enumerate(model.components, start=1):
        recon += outer3(comp.a, comp.b, comp.c)
        if q in wanted:
            # One rank-one vector-form component stores H+W+T real values.
            params = q * (H + W + T)
            rows.append(
                _row(
                    video,
                    "KPD-CLSA",
                    f"K={q}",
                    params,
                    n_data,
                    x,
                    recon.copy(),
                )
            )
    return rows


# -------------------------------------------------------------------------
# 2) Matrix SVD of the (HW) x T unfolding
# -------------------------------------------------------------------------
def curve_svd(x, video, points=16):
    H, W, T = x.shape
    n_data = x.size

    M = x.reshape(H * W, T)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    ranks = _geom_int_schedule(1, min(M.shape), points)

    rows = []
    for r in ranks:
        Mr = (U[:, :r] * s[:r]) @ Vt[:r, :]
        xhat = Mr.reshape(H, W, T)

        # Singular values are absorbed into one factor:
        # P_SVD = r(HW + T).
        params = r * (H * W + T)
        rows.append(
            _row(video, "Matrix-SVD", f"r={r}", params, n_data, x, xhat)
        )
    return rows


# -------------------------------------------------------------------------
# 3) Truncated t-SVD
# -------------------------------------------------------------------------
def _prepare_tsvd(x):
    """
    Compute the Fourier-domain frontal-slice SVDs once.

    rFFT is used because the input video is real-valued; irFFT reconstructs
    the real tensor and implicitly handles the conjugate-symmetric frequencies.
    """
    xf = np.fft.rfft(x, axis=2)
    factors = []
    for k in range(xf.shape[2]):
        U, s, Vh = np.linalg.svd(xf[:, :, k], full_matrices=False)
        factors.append((U, s, Vh))
    return xf.shape, factors


def _tsvd_reconstruct_from_factors(shape, factors, rank, T):
    xf_hat = np.zeros(shape, dtype=np.complex128)

    for k, (U, s, Vh) in enumerate(factors):
        rr = min(int(rank), len(s))
        xf_hat[:, :, k] = (U[:, :rr] * s[:rr]) @ Vh[:rr, :]

    xhat = np.fft.irfft(xf_hat, n=T, axis=2)
    return np.asarray(xhat.real, dtype=np.float64)


def curve_tsvd(x, video, max_rank=32, points=16):
    """
    Truncated t-SVD curve.

    For tubal rank r, compact t-SVD factors are counted as
        U_r: H x r x T
        S_r: r diagonal tubes, each of length T
        V_r: W x r x T
    giving
        P_tSVD = r*T*(H + W + 1).

    This count is made in the real tensor/t-product domain. We do not give
    an extra storage discount for Fourier conjugate symmetry.
    """
    H, W, T = x.shape
    n_data = x.size

    rmax = min(int(max_rank), H, W)
    ranks = _geom_int_schedule(1, rmax, points)

    print(f"[t-SVD] preparing {T//2 + 1} rFFT frontal-slice SVDs ...")
    fft_shape, factors = _prepare_tsvd(x)

    rows = []
    for r in ranks:
        xhat = _tsvd_reconstruct_from_factors(fft_shape, factors, r, T)
        params = r * T * (H + W + 1)

        rows.append(
            _row(
                video,
                "t-SVD",
                f"tubal-rank={r}",
                params,
                n_data,
                x,
                xhat,
            )
        )
        print(
            f"    tubal rank {r:02d}: "
            f"storage={100.0*params/n_data:.6f}%"
        )

    return rows


# -------------------------------------------------------------------------
# 4) Tucker / truncated HOSVD
# -------------------------------------------------------------------------
def _mode_unfold(x, mode):
    y = np.moveaxis(x, mode, 0)
    return y.reshape(y.shape[0], -1)


def _mode_product(x, U, mode):
    y = np.tensordot(U, x, axes=(1, mode))
    return np.moveaxis(y, 0, mode)


def curve_tucker(x, video, max_frac=0.40, points=16):
    H, W, T = x.shape
    n_data = x.size

    U0, _, _ = np.linalg.svd(_mode_unfold(x, 0), full_matrices=False)
    U1, _, _ = np.linalg.svd(_mode_unfold(x, 1), full_matrices=False)
    U2, _, _ = np.linalg.svd(_mode_unfold(x, 2), full_matrices=False)

    r0max = max(1, min(H, int(round(max_frac * H))))
    r1max = max(1, min(W, int(round(max_frac * W))))
    r2max = max(1, min(T, int(round(max_frac * T))))

    triples = []
    for alpha in np.geomspace(0.04, 1.0, points):
        rr = (
            max(1, int(round(alpha * r0max))),
            max(1, int(round(alpha * r1max))),
            max(1, int(round(alpha * r2max))),
        )
        if rr not in triples:
            triples.append(rr)

    rows = []
    for r0, r1, r2 in triples:
        A = U0[:, :r0]
        B = U1[:, :r1]
        C = U2[:, :r2]

        core = _mode_product(x, A.T, 0)
        core = _mode_product(core, B.T, 1)
        core = _mode_product(core, C.T, 2)

        xhat = _mode_product(core, A, 0)
        xhat = _mode_product(xhat, B, 1)
        xhat = _mode_product(xhat, C, 2)

        params = H * r0 + W * r1 + T * r2 + r0 * r1 * r2
        rows.append(
            _row(
                video,
                "Tucker-HOSVD",
                f"({r0},{r1},{r2})",
                params,
                n_data,
                x,
                xhat,
            )
        )

    return rows
