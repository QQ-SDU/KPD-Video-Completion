# -*- coding: utf-8 -*-
"""
Complete-data CLSA/KPD core reused from the current KPD project line.

For a third-order tensor X, each component is
    a_r o b_r o c_r
whose vectorization is a Kronecker product of c_r, b_r, and a_r.

Algebraically this is the same rank-one model class as CP.
The distinction in this experiment is the sequential residual CLSA ordering,
NOT a new tensor-rank notion.
"""
from dataclasses import dataclass
import numpy as np

EPS = 1e-12

@dataclass
class Rank1Result:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    error: float
    sweeps: int

@dataclass
class KPDModel:
    components: list
    reconstruction: np.ndarray
    residual: np.ndarray
    residual_norms: list

def outer3(a, b, c):
    return np.einsum("i,j,k->ijk", a, b, c, optimize=True)

def clsa_rank1(tensor, max_sweeps=40, tol=1e-6, restarts=2, seed=0):
    x = np.asarray(tensor, dtype=np.float64)
    H, W, T = x.shape
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(max(1, restarts)):
        a = rng.normal(size=H)
        b = rng.normal(size=W)
        c = rng.normal(size=T)
        b /= np.linalg.norm(b) + EPS
        c /= np.linalg.norm(c) + EPS
        prev = outer3(a, b, c)
        sweeps_done = 0

        for sweep in range(1, max_sweeps + 1):
            sweeps_done = sweep
            a = np.einsum("ijk,j,k->i", x, b, c, optimize=True) / ((b @ b)*(c @ c) + EPS)
            b = np.einsum("ijk,i,k->j", x, a, c, optimize=True) / ((a @ a)*(c @ c) + EPS)
            c = np.einsum("ijk,i,j->k", x, a, b, optimize=True) / ((a @ a)*(b @ b) + EPS)

            nb = np.linalg.norm(b) + EPS
            nc = np.linalg.norm(c) + EPS
            b /= nb; c /= nc; a *= nb * nc

            cur = outer3(a, b, c)
            change = np.linalg.norm(cur - prev) / (np.linalg.norm(prev) + EPS)
            prev = cur
            if change < tol:
                break

        err = float(np.linalg.norm(x - prev))
        rr = Rank1Result(a.copy(), b.copy(), c.copy(), err, sweeps_done)
        if best is None or rr.error < best.error:
            best = rr

    return best

def finite_sum_clsa(tensor, rank, max_sweeps=40, tol=1e-6, restarts=2, seed=0, verbose=True):
    x = np.asarray(tensor, dtype=np.float64)
    residual = x.copy()
    recon = np.zeros_like(x)
    comps = []
    residual_norms = [float(np.linalg.norm(residual))]
    xnorm = np.linalg.norm(x) + EPS

    for k in range(rank):
        comp = clsa_rank1(
            residual,
            max_sweeps=max_sweeps,
            tol=tol,
            restarts=restarts,
            seed=seed + 1009*k,
        )
        term = outer3(comp.a, comp.b, comp.c)
        recon += term
        residual -= term
        comps.append(comp)
        residual_norms.append(float(np.linalg.norm(residual)))
        if verbose:
            print(
                f"    KPD component {k+1:02d}/{rank}: "
                f"sweeps={comp.sweeps:02d}, relative residual={residual_norms[-1]/xnorm:.6f}"
            )

    return KPDModel(comps, recon, residual, residual_norms)
