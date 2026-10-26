#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick deterministic checks for the theory-aligned masked KPD implementation."""
from __future__ import annotations
import numpy as np

from kpd_video_core import smooth_fill
from masked_kpd import (
    run_masked_clsa, masked_objective, _regularizer_optimal_balance,
    reconstruct,
)


def main():
    rng = np.random.default_rng(7)
    H,W,T,K = 10,9,6,2
    A0=rng.normal(size=(H,K)); B0=rng.normal(size=(W,K)); C0=rng.normal(size=(T,K))
    gt=np.clip(reconstruct(A0,B0,C0),-2,2)

    tube=rng.random((H,W))<0.45
    mask=np.repeat(tube[:,:,None],T,axis=2)
    observed=np.zeros_like(gt); observed[mask]=gt[mask]
    init=smooth_fill(observed,mask)

    res=run_masked_clsa(
        observed=observed,mask=mask,initial_tensor=init,
        rank=2,outer_sweeps=5,lam_space=5e-3,lam_time=5e-2,ridge=1e-5,
        seed=3,init='warm',project_observed=False,verbose=False,
        balance_scales=True,check_monotonic=True,
    )
    objs=[h['objective'] for h in res.history]
    mono=all(objs[i] <= objs[i-1] + 1e-8*max(1,abs(objs[i-1])) for i in range(1,len(objs)))
    print('CHECK 1 sweep objective monotone:',mono)
    print('  objectives:', ' -> '.join(f'{x:.6e}' for x in objs))

    # No-leakage check: change values outside the fitting mask; result must not change.
    observed2=observed.copy()
    observed2[~mask]=1000.0+rng.normal(size=np.count_nonzero(~mask))
    res2=run_masked_clsa(
        observed=observed2,mask=mask,initial_tensor=init,
        rank=2,outer_sweeps=5,lam_space=5e-3,lam_time=5e-2,ridge=1e-5,
        seed=3,init='warm',project_observed=False,verbose=False,
        balance_scales=True,check_monotonic=True,
    )
    leak=max(
        np.max(np.abs(res.A-res2.A)),
        np.max(np.abs(res.B-res2.B)),
        np.max(np.abs(res.C-res2.C)),
        np.max(np.abs(res.raw_reconstruction-res2.raw_reconstruction)),
    )
    print('CHECK 2 non-context values have no influence:', leak < 1e-12)
    print('  max difference:', f'{leak:.3e}')

    # Scale-balancing check on one arbitrary component.
    a=rng.normal(size=H); b=3*rng.normal(size=W); c=0.2*rng.normal(size=T)
    AA=a[:,None]; BB=b[:,None]; CC=c[:,None]
    before=masked_objective(observed,mask,AA,BB,CC,5e-3,5e-2,1e-5)
    aa,bb,cc=_regularizer_optimal_balance(a,b,c,5e-3,5e-2,1e-5)
    after=masked_objective(observed,mask,aa[:,None],bb[:,None],cc[:,None],5e-3,5e-2,1e-5)
    prod_err=np.max(np.abs(np.einsum('i,j,k->ijk',a,b,c)-np.einsum('i,j,k->ijk',aa,bb,cc)))
    print('CHECK 3 optimal balancing preserves component:',prod_err < 1e-10)
    print('  product max error:',f'{prod_err:.3e}')
    print('CHECK 4 optimal balancing does not increase J:',after <= before + 1e-10*max(1,abs(before)))
    print('  J before/after:',f'{before:.6e} -> {after:.6e}')

    if not (mono and leak < 1e-12 and prod_err < 1e-10 and after <= before + 1e-10*max(1,abs(before))):
        raise SystemExit('THEORY CHECK FAILED')
    print('\nALL THEORY-ALIGNMENT CHECKS PASSED.')


if __name__=='__main__':
    main()
