#!/usr/bin/env python3
"""Show the fixed DAVIS RGB spatial-mask fingerprints without dataset access."""
from __future__ import annotations
import hashlib
import numpy as np

SEED=20261014
H,W,T,C=360,640,50,3


def make_spatial(sr):
    rng=np.random.default_rng(SEED)
    mask=np.zeros(H*W,dtype=np.uint8)
    mask[rng.choice(H*W,size=round(sr*H*W),replace=False)]=1
    return mask.reshape(H,W)


def main():
    print('DAVIS RGB 360p: seed=',SEED,'numpy=',np.__version__, 'shape=',(H,W,T,C))
    for sr in (0.2,0.3,0.4):
        mask=make_spatial(sr)
        digest=hashlib.sha256(mask.tobytes()).hexdigest()
        print(f'SR={sr:.1f} observed={int(mask.sum())}/{H*W} mask2d_sha256={digest}')
    print('The same HxW mask is broadcast to every frame and RGB channel.')
    return 0

if __name__=='__main__':raise SystemExit(main())
