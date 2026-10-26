#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiny synthetic I/O smoke test; NOT real DAVIS/GPU/MATLAB evaluation."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import savemat
import final_davis as fd
import frozen_benchmark as fb
import video_supplement as mv


def main():
    assert fd.choose(['blackswan','camel','parkour'],'all',2,1)==['camel','parkour']
    with tempfile.TemporaryDirectory(prefix='kpd_v14_test_') as td:
        root=Path(td);dav=root/'DAVIS'/'JPEGImages'/'480p'
        for name in ['blackswan','camel','parkour']:
            (dav/name).mkdir(parents=True)
            (dav/name/'00000.jpg').write_bytes(b'x')
        assert fd.scan(root/'DAVIS')[:3]==['blackswan','camel','parkour']
        src=root/'source';dst=root/'new';case=fb.case_id('blackswan',.3)
        (src/'cases').mkdir(parents=True)
        fb.SHAPE=(24,32,4,3)
        fb.v11.SHAPE=fb.SHAPE
        gt=np.random.default_rng(123).uniform(.08,.95,size=fb.SHAPE).astype('float32')
        mask2=np.zeros((24*32,),dtype=bool)
        mask2[np.random.default_rng(12).choice(24*32,round(24*32*.3),replace=False)]=True
        mask2=mask2.reshape(24,32)
        mask=np.broadcast_to(mask2[:,:,None,None],fb.SHAPE).copy()
        obs=np.where(mask,gt,0).astype('float32')
        np.savez_compressed(fb.orig.paths(src,case),gt=gt,observed=obs,mask=mask.astype('uint8'))
        (src/'PROTOCOL.json').write_text(json.dumps({'color':'RGB','seed':fb.SEED}),encoding='utf-8')
        methods=(*fb.BASELINES,'Ours')
        for i,m in enumerate(methods):
            rec=(.92*gt+.08*np.float32(i/14)).astype('float32')
            rec[mask]=obs[mask]
            p=fb.orig.rgb_out(src,m,case);p.parent.mkdir(parents=True,exist_ok=True)
            savemat(p,{'recon':rec,'runtime_seconds':np.array([[3.0]])},do_compression=True)
        p=fb.frozen_out(dst,case);p.parent.mkdir(parents=True,exist_ok=True)
        rec=gt.copy();rec[mask]=obs[mask]
        savemat(p,{'recon':rec,'rank':[[12]],'hidden':[[32]],'steps':[[3600]],
                   'patch_h':[[96]],'patch_w':[[96]],'seed':[[fb.SEED]],
                   'runtime_seconds':[[23.0]]},do_compression=True)
        rows,complete=fb.evaluate(src,root/'none_v12',dst,['blackswan'],[.3],'test')
        assert complete and len(rows)==10
        assert (dst/'tables'/'TABLE_MAIN_9METHODS.tex').exists()
        fb.visual(src,root/'none_v12',dst,['blackswan'],[.3],2,'2,2,12,10','main',True)
        assert list((dst/'figures'/'main_8columns').glob('*.pdf'))
        old_shape=fb.SHAPE
        try:
            mp4=mv.create_mp4(src,root/'none_v12',dst,'blackswan',.3,'main',
                ','.join(fb.MAIN),'2,2,12,10',5,180,0,4,True)
            assert mp4.is_file() and mp4.stat().st_size>1000
        finally:fb.SHAPE=old_shape
        fd.progress(src,root/'none_v12',dst,['blackswan'])
    print('SMOKE TEST PASS: discovery / selection / CSV / LaTeX / 8col PDF / true-array MP4')

if __name__=='__main__':main()
