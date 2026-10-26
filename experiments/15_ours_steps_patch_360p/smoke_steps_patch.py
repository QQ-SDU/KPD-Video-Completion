#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline mocked integration test; does not claim to validate real GPU numerical results."""
from __future__ import annotations
import csv,json,tempfile,types
from pathlib import Path
import numpy as np
from scipy.io import savemat
import steps_patch_rgb360 as mod


def main():
    original_shape=mod.SHAPE
    original_run=mod.subprocess.run
    mod.SHAPE=(16,20,5,3)
    try:
        with tempfile.TemporaryDirectory(prefix='kpd_v11_') as temp:
            root=Path(temp);src=root/'v8';out=root/'v11';v10=root/'v10'
            (src/'cases').mkdir(parents=True);(src/'channel_cases').mkdir(parents=True)
            case=mod.case_id('blackswan',.3)
            rng=np.random.default_rng(43)
            gt=rng.random(mod.SHAPE,dtype=np.float32)
            sm=rng.random(mod.SHAPE[:2]) < .3
            mask=np.broadcast_to(sm[:,:,None,None],mod.SHAPE).copy()
            observed=np.where(mask,gt,0).astype(np.float32)
            np.savez_compressed(mod.legacy.casefile(src,case),gt=gt,observed=observed,mask=mask)
            for ci,ch in enumerate(mod.legacy.CHANNELS):
                savemat(mod.source_channel(src,case,ch),{'observed':observed[:,:,:,ci],
                         'mask':mask[:,:,:,ci]})
            baseline=mod.baseline_rgb(v10,case)
            baseline.parent.mkdir(parents=True)
            savemat(baseline,{'recon':np.where(mask,observed,.9*gt),
                              'runtime_seconds':np.asarray([[100.]])})
            calls=[]
            def fake_run(cmd,**kwargs):
                def arg(a):return cmd[cmd.index(a)+1]
                steps=int(arg('--steps'));patch=int(arg('--patch-h'))
                assert int(arg('--rank'))==12 and int(arg('--hidden'))==32
                assert int(arg('--patch-w'))==patch
                assert int(arg('--patch-t'))==16
                assert int(arg('--seed'))==mod.SEED
                dst=Path(arg('--out'))
                ch=dst.name.rsplit('__',1)[-1].split('.')[0]
                pred=gt[:,:,:,('R','G','B').index(ch)]*.8
                savemat(dst,{'recon':pred,'runtime_seconds':np.asarray([[17.]])})
                diag={'config':{'rank':12,'hidden':32,'spatial_kernel':3,
                                'steps':steps,'patch':[16,patch,patch]},'seed':mod.SEED}
                dst.with_suffix('.json').write_text(json.dumps(diag),encoding='utf-8')
                calls.append((ch,steps,patch))
                return types.SimpleNamespace(returncode=0)
            mod.subprocess.run=fake_run
            args=types.SimpleNamespace(device='cpu',dry_run=False,overwrite=False)
            mod._protocol(out,src)
            for steps,patch in mod.NEW_CONFIGS.values():
                assert mod.run_one(src,out,'blackswan',.3,steps,patch,args)
            assert len(calls)==12,len(calls)
            for steps,patch in mod.NEW_CONFIGS.values():
                assert mod.run_one(src,out,'blackswan',.3,steps,patch,args)
            assert len(calls)==12,'Resume should not rerun valid RGB'
            assert mod.evaluate(src,out,v10,['blackswan'],[.3],baseline=True)==0
            with (out/'results_per_video.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            assert len(rows)==5,len(rows)
            assert {(int(x['training_steps']),int(x['patch_h'])) for x in rows}==set(mod.ALL_CONFIGS.values())
            assert mod.evaluate(src,out,v10,['blackswan'],[.3],baseline=True)==0
            with (out/'results_per_video.csv').open(encoding='utf-8-sig',newline='') as f:
                assert len(list(csv.DictReader(f)))==5
            mod.plot(out)
            assert (out/'figures'/'steps_PSNR_SR030.pdf').is_file()
            assert (out/'figures'/'patch_PSNR_SR030.pdf').is_file()
            # no baseline case: never manufacture A0, report missing
            baseline.unlink()
            assert mod.evaluate(src,out,v10,['blackswan'],[.3],baseline=True)==1
            with (out/'results_per_video.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            assert len(rows)==4
            for steps,patch in mod.NEW_CONFIGS.values():
                f=mod.rgb_path(out,case,steps,patch)
                assert mod._valid_rgb(f,steps,patch)
                assert not mod._valid_rgb(f,steps+1,patch)
        print('SMOKE TEST: PASS | 4 configs x 3 channels, resume, baseline, missing baseline, CSV merge, plots, config validation')
    finally:
        mod.SHAPE=original_shape
        mod.subprocess.run=original_run

if __name__=='__main__':main()
