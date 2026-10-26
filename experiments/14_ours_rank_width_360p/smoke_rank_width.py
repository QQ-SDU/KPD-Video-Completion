#!/usr/bin/env python3
"""Fast fixture smoke test. No GPU, no MATLAB, no DAVIS data needed."""
from __future__ import annotations
import csv
import tempfile
import types
from pathlib import Path
import numpy as np
from scipy.io import savemat
import rank_width_rgb360 as mod


def main():
    original_shape=mod.SHAPE
    original_run=mod.subprocess.run
    mod.SHAPE=(16,20,5,3)
    try:
        with tempfile.TemporaryDirectory(prefix='kpd_v10_') as s:
            base=Path(s);source=base/'v8';out=base/'v10';v9=base/'v9'
            (source/'cases').mkdir(parents=True)
            (source/'channel_cases').mkdir(parents=True)
            case=mod.old.stem('blackswan',0.3)
            rng=np.random.default_rng(11)
            gt=rng.random(mod.SHAPE,dtype=np.float32)
            spatial=rng.random((mod.SHAPE[0],mod.SHAPE[1])) < .3
            mask=np.broadcast_to(spatial[:,:,None,None],mod.SHAPE).copy()
            observed=np.where(mask,gt,0).astype(np.float32)
            np.savez_compressed(mod.old.casefile(source,case),gt=gt,observed=observed,mask=mask)
            for ci,ch in enumerate(mod.old.CHANNELS):
                savemat(mod.old.cfile(source,case,ch),{'observed':observed[:,:,:,ci],
                                                       'mask':mask[:,:,:,ci]})
            baseline=mod.v9_rgb(v9,case)
            baseline.parent.mkdir(parents=True,exist_ok=True)
            baseline_pred=np.where(mask,observed,gt*.8+.1).astype(np.float32)
            savemat(baseline,{'recon':baseline_pred,'runtime_seconds':np.array([[100.]])})
            call_count=[0]
            def fake_run(cmd,**kwargs):
                call_count[0]+=1
                dst=Path(cmd[cmd.index('--out')+1]); rank=int(cmd[cmd.index('--rank')+1]);hidden=int(cmd[cmd.index('--hidden')+1])
                ch=dst.name.rsplit('__',1)[-1].split('.')[0]
                ci=('R','G','B').index(ch)
                pred=gt[:,:,:,ci]*(.65+.005*rank+.001*hidden)
                savemat(dst,{'recon':np.clip(pred,0,1),'runtime_seconds':np.array([[12.]])})
                dst.with_suffix('.json').write_text('{}',encoding='utf-8')
                return types.SimpleNamespace(returncode=0)
            mod.subprocess.run=fake_run
            args=types.SimpleNamespace(device='cpu',dry_run=False,overwrite=False)
            mod._check_protocol(out,source)
            for k,h in mod.NEW_CONFIGS:
                assert mod.run_one(source,out,'blackswan',.3,k,h,args)
            assert call_count[0]==15,call_count[0]
            for k,h in mod.NEW_CONFIGS:
                assert mod.run_one(source,out,'blackswan',.3,k,h,args)
            assert call_count[0]==15,'Resume re-ran existing configurations'
            missing=mod.evaluate(source,out,v9,['blackswan'],[.3])
            assert missing==0,missing
            with (out/'results_per_video.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            assert len(rows)==6,(len(rows),rows)
            assert {(int(r['rank']),int(r['hidden'])) for r in rows}==set(mod.ALL_CONFIGS)
            # incremental evaluation retains previous data, no duplicate entries
            assert mod.evaluate(source,out,v9,['blackswan'],[.3])==0
            with (out/'results_per_video.csv').open(encoding='utf-8-sig',newline='') as f:
                assert len(list(csv.DictReader(f)))==6
            mod.plot(out)
            assert (out/'figures'/'rank_PSNR_SR030.pdf').is_file()
            assert (out/'figures'/'width_PSNR_SR030.pdf').is_file()
            assert len(list((out/'logs').rglob('*.log')))==15
            assert valid_sidecars(out,case)
        print('SMOKE TEST: PASS (5 new configs, 15 channel fits, v9 baseline, resume, CSV, 2 PDF figures)')
    finally:
        mod.SHAPE=original_shape
        mod.subprocess.run=original_run


def valid_sidecars(out,case):
    return all(mod.new_channel(out,case,k,h,ch).with_suffix('.json').is_file()
               for k,h in mod.NEW_CONFIGS for ch in mod.old.CHANNELS)

if __name__=='__main__':main()
