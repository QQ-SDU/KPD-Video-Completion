"""Only tests RGB data and evaluation mechanics using synthetic arrays. Does NOT test the nine models."""
from __future__ import annotations
import sys,tempfile,subprocess,csv
from pathlib import Path
import numpy as np
import cv2
from scipy.io import savemat
HERE=Path(__file__).resolve().parent

def run(*args):
    cmd=[sys.executable,*map(str,args)]
    result=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.returncode:raise RuntimeError(' '.join(cmd)+'\n'+result.stdout+'\n'+result.stderr)
    return result.stdout

def main():
    with tempfile.TemporaryDirectory(prefix='DAVIS_中文_RGB_') as tmp:
        root=Path(tmp);data=root/'prep';res=root/'results'
        jpeg=root/'DAVIS'/'JPEGImages'/'480p'/'blackswan';jpeg.mkdir(parents=True)
        for i in range(8):
            bgr=np.empty((18,24,3),dtype=np.uint8)
            bgr[:,:,0]=np.arange(24)[None,:]*8
            bgr[:,:,1]=np.arange(18)[:,None]*9
            bgr[:,:,2]=20+i*20
            (jpeg/f'{i:05d}.jpg').write_bytes(cv2.imencode('.jpg',bgr)[1].tobytes())
        run(HERE/'prepare_davis_rgb.py','--davis-root',root/'DAVIS','--out',data,'--sequences','blackswan','--height','16','--width','20','--frames','8')
        run(HERE/'rgb9.py','prepare','--data',data,'--output',res,'--videos','blackswan')
        with np.load(res/'cases'/'blackswan__sr020__seed20261014.npz') as d:
            gt=d['gt'];observed=d['observed'];mask=d['mask'].astype(bool)
        assert gt.shape==(16,20,8,3)
        assert np.array_equal(mask[:,:,:,0],mask[:,:,:,1]) and np.array_equal(mask[:,:,:,0],mask[:,:,:,2])
        assert np.array_equal(gt[mask],observed[mask])
        assert np.mean(np.abs(gt[:,:,:,:1]-gt[:,:,:,1:2]))>0.01
        # This uses fake arrays solely to test the evaluation file format; never publish them as benchmark results.
        methods=['DRO-TFF','HLRTF','KBR','LRTFR','OTLRM','TNN','SALTS','S2NTNN','Ours']
        for sr in ('020','030','040'):
            stem='blackswan__sr'+sr+'__seed20261014'
            with np.load(res/'cases'/(stem+'.npz')) as d:
                g=d['gt'];o=d['observed'];m=d['mask'].astype(bool)
            for method in methods:
                x=g.copy();x[~m]=np.clip(x[~m]*0.99,0,1)
                path=res/'reconstructions'/method/(method+'__'+stem+'.mat');path.parent.mkdir(parents=True,exist_ok=True)
                savemat(path,{'recon':x.astype('float32'),'runtime_seconds':123.0})
        run(HERE/'rgb9.py','evaluate','--output',res,'--videos','blackswan')
        assert (res/'reports'/'selected__blackswan'/'SUMMARY_COMPLETE.csv').exists()
        run(HERE/'rgb9.py','visualize','--output',res,'--videos','blackswan',
            '--display-methods','KBR,S2NTNN,Ours','--frame','2')
        assert list((res/'reports'/'selected__blackswan'/'qualitative').glob('*.png'))
        print('PASS: Unicode filename, real RGB array, shared mask, RGB metrics & image layout (synthetic only)')
if __name__=='__main__':main()
