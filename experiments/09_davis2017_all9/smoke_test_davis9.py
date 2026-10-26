"""Synthetic-only file-contract test. Results are never included as paper data."""
from pathlib import Path
import tempfile,hashlib,subprocess,sys,csv
import numpy as np
import scipy.io as sio
from PIL import Image
HERE=Path(__file__).resolve().parent
PY=sys.executable

def call(*args,rc=0):
    cmd=[PY,str(HERE/'davis9.py'),*map(str,args)]
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print('TEST',cmd[2],r.returncode,'\n',r.stdout[-1200:]);assert r.returncode==rc

with tempfile.TemporaryDirectory(prefix='davis9_smoke_') as tmp:
    root=Path(tmp);data=root/'data';data.mkdir();out=root/'out'
    rng=np.random.default_rng(62)
    vid=(0.2+0.55*rng.random((24,32,8))).astype('float32')
    f=data/'blackswan.npy';np.save(f,vid)
    dig=hashlib.sha256(f.read_bytes()).hexdigest()
    with (data/'manifest.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,['name','file','sha256']);w.writeheader();w.writerow(dict(name='blackswan',file=f.name,sha256=dig))
    call('prepare','--data',data,'--output',out,'--rates','0.2,0.3,0.4','--seed','17')
    cases=sorted((out/'cases').glob('*.npz')); assert len(cases)==3
    for c in cases:
        with np.load(c) as a:
            mask=a['mask'].astype(bool)
            assert mask.shape==vid.shape and np.array_equal(mask[:,:,0],mask[:,:,7])
            assert np.array_equal(a['observed'][~mask],np.zeros_like(a['observed'][~mask]))
        for method in ['DRO-TFF','HLRTF','KBR','LRTFR','OTLRM','TNN','SALTS','S2NTNN','Ours']:
            file=out/'reconstructions'/method/(method+'__'+c.stem+'.mat')
            file.parent.mkdir(parents=True,exist_ok=True)
            # Mock reconstruction ONLY for I/O validation; not a real method result.
            rec=np.full(vid.shape,0.5,dtype='float32'); rec[mask]=vid[mask]
            sio.savemat(file,dict(recon=rec,runtime_seconds=1.2))
    call('inspect','--output',out)
    call('evaluate','--output',out)
    assert (out/'SUMMARY_COMPLETE.csv').exists()
    tex=subprocess.run([PY,str(HERE/'make_davis9_tex.py'),'--output',str(out)],capture_output=True,text=True)
    assert tex.returncode==0, tex.stdout+tex.stderr
    assert (out/'DAVIS9_TABLE.tex').exists()
    call('visualize','--output',out)
    assert list((out/'qualitative').glob('*.png'))
    print('SYNTHETIC CONTRACT TEST PASSED (No paper results produced)')
