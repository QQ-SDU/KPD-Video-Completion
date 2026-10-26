#!/usr/bin/env python3
"""Read-only dependency, CUDA, MATLAB and DAVIS input checks."""
from __future__ import annotations
import argparse, importlib, os, shutil, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--strict',action='store_true',help='exit nonzero if required components missing')
p.add_argument('--davis-root',default=os.environ.get('DAVIS_ROOT',str(Path.home() / "kpd_data" / 'DAVIS')))
args=p.parse_args()
errors=[]
print('Python:',sys.executable)
print('Conda env:',os.environ.get('CONDA_DEFAULT_ENV','(unknown)'))
names=['numpy','scipy','skimage','cv2','pandas','matplotlib','PIL','h5py','torch']
for name in names:
    try:
        x=importlib.import_module(name)
        print(f'  {name}: OK',getattr(x,'__version__',''))
    except Exception as exc:
        print(f'  {name}: FAILED: {exc}');errors.append(name)
try:
    import torch
    print('CUDA:',torch.cuda.is_available(),'Torch CUDA:',torch.version.cuda)
    if not torch.cuda.is_available(): errors.append('CUDA')
    else:
        print('GPU:',torch.cuda.get_device_name(0),'capability:',torch.cuda.get_device_capability(0))
        cap=torch.cuda.get_device_capability(0)
        if cap[0]>=12 and 'sm_120' not in torch.cuda.get_arch_list():
            print('FAIL: this PyTorch wheel lacks sm_120 support');errors.append('sm_120')
        else:
            x=torch.randn((1,4,4,8,8),device='cuda',requires_grad=True)
            m=torch.nn.Conv3d(4,12,3,padding=1).cuda()
            m(x).square().mean().backward();torch.cuda.synchronize()
            print('CUDA Conv3D forward/backward: PASS')
except Exception as exc:
    print('GPU probe FAILED:',repr(exc));errors.append('GPU probe')
matlab=os.environ.get('MATLAB_BIN','matlab')
found=shutil.which(matlab)
print('MATLAB:',found or 'NOT FOUND (set MATLAB_BIN or add to PATH)')
if not found:errors.append('MATLAB')
data=Path(args.davis_root)
candidates=[data/'JPEGImages'/'480p',data/'DAVIS'/'JPEGImages'/'480p',data/'2017'/'JPEGImages'/'480p']
located=next((c for c in candidates if (c/'blackswan').is_dir()),None)
print('DAVIS root:',args.davis_root,'JPEG directory:',located or 'NOT FOUND')
if located is None:errors.append('DAVIS')
print('RESULT:', 'READY' if not errors else 'NOT READY / warnings: '+', '.join(errors))
if args.strict and errors:sys.exit(1)
