#!/usr/bin/env python3
"""Test actual RTX 5080 CUDA compatibility, including Conv3D and backward pass."""
import sys
try:
 import torch
 print('PyTorch',torch.__version__,'CUDA build',torch.version.cuda)
 print('CUDA available:',torch.cuda.is_available())
 if not torch.cuda.is_available():raise RuntimeError('No CUDA available in this environment')
 print('GPU:',torch.cuda.get_device_name(0))
 cap=torch.cuda.get_device_capability(0)
 print('GPU compute capability:',cap,'compiled arch list:',torch.cuda.get_arch_list())
 if cap[0]>=12 and 'sm_120' not in torch.cuda.get_arch_list():
  raise RuntimeError('GPU architecture sm_120 absent from torch build. Install CUDA 12.8+ PyTorch >=2.7 in a CLONED conda environment.')
 x=torch.randn(1,4,8,12,12,device='cuda',requires_grad=True)
 conv=torch.nn.Conv3d(4,12,3,padding=1).cuda()
 y=conv(x).square().mean();y.backward();torch.cuda.synchronize()
 print('CUDA Conv3D forward/backward: PASS',float(y.detach().cpu()))
except Exception as exc:
 print('CUDA_COMPATIBILITY_CHECK_FAILED:',repr(exc))
 sys.exit(2)
