"""Preflight: check server dependencies, data, 9-method source availability."""
from pathlib import Path
import sys,shutil,importlib.util,os,argparse
parser=argparse.ArgumentParser(description='Preflight verification for all nine DAVIS baselines')
parser.add_argument('--strict-nine',action='store_true',help='require MATLAB and named Conda environment too')
args=parser.parse_args()
expected=os.environ.get('KPD_CV_ENV_NAME','kpd_cv')
actual=os.environ.get('CONDA_DEFAULT_ENV','')
print('Python executable:',sys.executable)
print('Conda environment:',actual or '(not activated)', 'expected:', expected)
print('Conda prefix:',os.environ.get('CONDA_PREFIX','(unknown)'))
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmark_suite'
needs={
    'DRO-TFF':B/'Benchmark/adapters/dro_tff/run_drotff_same_mask.py',
    'HLRTF':B/'Methods/HLRTF/benchmark_runner_hlrtf.py',
    'KBR':B/'Benchmark/adapters/kbr_matlab/run_kbr_same_mask.m',
    'LRTFR':B/'Methods/LRTFR/run_same_mask.py',
    'OTLRM':B/'Methods/OTLRM/run_same_mask.py',
    'TNN':B/'Benchmark/adapters/tnn_matlab/run_tnn_same_mask.m',
    'SALTS':B/'Benchmark/adapters/salts/run_salts_same_mask.m',
    'S2NTNN':B/'Methods/S2NTNN/benchmark_runner_s2ntnn.py',
    'Ours':B/'Methods/Ours/run_same_mask.py'}
fail=bool(args.strict_nine and actual!=expected)
if actual!=expected: print("WARNING: This shell is not using the requested conda environment",expected)
for name,path in needs.items():
    ok=path.exists();print(name,'SOURCE_OK' if ok else 'SOURCE_MISSING',path.relative_to(ROOT));fail|=not ok
for pkg in ['numpy','scipy','skimage','cv2','torch','matplotlib','PIL']:
    ok=importlib.util.find_spec(pkg) is not None
    print(pkg,'OK' if ok else 'MISSING');fail|=not ok
try:
    import torch
    print('PyTorch',torch.__version__,'CUDA',torch.cuda.is_available())
    if torch.cuda.is_available():print('GPU',torch.cuda.get_device_name(0))
    else:print('WARNING: S2NTNN and other learning baselines require CUDA')
except Exception as e:print('PyTorch_ERROR',e);fail=True
matlab_bin=os.environ.get('MATLAB_BIN','matlab')
matlab_path=shutil.which(matlab_bin)
print('MATLAB',matlab_path or 'NOT_FOUND (set MATLAB_BIN or add MATLAB to PATH)')
if args.strict_nine and not matlab_path:
    print('ERROR: all nine methods require an available MATLAB executable for KBR/TNN/SALTS')
    fail=True
print('Freeze config',B/'config/method_parameters.json')
print('RESULT:', 'SOFTWARE SOURCES PRESENT' if not fail else 'CHECK DEPENDENCIES / SOURCES')
if fail:sys.exit(1)
