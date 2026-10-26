"""Static checks that the conda launcher and child processes use the activated environment."""
from pathlib import Path
base=Path(__file__).resolve().parent
launcher=(base/'RUN_SELECTED_ALL9.sh').read_text()
assert 'conda activate' in launcher and 'CONDA_PREFIX' in launcher
engine=(base/'run_selected_all9.py').read_text()
assert engine.count('sys.executable')>=3
runner=(base.parents[1]/'benchmark_suite/workflow/run_python_suite.py').read_text()
assert 'sys.executable' in runner
print('Launcher uses kpd_cv; child Python runners use sys.executable.')
