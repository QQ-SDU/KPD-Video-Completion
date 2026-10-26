# Reproduction instructions and limitations

## Scope

The research package contains (a) a masked KPD / cross-fitted residual-learning
implementation, (b) paper experiment scripts, and (c) shared-mask benchmark
orchestration. This public candidate intentionally excludes third-party algorithm
implementations, datasets, original benchmark outputs and weights. End-to-end
benchmark reproduction is **not validated here**.

## Environment

Create a clean Python 3.10 environment; install `requirements.txt`. PyTorch must
be installed separately with a build matching the target GPU/CUDA driver; see
https://pytorch.org/get-started/locally/ . MATLAB is separately needed for KBR,
TNN and SALTS. Verify any method-specific requirements from upstream packages.

## Non-dataset static check

```bash
python tools/validate_release.py
```

## Synthetic smoke tests (not paper benchmark results)

```bash
python experiments/19_final_heterogeneous_360p/smoke_heterogeneous.py
python experiments/19_final_heterogeneous_360p/smoke_concurrent.py
```

Synthetic tests examine selected I/O / plotting / video rendering and the hybrid
scheduler, respectively. They do not reproduce the DAVIS quantitative tables or
the third-party MATLAB baselines.

## DAVIS 360p v15 runner

```bash
python experiments/19_final_heterogeneous_360p/final_davis.py list --davis-root /path/to/DAVIS2017
python experiments/19_final_heterogeneous_360p/final_davis.py run \
  --davis-root /path/to/DAVIS2017 \
  --source /path/to/benchmark_results \
  --videos camel,parkour --jobs 2 --matlab-threads 4 --require-complete
```

These commands require the third-party methods, compatible environment and
sufficient CPU/GPU/memory resources. The runner may reuse older results; reuse
must be checked against video input, seed, method configs, and shared masks.
For Windows, write the command on a single line or use PowerShell backticks.
Pass `--source`, `--v12-results` and `--v14-results` where appropriate rather than
relying on old workstation-specific paths. `--dry-run` does not guarantee all
external dependencies work.

## Two distinct published configuration families

**Development and grayscale benchmarks**: `IMPLEMENTATION_CONFIG.json` specifies
KPD rank K=4, crossfit folds=5, CNN hidden channels=12 and 1200 steps, among
other parameters. Seeds depend on the top-level runner; see `docs/SEED_POLICY.md`.

**DAVIS RGB 360p frozen experiments**: `FINAL_FROZEN_CONFIG.json` specifies
K=12, hidden channels=32, 3600 steps, five CrossFit folds, patch 16x96x96,
common mask seed 20261014, and three sampling rates. Hyperparameters were tuned
on a development video (Blackswan) and then frozen; do not describe Blackswan
as an independent untouched test sequence.

**DAVIS v15 scheduler**: `HETEROGENEOUS_V15_CONFIG.json` describes 4 MATLAB CPU
jobs + 1 serial Python lane as a scheduling option, not an algorithmic change.

## Paper mapping

See `PAPER_CODE_MAPPING.md`. Confirm the mapping against the final TeX revision;
it was copied from the provided code package, not auto-derived from the PDF.

## Accuracy and reproducibility caveat

Even a passing smoke test is not a substitute for recovering the paper tables,
checking common masks and metric definitions, and documenting CUDA/PyTorch and
MATLAB versions. Retain raw logs and configure output paths outside the repository.
