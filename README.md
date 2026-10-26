# Kronecker-Structured Video Completion with Cross-Fitted Self-Supervised Residual Learning

**Official research-code repository for the paper** *Kronecker-Structured Video Completion with Cross-Fitted Self-Supervised Residual Learning*.

Authors: Qi Qi, Yanfei Wang, Yanjun Cui, and Jun-e Feng.

## Method

The method reconstructs videos from tube-wise missing observations using a regularized masked Kronecker product decomposition (KPD) backbone and a self-supervised 3-D CNN residual refiner. Observed tubes are divided into folds so that residual targets are based on held-out structural predictions. The final reconstruction enforces exact data consistency on observed entries.

## Contents

| Path | Description |
|---|---|
| `src/ours_v7/` | Masked KPD, cross-fitting and residual CNN source |
| `experiments/` | Paper experiments, ablations and DAVIS RGB workflows |
| `benchmark_suite/` | Mask generation, experiment evaluation and adapters (no third-party solver sources) |
| `scripts/reproduce_rgb_ours.py` | Standalone orchestration for **our** DAVIS RGB pipeline |
| `FINAL_FROZEN_CONFIG.json` | Final RGB parameter settings |
| `docs/REPRODUCE_RGB.md` | Step-by-step RGB reproducibility protocol |
| `docs/SEED_POLICY.md` | Exact seed policy, including mask semantics |
| `PAPER_CODE_MAPPING.md` | Manuscript-to-code lookup |

This repository **does not distribute** any third-party baseline implementation, dataset, pretrained weights, or supplementary MP4 files. Existing integration adapters are not implementations of the external algorithms. For external comparisons, use the original authors' releases and licenses; see `docs/THIRD_PARTY_BASELINES.md`.

## Environment

Python 3.10 is recommended for the original code. Install required scientific packages:

```bash
conda create -n kpd-video python=3.10 -y
conda activate kpd-video
python -m pip install -r requirements.txt
```

Install PyTorch **separately** from the [official instructions](https://pytorch.org/get-started/locally/) with a build that matches your GPU/driver (or a CPU build). `requirements-torch.txt` is advisory rather than a fully pinned CUDA environment. Computing all 360p cases is GPU-memory and time intensive.

## DAVIS 2017 RGB replication

The reported color experiment uses **seed `20261014`** for both the shared tube-wise observation mask and the neural model's seeded initialization. The same 2-D spatial mask is reused across all frames and RGB channels for each (height, width, sampling rate). The 360p setting uses the first 50 frames, resized to `360 x 640`, at observed sampling rates `0.2, 0.3, 0.4`.

The frozen configuration uses KPD rank **12**, five cross-fitting folds, CNN width **32**, spatial/temporal kernels **3 x 3 x 3**, **3600** training steps, patch shape **16 x 96 x 96**, 12 masked KPD sweeps, batch size **2** and maximum residual **0.25**. This differs from the grayscale/development settings in `IMPLEMENTATION_CONFIG.json`.

Example from the repository root (requires a separately downloaded DAVIS 2017 dataset):

```bash
python tools/verify_rgb_mask_seed.py
python scripts/reproduce_rgb_ours.py \
  --davis-root /path/to/DAVIS2017 \
  --videos camel,parkour \
  --work-dir ./local_runs/rgb360 \
  --device cuda
```

On Windows PowerShell, place the example command on one line or use PowerShell backticks instead of Unix backslashes. Use `--dry-run` to print the planned commands without executing expensive training. The script prepares common-mask RGB cases and fits **our method only**. It does **not** reproduce third-party baselines or the full comparison table. More details: [RGB reproduction guide](docs/REPRODUCE_RGB.md).

## Basic validation

```bash
python tools/validate_release.py
python experiments/19_final_heterogeneous_360p/smoke_heterogeneous.py
python experiments/19_final_heterogeneous_360p/smoke_concurrent.py
```

These checks only verify source consistency and selected synthetic workflows; they do not validate the numerical findings on the entire paper benchmark. Exact floating-point identity on GPU is not guaranteed by a seed alone. Please retain package versions, GPU specifications and execution logs.

## Data, baselines and citation

The DAVIS 2017 data are acquired from their official source and remain subject to their own terms. `docs/DATA_SETUP.md` explains the expected layout; `docs/DATASET_AND_METRICS.md` describes the grayscale and structured-mask protocols. The Blackswan clip was used for parameter development and should **not** be counted as an untouched evaluation sequence.

Please cite the accompanying paper when using the code; the `CITATION.cff` file contains author and title information without asserting a journal publication status.

## License

Project-authored code in this repository is released under the [MIT License](LICENSE). This license **does not** extend to external benchmark implementations or datasets, which are not bundled here. Historical notes and third-party attribution are retained when relevant.
