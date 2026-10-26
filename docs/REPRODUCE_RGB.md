# Reproduce the DAVIS 2017 RGB results (our method)

This document describes **our KPD/CrossFit/CNN method**, not the implementations of the external baselines. The code uses the exact RGB fixed seed and frozen hyperparameters recorded in the experiment configuration. The full reference comparison requires separate baseline installations.

## 1. Prepare environment

From the repository root, use Python 3.10, install `requirements.txt`, then install a compatible PyTorch build. Check GPU memory and disk capacity: one 360p RGB video with three observation rates, five crossfit folds, per-channel CNNs and 3600 training steps can be expensive.

## 2. Download the dataset

Get the **DAVIS 2017** RGB JPEG frame sequences under the official dataset's terms. Expected root: `/path/to/DAVIS2017/JPEGImages/480p/<video>/*.jpg` (the loader also accepts the nested `DAVIS` and `2017` layout variants). Do not upload the original dataset to this repository.

## 3. Run from a clean directory

```bash
python tools/verify_rgb_mask_seed.py
python scripts/reproduce_rgb_ours.py --davis-root /path/to/DAVIS2017 --videos camel,parkour --work-dir ./local_runs/rgb360 --device cuda
```

A staged pipeline is also possible:

```bash
python scripts/reproduce_rgb_ours.py --davis-root /path/to/DAVIS2017 --videos camel,parkour --work-dir ./local_runs/rgb360 --stage prepare
python scripts/reproduce_rgb_ours.py --davis-root /path/to/DAVIS2017 --videos camel,parkour --work-dir ./local_runs/rgb360 --stage fit --device cuda
```

The wrapper runs (1) `prepare_davis_rgb.py` at `360 x 640`, using the first 50 RGB frames, (2) `rgb9.py prepare` to generate shared tube-wise masks and per-channel observed cases, then (3) `frozen_benchmark.py --fit-only --force-retrain` to fit our method from scratch. The final step does not launch the third-party algorithms. The generated RGB reconstructions will be under `local_runs/rgb360/ours_frozen/rgb/K12_C32_T3600_P96/`.

This code path was checked by source inspection and syntax checks, **not** by a full clean GPU benchmark run in this environment. Do not promise identical final PSNR/SSIM until full experiments are rerun. An optional `--dry-run` prints commands only.

The supporting RGB visualizations cover **bear, drift-chicane, cows, dog, camel, kite-surf, parkour, and breakdance** at all three observed sampling rates. Those video names correspond to the submitted supplementary MP4 filenames; this repository does not include their source clips or their external-baseline reconstructions. To reproduce our method on all eight, use `--videos bear,drift-chicane,cows,dog,camel,kite-surf,parkour,breakdance` if all sequences are present in the installed DAVIS release.

## 4. Fixed protocol

| Setting | Value |
|---|---|
| Seed | **20261014** |
| Input | RGB 360 x 640 x first 50 frames |
| Rates | 0.2, 0.3, 0.4 (observed) |
| Mask | exact-count spatial tube mask, identical in time and across RGB channels |
| KPD rank | 12 |
| Cross-fitting folds | 5 |
| Hidden width | 32 |
| Convolution kernel | 3 x 3 x 3 |
| Training steps | 3600 |
| Patch | 16 x 96 x 96 |
| Masked KPD sweeps | 12 |
| Batch size | 2 |
| Maximum residual | 0.25 |

Hyperparameters were selected using the development video Blackswan. The other videos must not be used for tuning if the original split protocol is to be preserved. Report the actual dataset version, hashes, NumPy, PyTorch, CUDA and device versions alongside numerical results.

## 5. Evaluation

The GT video is used for evaluation/plotting and is not passed into the solver subprocess. Use the existing evaluation routines with the saved observed cases and completed reconstructions. Avoid labeling an Ours-only run as a nine-method comparison. To reproduce external baselines, obtain each original repository and its dependencies separately; see `docs/THIRD_PARTY_BASELINES.md`.
