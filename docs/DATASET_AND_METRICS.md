# Data and metric protocol

## 23-video grayscale benchmark

Use the 23 files listed in `FORMAL23_MANIFEST.csv` under `benchmark_suite/data/formal23/`.

- 17 QCIF videos: `144 x 176 x 100` (first 100 frames).
- 6 CIF videos: `288 x 352 x 50` (first 50 frames).
- Tube-wise exact-count spatial masks.
- SR = 0.1, 0.2, 0.3, 0.4, 0.5.
- The same saved mask is used by every method for each case.

Main benchmark metrics in the current code:

- Paper PSNR: `psnr_full` (full reconstructed video).
- Paper SSIM: `ssim_frame_mean` (frame-wise full-frame SSIM averaged over time).
- Temporal-gradient MAE: `temporal_gradient_mae_missing` (missing tubes only).
- Runtime: wall-clock time reported by each method adapter and aggregated per sequence.

The evaluator additionally computes missing-entry PSNR/MAE, but those are not the PSNR values used in Table IV.

## Development protocol

An 80/20 split of observed tubes is used for development training / strict validation.
True missing ground truth is diagnostic only and is not used to select hyperparameters.

## Structured color experiment

Paper subset:

- Coastguard
- Container
- masks: `block`, `grid`
- approximately 6% missing spatial tubes
- one 2-D mask shared across all frames and RGB channels
- channels reconstructed independently and reassembled as RGB

Metrics:

- PSNR: missing RGB entries
- SSIM: frame-wise RGB full-frame SSIM
