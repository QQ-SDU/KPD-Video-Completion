# Paper-to-code mapping

This index maps the paper's mathematics, experiments, and supplementary analyses to the **retained** source. It does not claim independent full-benchmark replication.

| Paper item | Source / entry point |
|---|---|
| Masked regularized KPD (Eqs. 20, 26–32, 38; Algorithm 1) | `src/ours_v7/masked_kpd.py`, `src/ours_v7/kpd_video_core.py` |
| Tube-wise cross-fitting (Eqs. 46–52) | `src/ours_v7/crossfit_tube_refiner.py` |
| Four-channel CNN and residual loss (Eqs. 53–63) | `src/ours_v7/detail_refiner_v41.py`, `src/ours_v7/detail_refiner_tube.py` |
| Final inference / data consistency (Eqs. 64–68; Algorithm 2) | `src/ours_v7/run_same_mask_benchmark_adapter.py`, `benchmark_suite/Methods/Ours/run_same_mask.py` |
| Fig. 1: representation efficiency | `experiments/01_representation_efficiency/run_fig1_3dev.py` |
| Table II: residual supervision | `experiments/common_ablation/run_paper_ablation.py` (`--study methods`) |
| Table III: input ablation | `experiments/common_ablation/run_paper_ablation.py` (`--study inputs`) |
| Fig. 3: stage-wise visualization | `experiments/05_stagewise_visual/RUN_STAGEWISE_DATA.bat` |
| Fig. 4: sensitivity | `experiments/04_sensitivity/RUN_FIG4_ALL.bat` |
| Table IV and grayscale 23-video curves | `experiments/06_benchmark_23videos/`, `benchmark_suite/workflow/` |
| Structured missing-mask tests | `experiments/07_color_structured/`, `benchmark_suite/color_structured9/` |
| DAVIS RGB 360p data preparation and fixed seed | `experiments/10_davis2017_rgb_all9/prepare_davis_rgb.py`, `rgb9.py` |
| DAVIS RGB method-only replication | `scripts/reproduce_rgb_ours.py`, `experiments/17_frozen_rgb_360p_reuse/frozen_benchmark.py` |
| Supplement: rank × kernel | `experiments/13_ours_rank_kernel_360p/rank_kernel_rgb360.py` |
| Supplement: rank × CNN width | `experiments/14_ours_rank_width_360p/rank_width_rgb360.py` |
| Supplement: training steps × patch | `experiments/15_ours_steps_patch_360p/steps_patch_rgb360.py` |
| Supplement: 8-sequence RGB evaluation / tables / videos | `experiments/19_final_heterogeneous_360p/final_davis.py`, `video_supplement.py` |

The RGB preparation utility `experiments/09_davis2017_all9/davis9.py` is retained because `rgb9.py` imports its shared manifest, naming, and reporting helpers; its old scheduling scripts are excluded.

## Experiment configuration

- Grayscale benchmark: `IMPLEMENTATION_CONFIG.json` (default KPD rank 4).
- RGB 360p fixed configuration: `FINAL_FROZEN_CONFIG.json`, `HETEROGENEOUS_V15_CONFIG.json` (rank 12, CrossFit folds 5, 3600 steps).
- RGB seed: **20261014** for the tube masks and model initialization; see `docs/SEED_POLICY.md`.
- The evaluation eight are **bear, drift-chicane, cows, dog, camel, kite-surf, parkour, breakdance**; **blackswan** is the parameter-development sequence and must not be counted among the evaluation eight.

External baseline solvers, original DAVIS data, and supplementary videos are not bundled. See `docs/THIRD_PARTY_BASELINES.md`.
