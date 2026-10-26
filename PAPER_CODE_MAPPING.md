# Paper-to-code mapping

| Paper item | Code in this package |
|---|---|
| Eq. (20), masked regularized KPD objective | `src/ours_v7/masked_kpd.py` |
| Eqs. (26)-(32), cyclic component updates | `src/ours_v7/masked_kpd.py` |
| Eq. (38), regularizer-optimal scale balancing | `src/ours_v7/masked_kpd.py` |
| Algorithm 1 | `src/ours_v7/masked_kpd.py` + `src/ours_v7/kpd_video_core.py` |
| Eqs. (46)-(52), tube-wise cross-fitting | `src/ours_v7/crossfit_tube_refiner.py` |
| Eqs. (53)-(63), four-channel CNN and loss | `src/ours_v7/detail_refiner_v41.py`, `detail_refiner_tube.py`, `crossfit_tube_refiner.py` |
| Eqs. (64)-(68), final inference/data consistency | `src/ours_v7/run_same_mask_benchmark_adapter.py` / benchmark adapter |
| Algorithm 2 | `src/ours_v7/crossfit_tube_refiner.py` + `src/ours_v7/run_same_mask_benchmark_adapter.py` |
| Fig. 1 | `experiments/01_representation_efficiency/` |
| Table II | `experiments/02_residual_supervision_ablation/` + `experiments/common_ablation/` |
| Table III | `experiments/03_input_ablation/` + `experiments/common_ablation/` |
| Fig. 4 | `experiments/04_sensitivity/` |
| Fig. 3 reconstruction ingredients | `experiments/05_stagewise_visual/` |
| Table IV / Fig. 5 / Fig. 6 data | `benchmark_suite/` via `experiments/06_benchmark_23videos/` |
| Table V / Fig. 7 data | `benchmark_suite/color_structured9/` via `experiments/07_color_structured/` |

## Controlled ablations

**Direct**: the KPD backbone is fitted with all development-training observed tubes. Pseudo-target tubes are hidden only from the CNN input, so they are in-sample with respect to KPD.

**CrossFit**: pseudo-target tubes are excluded from the corresponding fold-specific KPD fit before their residual targets are evaluated.

**Input ablation** keeps the same 4-channel CNN architecture and zeros removed channels:

- B = `[B_K, 0, 0, 0]`
- BR = `[B_K, E_c, 0, 0]`
- BRM = `[B_K, E_c, M_c, 0]`
- FULL = `[B_K, E_c, M_c, H_s(B_K)]`


## Random seed

本整理版不在实现代码中固定论文某次运行的 seed。用户可选传入 `--seed`；省略时由顶层 runner 为该次运行生成并打印新的 seed。
