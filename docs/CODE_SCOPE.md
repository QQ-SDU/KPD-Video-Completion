# Public code scope and consolidation

The default branch is the maintained research-code release for the accompanying paper, not a chronological development archive.

## Consolidations

- One canonical algorithm implementation lives in `src/ours_v7/`. The duplicate historical `benchmark_suite/Methods/Ours/reference_v7` has been removed.
- The canonical CNN implementation retains the `spatial_kernel` argument (`3` or `5`) needed by the supplementary kernel study; default `3` reproduces the original architecture.
- Both same-mask CLI entry points now load the identical canonical core, rather than two potentially divergent copies.
- Older DAVIS grayscale, preliminary 360p scheduling, and obsolete summary scripts have been removed.
- `experiments/09_davis2017_all9/davis9.py` remains as a dependency of the final RGB preparation code.
- The 8 evaluation-video defaults reflect the paper; Blackswan remains development only.

## Remaining intentional modules

`benchmark_suite/workflow/` and adapters are retained for the grayscale/structured-mask experiments and metric aggregation. These adapters do **not** bundle or independently implement external reference methods; a full multi-method comparison requires original authors' software.

The final `experiments/19_final_heterogeneous_360p/` orchestration can consume external baseline outputs for paper tables/videos, but using `scripts/reproduce_rgb_ours.py` does not run those baselines. No full clean GPU benchmark has been claimed.

The original release stays available through the repository Git history.
