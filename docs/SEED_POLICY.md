# Random-seed policy

## DAVIS 2017 color experiments (RGB 360p)

The **fixed seed is `20261014`**, consistently stated in `FINAL_FROZEN_CONFIG.json`, `HETEROGENEOUS_V15_CONFIG.json`, `experiments/10_davis2017_rgb_all9/rgb9.py` and `experiments/17_frozen_rgb_360p_reuse/frozen_benchmark.py`.

- **Observation mask:** initialize `numpy.random.default_rng(20261014)` separately for each mask construction. Select exactly `round(H * W * sampling_rate)` distinct spatial indices without replacement; reshape to `(H, W)`. Broadcast the resulting mask along all 50 frames and all three RGB channels. The implementation is `benchmark_suite/Benchmark/benchmark/io_utils.py::make_tube_mask`.
- **Sampling rates:** 0.2, 0.3, 0.4 (the observed fraction, **not** the missing fraction).
- **RGB representation:** the input is `(H, W, T, 3)` after RGB conversion, normalized to [0,1]. Reference 360p shape is `(360, 640, 50, 3)`.
- **Model training:** each per-channel solver is invoked with `--seed 20261014`; source code seeds Python, NumPy and PyTorch. This seeds stochastic routines but by itself does **not** guarantee bitwise identical CUDA results across hardware/framework versions.
- **Training configuration:** see `FINAL_FROZEN_CONFIG.json` and the 360p reproduction guide. The optional heterogeneous CPU/GPU scheduler does not change this method configuration.
- **Development split:** Blackswan is a parameter-development video. Do not treat it as a previously unseen test sequence.

Run `python tools/verify_rgb_mask_seed.py` to print deterministic mask counts and digests for an independently regenerated two-dimensional mask. The digests are useful for comparing environments; the printed example was computed with NumPy 2.3.5, whereas `requirements.txt` recommends NumPy 1.26.4, so confirm hashes in your own pinned environment rather than assuming cross-version binary equality.

## Other experiments

Some grayscale and ablation runners use a supplied `--seed` or generate a new seed and record it when omitted. Do **not** assert that all paper experiments share the RGB seed. Follow the explicit per-experiment runner and retain its seed/log. The algorithm's architecture parameters are defined separately in `IMPLEMENTATION_CONFIG.json`.
