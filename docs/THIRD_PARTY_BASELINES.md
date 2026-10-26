# Third-party baselines — sources and redistribution review

The original internal research workspace contained copies of third-party benchmark methods,
including a substantial MATLAB Tensor Toolbox tree. This repository **does
not redistribute those upstream implementations**. The user-project benchmark
orchestration, metrics and known custom wrappers are retained. Therefore a complete
9-method reproduction **will not run** until the official external baseline code is
obtained and installed in the expected paths and the required MATLAB dependencies
are made available.

| Method | Upstream source as documented in supplied package | Expected local directory |
|---|---|---|
| DRO-TFF | https://github.com/Xbow-Git/DRO-TFF | `benchmark_suite/Methods/DRO-TFF/` |
| HLRTF | https://github.com/YisiLuo/HLRTF | `benchmark_suite/Methods/HLRTF/` |
| KBR | https://github.com/XieQi2015/KBR-TC-and-RPCA | `benchmark_suite/Methods/KBR/` |
| LRTFR | https://github.com/YisiLuo/Continuous-Tensor-Toolbox | `benchmark_suite/Methods/LRTFR/` |
| OTLRM | https://github.com/xianggkl/OTLRM | `benchmark_suite/Methods/OTLRM/` |
| TNN | https://github.com/canyilu/tensor-completion-tensor-recovery | `benchmark_suite/Methods/TNN/` |
| S2NTNN | https://github.com/YisiLuo/S2NTNN | `benchmark_suite/Methods/S2NTNN/` |
| SALTS | The original package contains SALTS source and MATLAB Tensor Toolbox; no verified official SALTS repository URL was documented in the inspected provenance notes. | `benchmark_suite/Methods/SALTS/` |

The versions and methods recorded above are **source attributions from internal experiment notes**, not newly verified upstream installs or a confirmed legal licensing review.
Where supplied: LRTFR source commit `be8e625ff66d6c2ead345a71b53e6c708ce1a148`,
OTLRM source commit `ee9e97924fde4dad1a841746df9889f1acdf1a37`.

Notes:
- The `run_same_mask.py` / `benchmark_runner_*.py` files remaining in some `Methods/`
  folders are **integration adapters**, not copies of complete official projects.
- Clone/download the original method **separately**, observe its own license, and
  merge or adapt its expected files cautiously. Placing a repository wholesale
  under the path may not match the original benchmark layout. Validate each
  adapter's relative imports and parameters before claiming results are reproduced.
- KBR/SALTS MATLAB Tensor Toolbox dependencies are **not** included.
- For exact paper comparisons, retain versions, seed and shared masks and report
  any deviations from the supplied protocol.
- License / authorship of each custom adapter also requires review for reuse in new environments.
