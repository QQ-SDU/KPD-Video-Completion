# DAVIS 2017：9种方法统一实验（v5，单随机种子，使用现有 kpd_cv）

## 实验协议

- 固定一个种子：`20261014`。同一个视频、同一个采样率下，九种方法使用**完全相同**的随机空间tube掩码。不同视频也使用同一个固定基础种子，不进行多种子重复。
- 固定 **采样率 SR=0.2、0.3、0.4**，即随机观测20%、30%、40%的空间tube，剩余位置缺失。不是缺失率0.2/0.3/0.4。
- 8段视频候选：`blackswan,bmx-trees,breakdance,camel,car-roundabout,drift-chicane,kite-surf,parkour`，默认灰度144×256×50。使用 DAVIS 2017 原始帧，不使用分割标签。
- 九种方法：**DRO-TFF、HLRTF、KBR、LRTFR、OTLRM、TNN、SALTS、S2NTNN、Ours**。一次运行指定视频后，自动串行执行该视频三个采样率对应的全部9种算法。
- 方法超参数保持论文已有冻结配置。不同视频的数据只改变视频输入，不重新调整网络或基线超参数。

## 0. 使用已配置的 Conda 环境 `kpd_cv`（无需重新安装）

本工程直接复用服务器已经配置好的 `kpd_cv`。不要创建新环境，也不要直接在已有环境中执行 `pip install -r ...` 覆盖原有依赖版本。

```bash
conda activate kpd_cv
which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 包含九种方法的服务器完整依赖检查，MATLAB 不可用时给出错误
bash experiments/09_davis2017_all9/RUN_SELECTED_ALL9.sh --check-env
```

`RUN_SELECTED_ALL9.sh` 在没有激活环境时尝试自动激活 `kpd_cv`，并使用该环境的 `bin/python`；所有 Python 基线及 Ours 的子进程继承同一解释器。Linux 系统中的 MATLAB 需独立安装；如不在 PATH 中，设置 `export MATLAB_BIN=/path/to/matlab`。

## 1. 数据位置及检查

建议 Linux+CUDA GPU+MATLAB（KBR/TNN/SALTS）+可用 MATLAB License。Python 工程运行在现有 `kpd_cv` 中；启动前做一次依赖检查，如缺包再单独处理，不要批量重新安装。

将 DAVIS 解压至 `/datasets/DAVIS`（内部存在 `JPEGImages/480p/blackswan` 等子目录），在**工程根目录**运行：

```bash
export DAVIS_ROOT=/datasets/DAVIS
python experiments/09_davis2017_all9/check_server.py
```

环境检查可在不具备 CUDA 时显示提醒，但正式学习型方法推荐使用 GPU。若 MATLAB 不在 PATH：`export MATLAB_BIN=/path/to/matlab`。

## 2. 按视频选择：最常用的4条命令

**调试一个视频（27个方法任务=1视频×3采样率×9方法）：**

```bash
bash experiments/09_davis2017_all9/RUN_SELECTED_ALL9.sh --videos blackswan
```

**选择两个或多个视频（默认按视频顺序运行，每个视频都跑完九种方法）：**

```bash
bash experiments/09_davis2017_all9/RUN_SELECTED_ALL9.sh --videos blackswan,camel,parkour
```

**同时跑两个视频（并发运行；每个视频内部仍按九种方法依次运行）：**

```bash
bash experiments/09_davis2017_all9/RUN_SELECTED_ALL9.sh --videos blackswan,camel --jobs 2
```

**全部八个视频：**

```bash
bash experiments/09_davis2017_all9/RUN_SELECTED_ALL9.sh --videos all
```

运行`--videos all`时，如此前已经完成若干视频，合法结果会自动跳过，只补齐未完成的部分。

**资源建议：** 默认 `--jobs 1`（安全）。`--jobs 2` 会同时启动两组GPU训练及可能并行的MATLAB实例，可能导致显存不足、内存抢占或许可证数不够。在单卡GPU上建议 `--jobs 1`，先测通，再提高并发数。

## 3. 调试功能

先生成同一随机掩码但**不启动任何方法**：

```bash
bash experiments/09_davis2017_all9/RUN_SELECTED_ALL9.sh --videos blackswan --prepare-only
```

预览该视频3种采样率×9方法的具体运行命令（不求解）：

```bash
bash experiments/09_davis2017_all9/RUN_SELECTED_ALL9.sh --videos blackswan --dry-run
```

如果视频 `.npy` 已经准备好，跳过JPEG预处理：

```bash
bash experiments/09_davis2017_all9/RUN_SELECTED_ALL9.sh --videos camel --skip-preprocess
```

强制重新计算指定视频的9种方法：

```bash
bash experiments/09_davis2017_all9/RUN_SELECTED_ALL9.sh --videos camel --overwrite
```

不需要 `DAVIS_ROOT` 的情况：只有 `--skip-preprocess` 可直接使用之前的 `benchmark_suite/data/davis2017_new/manifest.csv`。

查看帮助：`python experiments/09_davis2017_all9/run_selected_all9.py --help`。默认种子只使用1个；`--seed 20261014` 可显式给出固定种子，不允许一次传入多个种子。更换种子须使用新的 `--output` 目录，以免混合实验。

## 4. 实验结果及日志

默认结果目录：`results/davis2017_all9_single_seed/`

- `cases_index.csv`：可增量追加多次选择的视频，不覆盖已有案例。
- `cases/*.mat`：所有方法共用的原始观测、mask、初始化等输入。
- `reconstructions/METHOD/`：九种方法的输出。
- `logs/METHOD/CASE.log`：每个算法、每个采样率的真实命令日志。
- `video_logs/blackswan.log`：该视频所有27个算法任务的控制台调度日志。
- `task_status.jsonl`：任务状态历史；失败可继续运行，同一命令会跳过成功任务。
- `reports/selected__blackswan/`：仅选中 `blackswan` 时的完整九方法报告。
- `reports/selected__blackswan__camel/`：仅选中这两段视频时的报告。
- 当 `--videos all` 运行结束且八段视频的所有输出完整时，在结果根目录生成 `SUMMARY_COMPLETE.csv` 和 `DAVIS9_TABLE.tex`。

每个选中视频均完成所有方法时，自动评估并生成所选视频集合的 `SUMMARY_COMPLETE.csv`、`DAVIS9_TABLE.tex`；若有失败则只生成 `SUMMARY_PARTIAL.csv` 与 `missing_or_failed.csv`，不能用于正式论文汇总。报告的定量样本仅来自本次选择的视频，不会把之前其他视频混入。

## 5. 预设规模

| 选中视频 | 实验案例（SR×视频） | 方法求解（案例×9） |
|---|---:|---:|
| 1 | 3 | 27 |
| 2 | 6 | 54 |
| 3 | 9 | 81 |
| 全部8 | 24 | 216 |

**一次运行必含9种方法**；`--jobs` 仅控制不同视频之间的并发数量，不会减少比较方法。

## 6. 正式实验和图像选择

正式表格必须包含8段预设视频×3种SR×1个种子的所有结果（216次）。由于只有一个种子，不能从实验中声称“跨随机种子稳定性”。可在全部结果完成后调用：

```bash
python experiments/09_davis2017_all9/davis9.py visualize \
  --output results/davis2017_all9_single_seed
```

完整定量表格仍报告所有预设视频，不因视觉展示时挑选表现较好的序列而删除数据；建议另附一例方法表现不足的案例。

## 7. 测试状态

已通过合成数据的文件接口、mask、汇总、增量追加、命令预览及两视频并行调度测试。**尚未在真实DAVIS视频及服务器MATLAB/GPU环境上执行9种算法的全量计算**。先用一个视频完成27个真实任务，确认各方法正常后再增加视频。
