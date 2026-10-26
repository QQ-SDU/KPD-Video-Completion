# V7 理论—实现一致性修改说明

本版本针对论文数学检查中发现的四个关键问题进行了同步修改。

## 1. KPD/CP 术语边界

论文明确说明三阶表示

`X_K = sum_r a_r o b_r o c_r`

在代数上等价于 rank-K CP 表示；本文保留 “Kronecker-structured KPD backbone” 的表述，是因为每个向量化 rank-one 分量满足 `vec(a o b o c)=c kron b kron a`。创新点不再放在“提出一种不同于 CP 的张量秩模型”，而放在 masked regularized fitting + cross-fitted residual learning。

## 2. 单调性与 factor balancing

旧代码在每个分量更新后把 `b,c` 强制归一化，并把尺度推给 `a`。该操作保持 rank-one 外积不变，却可能改变并增大正则项，因此不能由论文 Proposition 1 保证单调性。

V7 改为 **regularizer-optimal scale balancing**：在 `alpha*beta*gamma=1` 下最小化三因子的正则代价。该步骤保持 rank-one 外积不变，并保证完整目标函数不增。

`masked_kpd.py` 现在：
- 使用 `_regularizer_optimal_balance`；
- 每个 sweep 计算论文中的完整 masked objective；
- 默认检查 objective 非增，否则直接报错；
- 使用 tridiagonal banded solver，对应论文复杂度分析。

## 3. Cross-fitting no-leakage

`crossfit_tube_refiner.py` 不再接收完整 GT。它只接收观测张量 `observed`（真正缺失位置为 0）和 `train_mask`。

每个 fold：
1. 由 context mask 构造 `obs_context`；
2. warm initialization 只由 `obs_context` 产生；
3. fold KPD 只接收 `obs_context`；
4. pseudo-missing target 只从已观测的 target fold 中读取；
5. CNN 训练函数只接收预先构造的 cross-fit states，不接收 GT。

因此论文 no-leakage 假设在软件接口层也得到实现。

## 4. Raw KPD 与 bounded backbone

优化变量仍产生 raw reconstruction `X_tilde_K`。残差学习实际使用

`B_K = clip(X_tilde_K, 0, 1)`。

代码和论文现在统一：
- `raw_reconstruction`：未经范围约束的因子重构；
- `backbone`：`clip(raw_reconstruction,0,1)`；
- `reconstruction`：最终 data-consistent KPD 输出（观测位置覆盖回真实观测）。

## 5. development / final 两种协议

`run_v51_one_case.py` 新增：

- `--protocol development`：保留 strict validation，用于超参数和 residual-distribution 诊断；
- `--protocol final`：超参数固定后，所有 observed tubes 都进入 cross-fitting pool，以减小 cross-fit/final context-size shift。

23-video 主实验 `run_v6_tableiv_23video.py` 已改为自动调用 `--protocol final`。

## 推荐执行顺序

### A. 先做最小验证

双击：

`RUN_SMALL_AKIYO_VALIDATION.bat`

它会：
1. 若没有 Akiyo，则只下载 Akiyo；
2. 先运行 `verify_theory_alignment.py`；
3. 再用 72x88、30 帧、K=5、4 sweeps、3 folds、120 CNN steps 做一个快速 development 验证。

重点检查输出：
- 控制台三个 `KPD monotone` 均为 True；
- `KPD_FULL_HISTORY.csv` 的 `objective` 单调下降；
- `SUMMARY.json` 中 `true_missing_gt_used_in_kpd_or_cnn_training=false`；
- `METHODS.csv` 中 CrossFit-CNN 与 KPD 的 PSNR 差值。小实验只用于确认流程，不要求一定提升。

### B. 小视频 final 协议

```bat
python run_v51_one_case.py --input data\akiyo_qcif.y4m --video-name Akiyo-small-final --output validation_akiyo_sr20_final --seed 1 --sampling-rate 0.20 --resize 72x88 --max-frames 30 --rank 5 --masked-sweeps 4 --protocol final --crossfit-folds 3 --steps 120 --patch-t 12 --patch-h 32 --patch-w 32 --batch-size 1 --hidden 8 --device auto --show-frames 5,15,25
```

final 协议没有 strict-validation 指标，这是有意的：此时所有 observed tubes 都用于 cross-fitting pool。

### C. 正式 23-video benchmark

确认小验证无错误后，再运行原批处理脚本。V7 的 `run_v6_tableiv_23video.py` 已固定使用 final protocol。

