# 08 DAVIS2017 独立泛化实验（无需重新调参）

该目录针对当前论文第 V 节设计，只使用 **DAVIS 的视频画面**，不使用分割标签，不涉及 DAVIS 分割任务。

## 0. 数据集获取

官方地址：https://davischallenge.org/davis2017/code.html

在官方下载 `DAVIS-2017-trainval-480p.zip`，解压后确认存在 `DAVIS/JPEGImages/480p/blackswan/00000.jpg` 等目录。
**外部视频版权归原始发布方，本实验包不包含 DAVIS 视频。**

推荐8个 validation 视频：blackswan, bmx-trees, breakdance, camel, car-roundabout, drift-chicane, kite-surf, parkour。
这些名字与已有23-video manifest 不重合。评测前不要根据这些视频测试结果选择子集。

## 1. 安装 Python 依赖

直接使用项目原有环境，另外需 `opencv-python`、`numpy`、`scipy`、`torch`、`scikit-image`。Python>=3.10。

## 2. 生成新的视频张量

在完整工程根目录运行（不要在本目录中单独调用）：

```bat
python experiments/08_davis2017_generalization/prepare_davis.py --davis-root "D:\datasets\DAVIS2017"
```

生成 `benchmark_suite/data/davis2017_new/{name}.npy` 和 `manifest.csv`、`PREPROCESSING.json`。
格式固定为 144x256x50、HWT、灰度 `[0,1]`，保留前50帧，不作标签训练。

## 3. 正式复现实验（GPU 推荐）

```bat
python experiments/08_davis2017_generalization/run_davis_frozen.py --device auto
python experiments/08_davis2017_generalization/summarize_davis.py --input results/davis2017_new
```

默认对 8 视频 × 2 个 SR × 3 个固定 seed × 3 个方法 (KPD, DIRECT, CROSSFIT) 运行 **144 个 method-case**。同一视频/采样率/seed 的三种方法共享完全相同的观测掩码。保留论文 Table I 超参数：K=4、S=12、F=5、CNN1200步、隐藏通道12。**全观测 tubes 被用于最终 CNN 训练，不做内部调参。**

若资源有限，先按以下顺序做预检查、再做完整：

```bat
python experiments/08_davis2017_generalization/run_davis_frozen.py --device auto --videos blackswan --rates 0.2 --seeds 20261014
```

该运行采用正式参数，可以先判断性能是否符合预期；**正式论文仍应完成预先确定的全部8个视频及所有种子，不能只报告有利视频**。

### 代码自检（严禁当作论文数值）

```bat
python experiments/08_davis2017_generalization/run_davis_frozen.py --smoke-test --data "测试数据目录" --out results/davis_smoketest --videos synthetic --rates 0.3 --seeds 11
```

`--smoke-test` 会改变 KPD rank/sweeps、CNN steps 等，只用于确认代码可运行；结果禁止写入论文。

## 4. 结果

- `results/davis2017_new/per_case/*.json` 每视频/采样率/种子的逐方法指标和耗时。
- `PER_CASE_METRICS.csv` 完整结果。
- `AGGREGATE.csv` 每个 SR/方法上的 mean±std。
- `PAIRED_CROSSFIT_VS_DIRECT.csv` 配对差值，`PER_VIDEO_MEANS.csv` 与 `PAIRED_GAIN_BY_VIDEO.csv` 提供按视频汇总，避免把同一视频的多种子运行误当成完全独立样本。

论文中报告全视频 PSNR、缺失位置 PSNR、SSIM，并附 CrossFit 对 Direct 的配对均值和胜出比例；方法排名只比较实际测过的方法。时间计数主要覆盖模型拟合/训练/推理，读入和指标计算不计入。

## 5. 适用边界

本实验在 DAVIS 视频上仍沿用固定空间 tube-wise 掩码，不等于 video object removal 或任意随时间变化遮挡；更不能报告 DAVIS 官方分割指标。
输入从480p缩小到 144×256 是额外实验设计，并非现有23-video benchmark 的官方分辨率。

IMPORTANT：所有完整运行之前，使用同一个 `--data`、`--out`，不要把 `--smoke-test` 结果写入正式结果目录；所有指标都只能由真实运行生成，不提供预填结果。

## 6. 完整结果后生成论文用 LaTeX 小节

只有完成全部8视频×2 SR×3 seed×3方法之后，才运行：

```bat
python experiments/08_davis2017_generalization/make_davis_tex.py --results results/davis2017_new
```

脚本会拒绝 smoke test 和任何不完整结果。成功后生成 `results/davis2017_new/DAVIS_SECTION.tex`，可以放入论文 Section V 结束前，原文参数表交叉引用 `tab:reference-config` 已匹配。
