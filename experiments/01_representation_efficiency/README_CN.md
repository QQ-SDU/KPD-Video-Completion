# Fig. 1 完整实验包：KPD-CLSA vs Matrix-SVD vs t-SVD vs Tucker-HOSVD

本工程用于论文 Fig. 1 的最终 representation-efficiency 实验。

## 1. 本版与上一版的关键变化

主图中的 TT-SVD 已替换为 **t-SVD**。

最终比较四类 representation：

1. `KPD-CLSA`
2. `Matrix-SVD`
3. `t-SVD`
4. `Tucker-HOSVD`

不把 CP-ALS 作为第五种 representation 加入 Fig. 1。原因是当前三阶 vector-form finite-sum KPD 与 rank-K CP 具有相同的 separable parameterization；CP-ALS 更适合作为后续 solver ablation。

---

## 2. 数据

Fig. 1 只使用三个 development videos：

- Akiyo
- Foreman
- Mobile

默认下载脚本提供 Xiph QCIF Y4M 文件：

- `data/akiyo_qcif.y4m`
- `data/foreman_qcif.y4m`
- `data/mobile_qcif.y4m`

如果你的论文实验使用的是自己已有的不同分辨率版本，只需要在 `RUN_FIG1_3DEV_PAPER.bat` 中把三个输入路径替换为实际文件路径，算法本身不要求三个视频尺寸相同。

支持 `.y4m` 和三阶 `.npy`。

---

## 3. 参数存储量定义

设视频张量为

`X in R^{H x W x T}`。

### KPD-CLSA

K 个 vector-form separable components：

`P_KPD = K(H + W + T)`.

### Matrix-SVD

将视频 reshape 为 `(HW) x T`：

`P_SVD = r(HW + T)`.

奇异值吸收到一个 factor 中。

### t-SVD

tubal rank 为 r 时：

- `U_r`: `H x r x T`
- `S_r`: r 个 diagonal tubes，每个长度为 T
- `V_r`: `W x r x T`

因此：

`P_tSVD = r*T*(H + W + 1)`.

这里按 real tensor/t-product domain 的 compact factors 计数，不额外利用 FFT 共轭对称性打折。

### Tucker-HOSVD

multilinear rank `(r1,r2,r3)`：

`P_Tucker = H*r1 + W*r2 + T*r3 + r1*r2*r3`.

横轴统一为：

`100 * P_model / (HWT)` (%).

---

## 4. t-SVD 实现

沿时间维做 FFT：

`X_f = fft(X, axis=3)`（代码内部使用等价的 `rfft`）。

对每个 Fourier frontal slice 做矩阵 SVD，并保留前 r 个奇异方向，然后通过 inverse FFT 重构。

代码使用 `numpy.fft.rfft/irfft`，只计算独立频率，提高运行效率；数学上仍是标准 real-valued third-order t-SVD 截断重构。

---

## 5. Fig. 1 平均方式

这是本包里非常重要的修正。

三个 development videos 的尺寸可能不同，因此同一个 rank 对应的 normalized storage ratio 也可能不同。

本工程不会在某些横坐标上平均 1 个视频、另一些位置平均 2 或 3 个视频。对于每种方法：

1. 找到三个视频共同覆盖的 storage-ratio 区间；
2. 在共同的 log-storage grid 上分别插值；
3. 每一个横坐标都固定平均三个 development videos。

因此不会产生由于“参与平均的视频数量变化”造成的假跳变。

---

## 6. 图形风格

使用 Python / Matplotlib 绘图。

- x 轴：log scale
- 折线 + 稀疏 marker
- KPD：圆点
- Matrix-SVD：方块
- t-SVD：三角形
- Tucker-HOSVD：菱形
- `K=4`：单独星号标记

不使用图像生成。

---

## 7. 安装

在工程目录打开终端：

```bash
pip install -r requirements.txt
```

---

## 8. 下载默认的三个视频

双击：

`RUN_DOWNLOAD_3VIDEOS.bat`

或者：

```bash
python download_3videos.py
```

---

## 9. 正式运行 Fig. 1

双击：

`RUN_FIG1_3DEV_PAPER.bat`

或者运行：

```bash
python run_fig1_3dev.py \
  --inputs data/akiyo_qcif.y4m data/foreman_qcif.y4m data/mobile_qcif.y4m \
  --names Akiyo Foreman Mobile \
  --output results_fig1_3dev \
  --max-frames 100 \
  --kpd-max-rank 36 \
  --kpd-sweeps 40 \
  --kpd-restarts 2 \
  --tsvd-max-rank 32 \
  --tucker-max-frac 0.40 \
  --points 16 \
  --max-budget-pct 30 \
  --seed 1
```

---

## 10. 输出

正式结果：

- `results_fig1_3dev/ALL_FIG1_POINTS.csv`
- `results_fig1_3dev/FIG1_AVERAGED_POINTS.csv`

论文图：

- `results_fig1_3dev/figures/fig_kpd_psnr_vs_storage.pdf`
- `results_fig1_3dev/figures/fig_kpd_ssim_vs_storage.pdf`

同时输出 PNG 方便检查。

---

## 11. 只重新画图，不重新做分解

如果后面只想调整字体、legend、marker 等，不要重跑 KPD/t-SVD。

双击：

`RUN_REPLOT_ONLY.bat`

它直接读取：

`results_fig1_3dev/ALL_FIG1_POINTS.csv`

重新生成两张图。

---

## 12. 快速自检

双击：

`RUN_SMOKE_TEST.bat`

它使用三个不同尺寸的随机小 tensor 检查：

- KPD
- Matrix-SVD
- t-SVD
- Tucker
- 跨视频 common-support averaging
- PDF/PNG 输出

这只是代码正确性测试，不是论文数据。

---

## 13. 论文中相应的 storage 公式

Fig. 1 对应段落中，建议把原来的 TT 存储项

`P_TT = Hr + Wr^2 + Tr`

替换为

`P_tSVD = rT(H + W + 1)`.

等真实结果跑完后，再根据真实曲线决定正文是否写“favorable tradeoff”，不要提前预设 KPD 一定优于 t-SVD。
