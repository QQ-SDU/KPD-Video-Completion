# 彩色视频规则性缺失对比实验（Bus + Container + Coastguard，9 方法）

这个扩展包在原有工程基础上新增了一套**可直接运行**的彩色视频 structured-missing 对比实验代码，用于：

- 视频：**Bus / Container / Coastguard**
- mask：**grid / text / block** 三种规则性空间缺失
- 方法：DRO-TFF, HLRTF, KBR, LRTFR, OTLRM, TNN, SALTS, S$^2$NTNN, Ours

## 本版 mask 稀疏度（建议用于论文图）
- **Grid**：宽松稀疏大格子，缺失率约 **6%–7%**
- **Text**：`KPD / VIDEO` 两行文字遮挡，缺失率约 **6%**
- **Block**：单个局部方块遮挡，缺失率约 **6%**

这版更适合做你想要的**局部可辨识细节恢复**比较，而不是极端大面积 completion。

## 一、你要做的事情
1. 把三个彩色视频放到 `data_color_structured9/inputs/`
2. 如文件名不同，修改 `data_color_structured9/manifest.csv`
3. 先运行 `CHECK_COLOR_STRUCTURED_DATA.bat`
4. 再运行 `RUN_COLOR_STRUCTURED_ALL.bat`

## 二、保存哪些数据（方便后续绘图和计算）
这版会**额外保存图像数据和原始数组**，方便后续绘图与重新计算：

### 1）权威原始数组（推荐后续计算直接用）
- `artifacts_color_structured9/cases_rgb/*.npz`：保存 `gt_rgb / observed_rgb / mask2d`
- `artifacts_color_structured9/results_rgb/<method>/*.npz`：保存 `recon_rgb`
- `artifacts_color_structured9/cases_channels/*.npz` 与 `.mat`：每个通道的 case
- `artifacts_color_structured9/metrics/*.json`、`summary_all.csv`

### 2）直接可视化的图像资产（方便后续拼图）
- `artifacts_color_structured9/image_assets/<video>/<mask>/`
  - `mask_observed_white.png`
  - `mask_missing_white.png`
  - `gt_frame_XXX.png`
  - `observed_frame_XXX.png`
- `artifacts_color_structured9/case_previews/<video>/<mask>/gt_frames/*.png`
- `artifacts_color_structured9/case_previews/<video>/<mask>/observed_frames/*.png`

### 3）论文成图
- `artifacts_color_structured9/figures/<video>/...png/.pdf`

## 三、说明
为了尽量不改动原始三阶 completion 方法，这套彩色视频实验采用 **RGB 三通道分别恢复，再合成为 RGB** 的方式。三通道使用**完全相同的 2-D 结构化 mask**。

## 四、推荐论文中的使用方式
建议把这部分作为：
> **Structured Missing Patterns on Color Videos**

用于展示：在规则性局部遮挡下，方法对**可辨识细节**的恢复能力。
