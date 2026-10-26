# DAVIS 2017 RGB 高分辨率版本 v7

基于 v6 直接扩展；九种算法本身及 frozen hyperparameters **不修改**。RGB 三通道独立恢复再拼接，共用同一个空间 tube-wise mask。随机种子固定 `20261014`，采样率固定 `0.2, 0.3, 0.4`。

## 新硬件第一步（很重要）
RTX 5080 是 Blackwell 架构，不能继续使用旧机器 PyTorch 2.0.1+cu117。推荐新建一个克隆环境而不是覆盖已经跑通灰度实验的 `kpd_cv`：

```bat
conda create -n kpd_5080 --clone kpd_cv
conda activate kpd_5080
python -m pip install --upgrade torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python experiments\11_davis2017_rgb_highres\check_rtx5080.py
```

这仅是 PyTorch GPU 兼容建议，并不保证其他第三方 baseline 可直接兼容最新版依赖；先做完整软件检查。

## 使用（在工程根目录运行）

### 360p / 50帧：建议的首个测试
```bat
python experiments\11_davis2017_rgb_highres\run_highres_all9.py --davis-root "D:\datasets\DAVIS2017" --videos blackswan --size 360p --frames 50 --prepare-only
python experiments\11_davis2017_rgb_highres\run_highres_all9.py --davis-root "D:\datasets\DAVIS2017" --videos blackswan --size 360p --frames 50 --stop-on-error
```

### 直接使用 DAVIS 原始 480p JPEG 像素，不改变空间分辨率，只取前50帧
```bat
python experiments\11_davis2017_rgb_highres\run_highres_all9.py --davis-root "D:\datasets\DAVIS2017" --videos blackswan --size native --frames 50 --stop-on-error
```

### 使用原始空间分辨率和全部视频帧（计算/存储开销可能非常大）
```bat
python experiments\11_davis2017_rgb_highres\run_highres_all9.py --davis-root "D:\datasets\DAVIS2017" --videos blackswan --size native --all-frames --prepare-only
python experiments\11_davis2017_rgb_highres\run_highres_all9.py --davis-root "D:\datasets\DAVIS2017" --videos blackswan --size native --all-frames --stop-on-error
```

### 两个视频顺序执行
```bat
python experiments\11_davis2017_rgb_highres\run_highres_all9.py --davis-root "D:\datasets\DAVIS2017" --videos blackswan,camel --size 360p --frames 50 --jobs 1
```

支持 `--videos all`、`--dry-run`、`--skip-preprocess`、`--no-evaluate`、`--overwrite` 等。全部9种方法执行，已完成输出跳过（除非显式 `--overwrite`）。每个设置使用独立 data/results 路径，不覆盖 v6 与灰度输出。

## 现实限制
- S2NTNN 需要 CUDA；旧版 PyTorch + RTX 5080 不兼容。
- 九种基线中 KBR/TNN/SALTS 需要 MATLAB。
- 480p/all-frames **仅保证数据格式/调度支持**，并不保证九种方法都能在 RTX 5080 16 GB 上完整训练，特别是 S2NTNN 中的 repeated nuclear-norm SVD；不应跳过 360p 单视频可行性测试。
- 改变视频分辨率或帧数后，同一数据集下所有9种方法仍可公平比较，但结果不应与 144p 实验直接按 PSNR 做跨尺寸数值排名。
- 输出速度/内存随分辨率和帧数增加，原始数据是 `JPEGImages/480p` 帧序列，不是直接读取 MP4 文件。

## 图像与统计
新版例子：
```bat
python experiments\10_davis2017_rgb_all9\rgb9.py visualize --output results\davis2017_rgb_all9_seed20261014_360p_f050 --videos blackswan --display-methods "KBR,SALTS,S2NTNN,Ours"
```

## 改动核查
- `prepare_davis_rgb.py`: 支持 `--native-resolution` / `--all-frames`；保持中文路径下 `cv2.imdecode` 读取。
- `benchmark/io_utils.py`: tube-wise mask 不变，只把每一帧重复三角剖分改成一次性三角剖分＋跨帧共享插值权重；小规模随机数据与原 SciPy `griddata` 插值的最大差值低于 `1e-6`。
- 九方法调用及参数列表不变；未在实际 RTX 5080、原始 DAVIS 480p 的全套九方法上运行，不能保证其内存、计算复杂度和框架兼容性。
