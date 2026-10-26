# V7 theory-aligned update

本目录已同步论文数学修改：目标函数单调性、正则最优尺度平衡、cross-fitting no-leakage 软件接口、bounded KPD backbone、final cross-fit protocol。详细见 `V7_MODIFICATION_NOTES_CN.md`。

**首次运行请优先双击 `RUN_SMALL_AKIYO_VALIDATION.bat`，不要直接启动 23-video 全实验。**

---

# V6 — DELTA Table IV 23-video tube-wise benchmark

## 先说明两个重要事实

DELTA 正式发表版 Table IV 的 video tube-wise 结果是：

| Method | SR=.1 | SR=.2 | SR=.3 |
|---|---:|---:|---:|
| DELTA PSNR | 22.78 | 25.23 | 27.21 |
| DELTA SSIM | .636 | .761 | .834 |

Table IV 明确写明：23 个 ASU Video Trace Library 视频，其中
17 个为 `144×176×100`，6 个为 `288×352×50`。

论文正文没有逐一打印这 23 个文件名，也没有说明随机 mask seed。
因此本工程的最终比较应称为 **protocol-aligned comparison**，
除非再从作者官方 demo 中确认完全相同的 manifest 和 seed。

## 当前 working manifest

为了先把 23-video benchmark 跑起来，本工程采用以下 23 个经典 ASU
序列，并满足 Table IV 的 17/6 尺寸组成。

17 个 `144×176×100`：
Akiyo, Bridge Close, Bridge Far, Carphone, Claire, Coastguard, Container,
Foreman, Grandma, Hall Monitor, Highway, Miss America, Mother Daughter,
News, Salesman, Silent, Suzie.

6 个 `288×352×50`：
Bus, Flower, Mobile, Paris, Tempete, Waterfall.

这里默认排除 Stefan（该 ASU 序列只有 90 帧）。投稿前应使用作者官方
DELTA demo 再复核一次 manifest；工程已经提供对应下载和扫描脚本。

## 第一步：下载 23 个视频

直接双击：

```text
DOWNLOAD_DELTA23_XIPH.bat
```

程序从 Xiph Derf Test Media Collection 下载 Y4M。为了避免下载完整的
超长视频，下载器会边下载边解析 Y4M，只保存实验需要的：

- low group：前 100 帧；
- high group：前 50 帧。

所以本地数据量远小于把所有原始长视频完整下载下来。

已有合法文件会自动跳过；Akiyo / Carphone / Foreman 不需要重新下载。

下载结束后双击：

```text
VERIFY_DATASET.bat
```

应看到：

```text
DATASET CHECK PASSED: 23 videos = 17 low + 6 high.
```

所有直接 URL 也保存在：

```text
VIDEO_DOWNLOAD_URLS.txt
```

## 第二步：建议先跑 SR=.1

GTX 1650 上 69 个完整 V5.1 case 会比较耗时，因此先运行：

```text
RUN_V6_TABLEIV_SR01_FIRST.bat
```

这是 23 个视频 × 1 个 SR × 1 个 seed。

全部完成后，会第一次得到真正的 23-video average，并自动生成：

```text
v6_tableiv_23video_seed1\
    PER_VIDEO_RESULTS_PARTIAL.csv
    AGGREGATE_BY_SR.csv
    TABLE_IV_COMPARISON.csv
    FIG_TABLE_IV_PSNR.png
    SUMMARY.json
```

如果 SR=.1 有竞争力，再运行：

```text
RUN_V6_TABLEIV_SR02.bat
RUN_V6_TABLEIV_SR03.bat
```

或者直接：

```text
RUN_V6_TABLEIV_ALL_3SR.bat
```

已经完成的 case 会自动 `[RESUME]` 跳过。

## 方法参数

方法固定为当前成熟 V5.1，不再针对每个视频调参：

- KPD rank K = 12
- Masked-CLSA sweeps = 12
- 5-fold cross-fitting
- CNN hidden channels = 12
- CNN steps = 1200
- seed = 1
- tube-wise exact sampling
- native video dimensions
- observed entries are clamped for final data consistency

## 投稿前的严格 protocol 复核

作者官方 PyTorch demo 是分卷压缩包。可以双击：

```text
DOWNLOAD_OFFICIAL_DELTA_DEMO.bat
```

得到：

```text
official_DELTA_demo\
    DELTA.zip
    DELTA.z01
```

用 7-Zip 从 `DELTA.zip` 解压，然后运行：

```text
INSPECT_OFFICIAL_DELTA_DEMO.bat
```

它会扫描官方 demo 中可能涉及：

- video names
- 144/176/288/352 dimensions
- seed / manual_seed
- mask / tube
- sampling rate

并生成：

```text
official_DELTA_demo\PROTOCOL_SCAN_REPORT.txt
```

如果官方代码与 working manifest 有任何差异，应在最终正式实验前修改
manifest，而不是把当前结果称为 identical-mask reproduction。

## 正确的 Table IV reported rows

都已经放在：

```text
DELTA_TABLE_IV_REPORTED.csv
```

包含 HaLRTC, TNN, TLRR, HLRTF, S2NTNN, CoNoT, TWTC, TCTV, TNN-G,
KBR, SALTS, LRTFR 和 DELTA。

其中最重要的比较目标是：

```text
             SR=.1       SR=.2       SR=.3
DELTA        22.78       25.23       27.21
TNN-G        22.60       24.33       25.96
TLRR         22.29       24.10       25.43
KBR          17.51       20.54       23.94
SALTS        19.64       23.58       26.24
```

完成 23 个视频之前，我们的方法在 `TABLE_IV_COMPARISON.csv` 中显示 NA，
避免误把 partial average 当作正式 23-video average。

## 建议的实际执行顺序

1. `DOWNLOAD_DELTA23_XIPH.bat`
2. `VERIFY_DATASET.bat`
3. `RUN_V6_TABLEIV_SR01_FIRST.bat`
4. 把 SR=.1 最后的 V6 AGGREGATE 发给我分析
5. 若结果有竞争力，再跑 SR=.2 / .3
6. 投稿前用官方 DELTA demo 复核 manifest / mask seed
