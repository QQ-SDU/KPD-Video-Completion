# Dataset preparation

Data are not bundled with this repository. Obtain DAVIS 2017 from the dataset's
official distribution and confirm usage rights; place extracted frame sequences
under `data/DAVIS2017/JPEGImages/480p/<sequence>/*.jpg` or point `--davis-root`
to the relevant DAVIS root. The supplied v15 workflow searches nested `DAVIS/`
and `2017/` variants too.

The v15 frozen protocol in `FINAL_FROZEN_CONFIG.json` describes RGB input of
shape 360 x 640 x 50 x 3, the first 50 consecutive frames per sequence, and a
common spatial tube mask shared across frames and RGB channels. Sampling rates
are 0.2, 0.3, and 0.4; fixed mask seed is 20261014.

The 23-video grayscale benchmark and structured-mask RGB experiments follow
separate protocols in `docs/DATASET_AND_METRICS.md`. Their source datasets are
also not included; do not assume the DAVIS RGB setup covers these experiments.

Do not commit third-party video sequences, licensed datasets, or full reconstruction
results to the main Git repository. Use appropriately licensed external hosting
for supplementary visualizations and attribute datasets.
