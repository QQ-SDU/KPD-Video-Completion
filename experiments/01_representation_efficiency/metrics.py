# -*- coding: utf-8 -*-
import csv
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity

EPS = 1e-12

def mse(x, xhat):
    d = np.asarray(x, dtype=np.float64) - np.asarray(xhat, dtype=np.float64)
    return float(np.mean(d*d))

def psnr(x, xhat, data_range=1.0):
    m = mse(x, xhat)
    if m <= 1e-16:
        return 99.0
    return float(10.0*np.log10((data_range*data_range)/m))

def mean_frame_ssim(x, xhat, data_range=1.0):
    x = np.asarray(x, dtype=np.float64)
    xhat = np.asarray(xhat, dtype=np.float64)
    vals = []
    for t in range(x.shape[2]):
        vals.append(structural_similarity(x[:,:,t], xhat[:,:,t], data_range=data_range))
    return float(np.mean(vals))

def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def read_csv(path):
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
