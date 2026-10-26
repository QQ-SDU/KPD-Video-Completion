# -*- coding: utf-8 -*-
"""Small correctness smoke test; does not reproduce paper numbers."""
from pathlib import Path
import numpy as np

from metrics import write_csv
from representation_methods import curve_kpd, curve_svd, curve_tsvd, curve_tucker
from plot_fig1 import main as plot_main

rng = np.random.default_rng(7)
out = Path("results_smoke")
out.mkdir(parents=True, exist_ok=True)

all_rows = []
for idx, name in enumerate(["DevA", "DevB", "DevC"]):
    # Deliberately use different tensor sizes to test the common-support averaging.
    shape = [(20, 24, 12), (22, 26, 10), (18, 28, 14)][idx]
    x = rng.random(shape)

    rows = []
    rows += curve_kpd(x, name, max_rank=5, points=5, sweeps=4, restarts=1, seed=1)
    rows += curve_svd(x, name, points=5)
    rows += curve_tsvd(x, name, max_rank=5, points=5)
    rows += curve_tucker(x, name, max_frac=0.35, points=5)

    all_rows += rows

write_csv(all_rows, out / "ALL_FIG1_POINTS.csv")
plot_main(out / "ALL_FIG1_POINTS.csv", out / "figures")

print("Smoke test passed.")
