# -*- coding: utf-8 -*-
"""
Replot Fig. 1 from an existing ALL_FIG1_POINTS.csv without rerunning decompositions.
"""
import argparse
from pathlib import Path
from plot_fig1 import main as plot_main

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--output", default="replotted_fig1")
args = ap.parse_args()

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
plot_main(args.csv, out)
print("Saved:", out)
