# -*- coding: utf-8 -*-
"""
Paper-style plotting for Fig. 1.

Important averaging rule:
Each method is averaged only over the storage-ratio interval shared by ALL
three development videos. This prevents artificial jumps caused by changing
the set of videos contributing to the average.

The line is evaluated on a common log-storage grid, while sparse markers are
shown to match the compact manuscript style.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

METHOD_ORDER = [
    "KPD-CLSA",
    "Matrix-SVD",
    "t-SVD",
    "Tucker-HOSVD",
]

MARKERS = {
    "KPD-CLSA": "o",
    "Matrix-SVD": "s",
    "t-SVD": "^",
    "Tucker-HOSVD": "D",
}


def load_rows(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for key in ["budget_pct", "psnr", "ssim"]:
            row[key] = float(row[key])

    return rows


def _interp_method_common_support(rows, method, metric, n_grid=180):
    rr = [r for r in rows if r["method"] == method]
    if not rr:
        return None, None

    by_video = {}
    for r in rr:
        by_video.setdefault(r["video"], []).append(r)

    if len(by_video) != 3:
        raise RuntimeError(
            f"{method}: expected results for 3 development videos, "
            f"found {len(by_video)}."
        )

    mins = []
    maxs = []
    prepared = []

    for video, vv in by_video.items():
        vv = sorted(vv, key=lambda z: z["budget_pct"])
        x = np.array([z["budget_pct"] for z in vv], dtype=float)
        y = np.array([z[metric] for z in vv], dtype=float)

        # Remove any duplicated storage x values before interpolation.
        xu, unique_idx = np.unique(x, return_index=True)
        yu = y[unique_idx]

        mins.append(xu.min())
        maxs.append(xu.max())
        prepared.append((xu, yu))

    lo = max(mins)
    hi = min(maxs)

    if not (hi > lo):
        return None, None

    grid = np.geomspace(lo, hi, n_grid)

    curves = []
    for x, y in prepared:
        curves.append(
            np.interp(
                np.log10(grid),
                np.log10(x),
                y,
            )
        )

    return grid, np.mean(np.vstack(curves), axis=0)


def _k4_point(rows, metric):
    pts = [
        r for r in rows
        if r["method"] == "KPD-CLSA" and r["rank_spec"] == "K=4"
    ]
    if len(pts) != 3:
        raise RuntimeError(
            f"K=4 marker requires 3 development-video points; found {len(pts)}."
        )

    return (
        float(np.mean([r["budget_pct"] for r in pts])),
        float(np.mean([r[metric] for r in pts])),
    )


def make_fig(rows, outpath, metric):
    fig, ax = plt.subplots(figsize=(4.55, 3.45))

    for method in METHOD_ORDER:
        grid, y = _interp_method_common_support(
            rows,
            method,
            metric,
            n_grid=180,
        )
        if grid is None:
            continue

        # Show about seven visible markers, while retaining a smooth connecting line.
        markevery = max(1, len(grid) // 6)

        ax.plot(
            grid,
            y,
            linewidth=1.6,
            marker=MARKERS[method],
            markersize=4.5,
            markevery=markevery,
            label=method,
        )

    xstar, ystar = _k4_point(rows, metric)
    ax.scatter(
        [xstar],
        [ystar],
        marker="*",
        s=90,
        zorder=6,
        label="K = 4 backbone",
    )
    ax.annotate(
        r"$K=4$ backbone",
        xy=(xstar, ystar),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=7.5,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Stored parameters / tensor entries (%)")
    if metric == "psnr":
        ax.set_ylabel("Average approximation PSNR (dB)")
    else:
        ax.set_ylabel("Average approximation SSIM")

    ax.grid(True, alpha=0.22)
    ax.legend(
        fontsize=7.2,
        loc="best",
        frameon=False,
        handlelength=2.0,
    )

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(
        Path(outpath).with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_average_table(rows, outcsv):
    out = []

    for method in METHOD_ORDER:
        psnr_x, psnr_y = _interp_method_common_support(
            rows, method, "psnr", n_grid=50
        )
        ssim_x, ssim_y = _interp_method_common_support(
            rows, method, "ssim", n_grid=50
        )

        if psnr_x is None or ssim_x is None:
            continue

        # Same common support/grid construction is used for both metrics.
        for xval, pval, sval in zip(psnr_x, psnr_y, ssim_y):
            out.append(
                {
                    "method": method,
                    "budget_pct": float(xval),
                    "mean_psnr": float(pval),
                    "mean_ssim": float(sval),
                    "n_videos": 3,
                }
            )

    with open(outcsv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "budget_pct",
                "mean_psnr",
                "mean_ssim",
                "n_videos",
            ],
        )
        w.writeheader()
        w.writerows(out)


def main(csv_path, fig_dir):
    rows = load_rows(csv_path)

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    make_fig(
        rows,
        fig_dir / "fig_kpd_psnr_vs_storage.pdf",
        "psnr",
    )
    make_fig(
        rows,
        fig_dir / "fig_kpd_ssim_vs_storage.pdf",
        "ssim",
    )

    write_average_table(
        rows,
        fig_dir.parent / "FIG1_AVERAGED_POINTS.csv",
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fig-dir", required=True)
    args = ap.parse_args()

    main(args.csv, args.fig_dir)
