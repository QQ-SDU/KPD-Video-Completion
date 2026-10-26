# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pathlib import Path

from io_video import load_input
from metrics import write_csv
from representation_methods import (
    curve_kpd,
    curve_svd,
    curve_tsvd,
    curve_tucker,
)
from plot_fig1 import main as plot_main


def parse_args():
    ap = argparse.ArgumentParser(
        description="Reproduce manuscript Fig. 1 on exactly three development videos."
    )
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--output", default="results_fig1_3dev")
    ap.add_argument("--max-frames", type=int, default=100)

    ap.add_argument("--kpd-max-rank", type=int, default=36)
    ap.add_argument("--kpd-sweeps", type=int, default=40)
    ap.add_argument("--kpd-restarts", type=int, default=2)

    ap.add_argument("--tsvd-max-rank", type=int, default=32)
    ap.add_argument("--tucker-max-frac", type=float, default=0.40)

    ap.add_argument("--points", type=int, default=16)
    ap.add_argument("--max-budget-pct", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=None, help="Optional reproducibility seed; omitted => generate a fresh seed.")

    return ap.parse_args()


def main():
    args = parse_args()

    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    print(f"Run seed: {args.seed}")

    if len(args.inputs) != 3 or len(args.names) != 3:
        raise ValueError(
            "Fig. 1 must use exactly the three development videos."
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for path, name in zip(args.inputs, args.names):
        print("=" * 78)
        print("DEVELOPMENT VIDEO:", name)

        X = load_input(
            path,
            max_frames=args.max_frames,
            resize_hw=None,
        )
        print("shape =", X.shape)

        rows = []
        rows += curve_kpd(
            X,
            name,
            max_rank=args.kpd_max_rank,
            points=args.points,
            sweeps=args.kpd_sweeps,
            restarts=args.kpd_restarts,
            seed=args.seed,
        )
        rows += curve_svd(
            X,
            name,
            points=args.points,
        )
        rows += curve_tsvd(
            X,
            name,
            max_rank=args.tsvd_max_rank,
            points=args.points,
        )
        rows += curve_tucker(
            X,
            name,
            max_frac=args.tucker_max_frac,
            points=args.points,
        )

        # Keep a common manuscript-facing budget window.
        rows = [
            r for r in rows
            if r["budget_pct"] <= args.max_budget_pct + 1e-12
        ]

        write_csv(rows, out / f"{name}_FIG1_POINTS.csv")
        all_rows += rows

    write_csv(all_rows, out / "ALL_FIG1_POINTS.csv")

    plot_main(
        out / "ALL_FIG1_POINTS.csv",
        out / "figures",
    )

    print("=" * 78)
    print("DONE")
    print("Main manuscript outputs:")
    print(out / "figures" / "fig_kpd_psnr_vs_storage.pdf")
    print(out / "figures" / "fig_kpd_ssim_vs_storage.pdf")
    print("Numerical outputs:")
    print(out / "ALL_FIG1_POINTS.csv")
    print(out / "FIG1_AVERAGED_POINTS.csv")


if __name__ == "__main__":
    main()
