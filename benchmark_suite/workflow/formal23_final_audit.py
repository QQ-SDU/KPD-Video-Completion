from __future__ import annotations
from pathlib import Path
import csv, json, sys, math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow"))
from common import case_dir, recon_dir, metrics_dir, result_root, load_experiment, sync_profile, profile_id

DATASET = "formal23"
EXPECTED_VIDEOS = 23
EXPECTED_CASES = 115
EXPECTED_METHODS = 9
EXPECTED_RECONS = 1035

def main():
    sync_profile()
    exp = load_experiment()
    methods = exp["methods"]["all"]

    case_npz = sorted(case_dir(DATASET).glob("*.npz"))
    case_mat = sorted(case_dir(DATASET).glob("*.mat"))

    rows = []
    total_recons = 0
    for method in methods:
        files = sorted(recon_dir(DATASET, method).glob("*.mat"))
        total_recons += len(files)
        rows.append({
            "method": method,
            "expected": EXPECTED_CASES,
            "found": len(files),
            "complete": len(files) == EXPECTED_CASES,
        })

    metrics_file = metrics_dir(DATASET) / "per_run_metrics.csv"
    missing_file = metrics_dir(DATASET) / "missing_or_failed_outputs.csv"

    metric_rows = 0
    runtime_missing = 0
    finite_metric_failures = 0
    if metrics_file.exists() and metrics_file.stat().st_size > 0:
        df = pd.read_csv(metrics_file)
        metric_rows = len(df)
        for col in ["psnr_full","ssim_frame_mean"]:
            if col in df.columns:
                finite_metric_failures += int((~pd.to_numeric(df[col], errors="coerce").map(math.isfinite)).sum())
        if "runtime_seconds" in df.columns:
            runtime_missing = int(pd.to_numeric(df["runtime_seconds"], errors="coerce").isna().sum())

    missing_rows = 0
    if missing_file.exists() and missing_file.stat().st_size > 3:
        try:
            mdf = pd.read_csv(missing_file)
            missing_rows = len(mdf)
        except Exception:
            missing_rows = 1

    checks = {
        "videos_expected": EXPECTED_VIDEOS,
        "cases_npz_found": len(case_npz),
        "cases_mat_found": len(case_mat),
        "cases_expected": EXPECTED_CASES,
        "methods_expected": EXPECTED_METHODS,
        "reconstructions_found": total_recons,
        "reconstructions_expected": EXPECTED_RECONS,
        "per_run_metric_rows": metric_rows,
        "missing_or_failed_rows": missing_rows,
        "runtime_missing_rows": runtime_missing,
        "nonfinite_primary_metric_values": finite_metric_failures,
    }

    passed = (
        len(case_npz) == EXPECTED_CASES and
        len(case_mat) == EXPECTED_CASES and
        len(methods) == EXPECTED_METHODS and
        total_recons == EXPECTED_RECONS and
        metric_rows == EXPECTED_RECONS and
        missing_rows == 0 and
        finite_metric_failures == 0
    )

    outdir = result_root(DATASET)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(outdir / "FINAL_METHOD_COMPLETENESS.csv", index=False, encoding="utf-8-sig")
    (outdir / "FINAL_AUDIT.json").write_text(
        json.dumps({"profile_id": profile_id(), "pass": passed, **checks}, indent=2),
        encoding="utf-8"
    )

    lines = [
        "FORMAL23 FINAL AUDIT",
        "=" * 72,
        f"Profile: {profile_id()}",
        f"Cases NPZ: {len(case_npz)}/{EXPECTED_CASES}",
        f"Cases MAT: {len(case_mat)}/{EXPECTED_CASES}",
        f"Reconstructions: {total_recons}/{EXPECTED_RECONS}",
        f"Metric rows: {metric_rows}/{EXPECTED_RECONS}",
        f"Missing/failed rows: {missing_rows}",
        f"Missing runtime rows: {runtime_missing}",
        f"Non-finite PSNR/SSIM values: {finite_metric_failures}",
        "",
    ]
    for r in rows:
        lines.append(f"{r['method']:8s}: {r['found']:3d}/{EXPECTED_CASES}  {'PASS' if r['complete'] else 'INCOMPLETE'}")
    lines += ["", "FINAL STATUS: " + ("PASS" if passed else "CHECK REQUIRED")]
    text = "\n".join(lines) + "\n"
    (outdir / "FINAL_AUDIT.txt").write_text(text, encoding="utf-8")
    print(text)

    if not passed:
        raise SystemExit(4)

if __name__ == "__main__":
    main()
