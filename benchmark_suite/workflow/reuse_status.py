from __future__ import annotations
from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow"))

from common import (
    load_experiment, get_dataset, count_enabled_sources, case_dir,
    recon_dir, sync_profile, profile_id
)
from output_validation import validate_output

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="formal23")
    a = ap.parse_args()
    sync_profile()

    exp = load_experiment()
    dcfg = get_dataset(a.dataset)
    nsrc = count_enabled_sources(dcfg)
    expected_cases = nsrc * len(exp["sampling"]["sampling_rates"]) * len(exp["sampling"]["seeds"])
    cases = sorted(case_dir(a.dataset).glob("*.mat"))

    print("\nREUSE STATUS")
    print("=" * 76)
    print("Dataset:", a.dataset)
    print("Profile:", profile_id())
    print(f"Cases currently present: {len(cases)}/{expected_cases}")
    print("")
    total_valid = 0
    for method in exp["methods"]["all"]:
        valid = 0
        old_timestamp = 0
        invalid = 0
        for c in cases:
            out = recon_dir(a.dataset, method) / f"{method}__{c.stem}.mat"
            if not out.exists():
                continue
            ok, reason = validate_output(c, out)
            if ok:
                valid += 1
                if reason == "ok_reuse_older_than_case":
                    old_timestamp += 1
            else:
                invalid += 1
        total_valid += valid
        print(
            f"{method:8s}: valid/reusable={valid:3d} | "
            f"older-timestamp-but-valid={old_timestamp:3d} | invalid={invalid:3d}"
        )
    print("")
    print("Total valid reusable reconstructions:", total_valid)
    print("Important: 'older-timestamp-but-valid' is REUSED in V2.1 and will NOT be quarantined.")

if __name__ == "__main__":
    main()
