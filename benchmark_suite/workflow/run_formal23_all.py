from __future__ import annotations
from pathlib import Path
import argparse, os, secrets, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]

def run(cmd, title):
    print("\n" + "="*88)
    print(title)
    print("="*88)
    print(">", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional reproducibility seed; omitted => generate a fresh seed.")
    args = ap.parse_args()
    seed = int(args.seed) if args.seed is not None else secrets.randbits(32)
    os.environ["KPD_EXPERIMENT_SEED"] = str(seed)
    print(f"Run seed: {seed}")

    print("\n" + "#"*88)
    print("FORMAL 23-VIDEO BENCHMARK")
    print("8 baselines + Ours | SR=.10/.20/.30/.40/.50")
    print("MATLAB lane and Python lane run in parallel; each lane is internally serial.")
    print("Existing valid reconstruction files are skipped automatically.")
    print("#"*88)

    run([sys.executable, str(ROOT/"workflow/verify_package.py")],
        "1/6 Environment and bundled-source check")

    run([sys.executable, str(ROOT/"workflow/formal23_data_preflight.py")],
        "2/6 Verify the 23 source videos and auto-generate manifest.csv")

    run([sys.executable, str(ROOT/"workflow/run_all_pipeline.py"), "--dataset", "formal23"],
        "3/5 Build 115 cases, run all 9 methods, evaluate, and build standard reports")

    # Paper summaries are useful, but held-out20 detection is nonfatal.
    subprocess.run([sys.executable, str(ROOT/"workflow/build_paper_summaries.py")], cwd=str(ROOT))

    run([sys.executable, str(ROOT/"workflow/formal23_final_audit.py")],
        "4/5 Build All-23/Held-out-20 summaries and run final completeness audit")

    print("\n5/5 Formal benchmark finalized.")

    print("\n" + "#"*88)
    print("FORMAL EXPERIMENT COMPLETE")
    print("All reconstruction files are retained under artifacts/results/formal23/<profile>/")
    print("Do not delete artifacts/cases or artifacts/results.")
    print("#"*88)

if __name__ == "__main__":
    main()
