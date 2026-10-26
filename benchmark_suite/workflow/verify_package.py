from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
required=[
    "Methods/TNN/lrtc_tnn.m",
    "Methods/KBR/KBR_TC.m",
    "Methods/SALTS/TC_LTNND33.m",
    "Methods/S2NTNN/benchmark_runner_s2ntnn.py",
    "Methods/HLRTF/benchmark_runner_hlrtf.py",
    "Methods/LRTFR/run_same_mask.py",
    "Methods/DRO-TFF/model/drotff.py",
    "Methods/OTLRM/run_same_mask.py",
    "Methods/Ours/run_same_mask.py",
]
missing=[x for x in required if not (ROOT/x).exists()]
if missing:
    print("Missing required paper method files:")
    [print("  -",x) for x in missing]
    raise SystemExit(2)
print("PASS: final-paper method sources are present (8 baselines + Ours).")
