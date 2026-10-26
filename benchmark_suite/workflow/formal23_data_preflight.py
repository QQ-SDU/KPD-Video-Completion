from __future__ import annotations
from pathlib import Path
import csv, json, sys, re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Benchmark"))
from benchmark.io_utils import load_video
sys.path.insert(0, str(ROOT / "workflow"))
from common import get_dataset, supported_files, safe_name

DATASET = "formal23"
EXPECTED_TOTAL = 23
EXPECTED_SMALL = 17
EXPECTED_LARGE = 6
SMALL_HW = (144, 176)
LARGE_HW = (288, 352)
SMALL_FRAMES = 100
LARGE_FRAMES = 50

def infer_rule(src: Path, layout: str, key):
    # Read only as many frames as needed to validate the formal protocol.
    # Y4M reader stops early at max_frames; array formats are loaded once.
    x = load_video(src, key=key, layout=layout, max_frames=SMALL_FRAMES)
    h, w, t = map(int, x.shape)

    if (h, w) == SMALL_HW:
        required = SMALL_FRAMES
        size_class = "small"
    elif (h, w) == LARGE_HW:
        required = LARGE_FRAMES
        size_class = "large"
    else:
        raise RuntimeError(
            f"{src.name}: unsupported spatial size {h}x{w}. "
            f"Formal protocol expects 144x176 or 288x352."
        )

    # Verify enough frames. For large videos load_video above already allows up to 100;
    # for small videos we require exactly at least 100.
    if t < required:
        raise RuntimeError(
            f"{src.name}: only {t} frame(s), but formal protocol requires at least {required}."
        )
    return h, w, required, size_class

def main():
    dcfg = get_dataset(DATASET)
    root = ROOT / dcfg["root"]
    root.mkdir(parents=True, exist_ok=True)

    videos = supported_files(dcfg)
    if len(videos) != EXPECTED_TOTAL:
        print("\n[FORMAL23 DATA ERROR]")
        print(f"Expected exactly {EXPECTED_TOTAL} source videos in: {root}")
        print(f"Found: {len(videos)}")
        for p in videos:
            print("  ", p.name)
        raise SystemExit(2)

    rows = []
    counts = {"small": 0, "large": 0}
    names_seen = set()

    for i, src in enumerate(videos, 1):
        layout = str(dcfg.get("layout", "HWT"))
        key = dcfg.get("input_key", None)
        h, w, required, size_class = infer_rule(src, layout, key)
        counts[size_class] += 1

        name = safe_name(src.stem)
        if name.lower() in names_seen:
            raise RuntimeError(f"Duplicate normalized video name: {name}")
        names_seen.add(name.lower())

        rows.append({
            "file": src.name,
            "enabled": 1,
            "name": name,
            "layout": layout,
            "input_key": "" if key is None else key,
            "max_frames": required,
            "resize_height": "",
            "resize_width": "",
        })
        print(f"[{i:02d}/{EXPECTED_TOTAL}] {src.name}: {h}x{w}, use first {required} frames")

    if counts["small"] != EXPECTED_SMALL or counts["large"] != EXPECTED_LARGE:
        raise SystemExit(
            f"[FORMAL23 DATA ERROR] Expected {EXPECTED_SMALL} small + {EXPECTED_LARGE} large videos, "
            f"but found {counts['small']} small + {counts['large']} large."
        )

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["file","enabled","name","layout","input_key","max_frames","resize_height","resize_width"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    report = {
        "dataset": DATASET,
        "root": str(root),
        "videos": EXPECTED_TOTAL,
        "small_144x176_first100": counts["small"],
        "large_288x352_first50": counts["large"],
        "sampling_rates": [0.10, 0.20, 0.30, 0.40, 0.50],
        "seed": 1,
        "expected_cases": 115,
        "expected_methods": 9,
        "expected_reconstructions": 1035,
        "manifest": str(manifest),
    }
    report_path = ROOT / "artifacts" / "formal23_data_preflight.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[PASS] Formal23 source data verified.")
    print("       17 x 144x176 -> first 100 frames")
    print("        6 x 288x352 -> first 50 frames")
    print("       manifest.csv generated automatically.")
    print("       Expected cases: 115")
    print("       Expected reconstructions: 1035")

if __name__ == "__main__":
    main()
