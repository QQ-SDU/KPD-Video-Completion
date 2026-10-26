from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Benchmark"))
sys.path.insert(0, str(ROOT / "workflow"))

from benchmark.io_utils import (
    load_video, resize_hwt, make_tube_mask, linear_spatial_init, save_case
)
from common import (
    load_experiment, enabled_datasets, get_dataset, supported_files,
    load_optional_manifest, case_dir, safe_name, sr_tag, sync_profile
)
from case_compat import (
    scientific_core, scientific_signature, case_files_present,
    case_content_matches, refreshed_meta
)

def _int_or_none(x):
    if x is None or x == "" or str(x).lower() == "none":
        return None
    return int(x)

def _effective(d, row, key, default=None):
    return row[key] if row and row.get(key, "") != "" else d.get(key, default)

def prepare_dataset(name, force=False):
    exp = load_experiment()
    dcfg = get_dataset(name)
    root = ROOT / dcfg["root"]
    root.mkdir(parents=True, exist_ok=True)
    videos = supported_files(dcfg)
    if not videos:
        raise RuntimeError(f"No supported video files in {root}")

    manifest = load_optional_manifest(dcfg)
    outdir = case_dir(name)
    outdir.mkdir(parents=True, exist_ok=True)

    enabled = []
    for src in videos:
        row = manifest.get(src.name, {})
        if str(row.get("enabled", "1")).strip().lower() not in {"0", "false", "no"}:
            enabled.append((src, row))

    total = len(enabled) * len(exp["sampling"]["sampling_rates"]) * len(exp["sampling"]["seeds"])
    job = 0

    for src, row in enabled:
        layout = str(_effective(dcfg, row, "layout", "HWT"))
        key = _effective(dcfg, row, "input_key", None) or None
        max_frames = _int_or_none(_effective(dcfg, row, "max_frames", None))
        rh = _int_or_none(_effective(dcfg, row, "resize_height", None))
        rw = _int_or_none(_effective(dcfg, row, "resize_width", None))
        video_name = safe_name(Path(row.get("name") or src.stem).stem)

        print(
            f"[LOAD] {src.name} layout={layout} max_frames={max_frames} resize={rh}x{rw}",
            flush=True,
        )
        gt = resize_hwt(
            load_video(src, key=key, layout=layout, max_frames=max_frames), rh, rw
        )
        h, w, t = gt.shape
        stat = src.stat()

        for sr in exp["sampling"]["sampling_rates"]:
            for seed in exp["sampling"]["seeds"]:
                job += 1
                base = outdir / f"{video_name}__sr{sr_tag(sr)}__seed{int(seed)}"
                meta_path = Path(str(base) + ".json")

                core = scientific_core(
                    dataset=name,
                    video_name=video_name,
                    source_path=src,
                    source_size=int(stat.st_size),
                    shape_hwt=(h, w, t),
                    sampling_rate=float(sr),
                    seed=int(seed),
                    layout=layout,
                    input_key=key,
                    max_frames=max_frames,
                    resize_height=rh,
                    resize_width=rw,
                    sampling_type=exp["sampling"]["type"],
                )

                # V2.1 reuse rule:
                # Existing case is reused when its ACTUAL GT + deterministic mask
                # match the current experiment. Absolute path and source mtime do
                # not define scientific identity.
                if not force and case_files_present(base):
                    ok, reason = case_content_matches(base, gt, sr, seed)
                    if ok:
                        old_meta = {}
                        if meta_path.exists():
                            try:
                                old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            except Exception:
                                old_meta = {}
                        new_meta = refreshed_meta(
                            old_meta, core, src, int(stat.st_mtime_ns)
                        )
                        # Refresh JSON provenance only; do NOT touch MAT/NPZ mtimes.
                        meta_path.write_text(
                            json.dumps(new_meta, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        print(
                            f"[{job}/{total}] SKIP case {base.name} "
                            f"(V2.1 content-verified reuse)",
                            flush=True,
                        )
                        continue
                    print(
                        f"[{job}/{total}] REBUILD {base.name} "
                        f"(existing case incompatible: {reason})",
                        flush=True,
                    )
                else:
                    print(f"[{job}/{total}] BUILD {base.name}", flush=True)

                mask = make_tube_mask(h, w, t, float(sr), int(seed))
                obs = gt * mask.astype(np.float32)
                init = linear_spatial_init(obs, mask)

                meta = dict(core)
                meta.update(
                    {
                        "source_file": str(src.resolve()),
                        "source_mtime_ns": int(stat.st_mtime_ns),
                        "source_name": src.name,
                        "scientific_signature_sha256": scientific_signature(core),
                        "case_signature_sha256": scientific_signature(core),
                        "sampling_rate_realized": float(mask[:, :, 0].mean()),
                        "observed_spatial_tubes": int(mask[:, :, 0].sum()),
                        "reuse_compatibility_version": "V2.1",
                    }
                )
                save_case(base, gt, obs, mask, init, meta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sync_profile()
    names = a.dataset or [d["name"] for d in enabled_datasets()]
    for n in names:
        prepare_dataset(n, a.force)

if __name__ == "__main__":
    main()
