from __future__ import annotations
from pathlib import Path
import hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Benchmark"))
from benchmark.evaluate_output import load_case
from benchmark.io_utils import make_tube_mask

SCIENCE_KEYS = (
    "dataset", "video_name", "source_name", "source_size",
    "shape_hwt", "sampling_rate", "seed", "layout", "input_key",
    "max_frames", "resize_height", "resize_width", "sampling_type",
)

def _norm(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, tuple):
        return list(v)
    return v

def scientific_core(*, dataset, video_name, source_path, source_size, shape_hwt,
                    sampling_rate, seed, layout, input_key, max_frames,
                    resize_height, resize_width, sampling_type):
    p = Path(source_path)
    return {
        "dataset": str(dataset),
        "video_name": str(video_name),
        "source_name": p.name,
        "source_size": int(source_size),
        "shape_hwt": [int(x) for x in shape_hwt],
        "sampling_rate": float(sampling_rate),
        "seed": int(seed),
        "layout": str(layout),
        "input_key": None if input_key in (None, "") else str(input_key),
        "max_frames": None if max_frames is None else int(max_frames),
        "resize_height": None if resize_height is None else int(resize_height),
        "resize_width": None if resize_width is None else int(resize_width),
        "sampling_type": str(sampling_type),
    }

def scientific_signature(core):
    payload = {k: _norm(core.get(k)) for k in SCIENCE_KEYS}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def legacy_meta_scientific_core(meta):
    # V1.2/V2.0 metadata used absolute source_file and source_mtime_ns.
    # Those are intentionally excluded from scientific identity.
    source_name = meta.get("source_name")
    if not source_name:
        sf = meta.get("source_file", "")
        source_name = Path(sf).name if sf else None
    return {
        "dataset": meta.get("dataset"),
        "video_name": meta.get("video_name"),
        "source_name": source_name,
        "source_size": meta.get("source_size"),
        "shape_hwt": list(meta.get("shape_hwt", [])) if meta.get("shape_hwt") is not None else None,
        "sampling_rate": meta.get("sampling_rate"),
        "seed": meta.get("seed"),
        "layout": meta.get("layout"),
        "input_key": None if meta.get("input_key") in (None, "") else str(meta.get("input_key")),
        "max_frames": meta.get("max_frames"),
        "resize_height": meta.get("resize_height"),
        "resize_width": meta.get("resize_width"),
        "sampling_type": meta.get("sampling_type"),
    }

def meta_scientifically_compatible(old_meta, current_core):
    old = legacy_meta_scientific_core(old_meta)
    cur = {k: _norm(current_core.get(k)) for k in SCIENCE_KEYS}
    # source_size/source_name are provenance checks; scientific content verification
    # below is authoritative when actual case files are available.
    for k in SCIENCE_KEYS:
        ov, cv = _norm(old.get(k)), _norm(cur.get(k))
        if k == "sampling_rate":
            try:
                if abs(float(ov) - float(cv)) > 1e-12:
                    return False, f"meta_mismatch:{k}:{ov}!={cv}"
                continue
            except Exception:
                return False, f"meta_mismatch:{k}:{ov}!={cv}"
        if ov != cv:
            return False, f"meta_mismatch:{k}:{ov}!={cv}"
    return True, "meta_science_match"

def case_files_present(base):
    base = Path(base)
    npz = Path(str(base) + ".npz")
    mat = Path(str(base) + ".mat")
    js = Path(str(base) + ".json")
    return (
        npz.exists() and npz.stat().st_size > 256 and
        mat.exists() and mat.stat().st_size > 256 and
        js.exists()
    )

def case_content_matches(base, gt_current, sampling_rate, seed):
    """Authoritative compatibility test for an existing case.

    It compares the saved GT and deterministic same-mask definition to the
    current source-derived GT. Absolute paths and file mtimes are irrelevant.
    """
    base = Path(base)
    npz = Path(str(base) + ".npz")
    if not npz.exists():
        return False, "case_npz_missing"
    try:
        gt_old, obs_old, mask_old = load_case(npz)
    except Exception as e:
        return False, f"case_load_error:{type(e).__name__}:{e}"

    gt_current = np.asarray(gt_current, dtype=np.float32)
    if gt_old.shape != gt_current.shape:
        return False, f"gt_shape:{gt_old.shape}!={gt_current.shape}"

    # The video loaders used by these benchmark versions are deterministic.
    # Exact equality prevents accidental reuse across different source videos.
    if not np.array_equal(gt_old, gt_current):
        return False, "gt_content_mismatch"

    h, w, t = gt_current.shape
    expected_mask = make_tube_mask(h, w, t, float(sampling_rate), int(seed))
    if mask_old.shape != expected_mask.shape or not np.array_equal(mask_old, expected_mask):
        return False, "mask_content_mismatch"

    # Cheap sanity check for the stored observation.
    if obs_old.shape != gt_current.shape:
        return False, "observed_shape_mismatch"
    if np.any(mask_old) and not np.array_equal(obs_old[mask_old], gt_current[mask_old]):
        return False, "observed_values_mismatch"
    if np.any(~mask_old) and np.any(obs_old[~mask_old] != 0):
        return False, "observed_unobserved_region_nonzero"

    return True, "content_verified"

def refreshed_meta(old_meta, current_core, current_source_file, current_source_mtime_ns):
    old_meta = dict(old_meta or {})
    legacy_sig = old_meta.get("case_signature_sha256")
    out = dict(old_meta)
    out.update(current_core)
    out["source_file"] = str(Path(current_source_file).resolve())
    out["source_mtime_ns"] = int(current_source_mtime_ns)
    out["source_name"] = Path(current_source_file).name
    out["scientific_signature_sha256"] = scientific_signature(current_core)
    out["reuse_verified_by_content"] = True
    out["reuse_compatibility_version"] = "V2.1"
    if legacy_sig and legacy_sig != out["scientific_signature_sha256"]:
        out["legacy_case_signature_sha256"] = legacy_sig
    return out
