from __future__ import annotations
from pathlib import Path
from datetime import datetime
import math, shutil, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Benchmark"))
sys.path.insert(0, str(ROOT / "workflow"))

from benchmark.evaluate_output import load_case, load_recon, align_shape
from common import quarantine_dir

def validate_output(case_mat: Path, out: Path, require_runtime=True, check_stale=True):
    """Validate a reconstruction by scientific content, not filesystem age.

    V2.0 treated `out.mtime < case.mtime` as invalid. That is unsafe when
    experiments are migrated between folders or case files are regenerated with
    identical GT/mask content. V2.1 never invalidates an otherwise valid output
    solely because its file timestamp is older than the case.
    """
    case_mat = Path(case_mat)
    out = Path(out)
    case_npz = case_mat.with_suffix(".npz")

    if not out.exists():
        return False, "missing"
    if out.stat().st_size < 512:
        return False, "too_small"
    if not case_npz.exists():
        return False, "case_npz_missing"

    older_than_case = False
    if check_stale:
        try:
            newest = max(case_mat.stat().st_mtime, case_npz.stat().st_mtime)
            older_than_case = out.stat().st_mtime + 1e-3 < newest
        except Exception:
            older_than_case = False

    try:
        gt, _, _ = load_case(case_npz)
        rec, runtime = load_recon(out)
        rec = align_shape(rec, gt.shape)
        if rec.shape != gt.shape:
            return False, f"shape_{rec.shape}_vs_{gt.shape}"
        if not np.all(np.isfinite(rec)):
            return False, "nonfinite_reconstruction"
        if require_runtime:
            if runtime is None or not math.isfinite(float(runtime)) or float(runtime) < 0:
                return False, "invalid_runtime"

        # Timestamp mismatch is informational only.
        if older_than_case:
            return True, "ok_reuse_older_than_case"
        return True, "ok"
    except Exception as e:
        return False, f"load_error:{type(e).__name__}:{e}"

def quarantine_output(dataset, method, out, reason):
    out = Path(out)
    if not out.exists():
        return None
    qdir = quarantine_dir(dataset) / method
    qdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_reason = reason[:60].replace(":", "_").replace("/", "_").replace("\\", "_")
    dest = qdir / f"{out.stem}__{stamp}__{safe_reason}{out.suffix}"
    shutil.move(str(out), str(dest))
    return dest
