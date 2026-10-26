from __future__ import annotations
from pathlib import Path
import argparse, re, shutil, sys, json
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow"))

from common import (
    profile_id, load_experiment, case_dir, recon_dir, sync_profile, safe_name
)
from output_validation import validate_output

STALE_RE = re.compile(
    r"^(?P<orig>.+)__\d{8}_\d{6}_\d{6}__stale_older_than_case(?:[^.]*)?\.mat$",
    re.IGNORECASE,
)

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def copy_tree_missing(src: Path, dst: Path):
    n = 0
    if not src.exists():
        return n
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        q = dst / rel
        if not q.exists():
            copy_file(p, q)
            n += 1
    return n

def locate_source_profiles(src_root: Path, dataset: str):
    base = src_root / "artifacts" / "results" / safe_name(dataset)
    if not base.exists():
        return []
    exact = base / profile_id()
    dirs = []
    if exact.is_dir():
        dirs.append(exact)
    for p in sorted(base.iterdir()):
        if p.is_dir() and p not in dirs and (p / "reconstructions").is_dir():
            dirs.append(p)
    return dirs

def target_case_from_output(method: str, filename: str, dataset: str):
    prefix = f"{method}__"
    if not filename.startswith(prefix) or not filename.endswith(".mat"):
        return None
    case_stem = filename[len(prefix):-4]
    return case_dir(dataset) / f"{case_stem}.mat"

def import_reconstructions(src_root: Path, dataset: str):
    methods = load_experiment()["methods"]["all"]
    imported = kept = rejected = 0
    details = []

    profiles = locate_source_profiles(src_root, dataset)
    for prof in profiles:
        for method in methods:
            sdir = prof / "reconstructions" / method
            if not sdir.is_dir():
                continue
            for src in sorted(sdir.glob("*.mat")):
                case_mat = target_case_from_output(method, src.name, dataset)
                if case_mat is None or not case_mat.exists():
                    rejected += 1
                    details.append(("REJECT_NO_CASE", method, src.name, str(prof)))
                    continue
                dst = recon_dir(dataset, method) / src.name
                if dst.exists():
                    ok, _ = validate_output(case_mat, dst)
                    if ok:
                        kept += 1
                        continue
                ok, reason = validate_output(case_mat, src)
                if ok:
                    copy_file(src, dst)
                    imported += 1
                    details.append(("IMPORT", method, src.name, reason))
                else:
                    rejected += 1
                    details.append(("REJECT_INVALID_SOURCE", method, src.name, reason))
    return imported, kept, rejected, details

def recover_false_stale_from_source_quarantine(src_root: Path, dataset: str):
    methods = load_experiment()["methods"]["all"]
    qbase = src_root / "artifacts" / "quarantine" / safe_name(dataset)
    if not qbase.exists():
        return 0, 0, []

    restored = rejected = 0
    details = []
    for prof in sorted([p for p in qbase.iterdir() if p.is_dir()]):
        for method in methods:
            qdir = prof / method
            if not qdir.is_dir():
                continue
            for src in sorted(qdir.glob("*__stale_older_than_case*.mat")):
                m = STALE_RE.match(src.name)
                if not m:
                    continue
                orig_name = m.group("orig") + ".mat"
                case_mat = target_case_from_output(method, orig_name, dataset)
                if case_mat is None or not case_mat.exists():
                    rejected += 1
                    details.append(("REJECT_STALE_NO_CASE", method, src.name, ""))
                    continue
                dst = recon_dir(dataset, method) / orig_name
                if dst.exists():
                    ok, _ = validate_output(case_mat, dst)
                    if ok:
                        continue
                ok, reason = validate_output(case_mat, src)
                if ok:
                    copy_file(src, dst)
                    restored += 1
                    details.append(("RESTORE_FALSE_STALE", method, orig_name, reason))
                else:
                    rejected += 1
                    details.append(("REJECT_STALE_INVALID", method, src.name, reason))
    return restored, rejected, details

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--dataset", default="formal23")
    a = ap.parse_args()

    sync_profile()
    src_root = Path(a.source_root).expanduser().resolve()
    if not (src_root / "artifacts").exists():
        raise SystemExit(f"Source artifacts folder not found: {src_root / 'artifacts'}")

    dataset = a.dataset

    print("=" * 88)
    print("V2.1 REUSE-SAFE IMPORT")
    print("Source:", src_root)
    print("Target:", ROOT)
    print("Dataset:", dataset)
    print("Target profile:", profile_id())
    print("=" * 88)

    # Copy source videos only when target files are missing.
    data_n = copy_tree_missing(
        src_root / "data" / dataset,
        ROOT / "data" / dataset,
    )
    print(f"[1] Imported missing data files: {data_n}")

    # Import existing cases first. Existing target cases are never overwritten.
    cases_n = copy_tree_missing(
        src_root / "artifacts" / "cases" / safe_name(dataset),
        case_dir(dataset),
    )
    print(f"[2] Imported missing case files: {cases_n}")

    # Only now can reconstructions be scientifically validated against cases.
    imported, kept, rejected, det1 = import_reconstructions(src_root, dataset)
    print(f"[3] Final reconstructions: imported={imported}, already_valid={kept}, rejected={rejected}")

    restored, stale_rejected, det2 = recover_false_stale_from_source_quarantine(src_root, dataset)
    print(f"[4] False-stale quarantine recovery: restored={restored}, rejected={stale_rejected}")

    report_dir = ROOT / "artifacts" / "migration_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_root": str(src_root),
        "target_root": str(ROOT),
        "dataset": dataset,
        "target_profile": profile_id(),
        "data_files_imported": data_n,
        "case_files_imported": cases_n,
        "reconstructions_imported": imported,
        "reconstructions_already_valid": kept,
        "reconstructions_rejected": rejected,
        "false_stale_restored": restored,
        "false_stale_rejected": stale_rejected,
        "details": [list(x) for x in det1 + det2],
    }
    rp = report_dir / f"reuse_import_{safe_name(dataset)}.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Migration report:", rp)
    print("\nNext: run CHECK_REUSE_STATUS.bat, then RUN_ALL.bat.")

if __name__ == "__main__":
    main()
