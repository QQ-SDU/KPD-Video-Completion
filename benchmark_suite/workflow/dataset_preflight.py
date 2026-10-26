from __future__ import annotations
from pathlib import Path
import argparse,csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Benchmark")); sys.path.insert(0,str(ROOT/"workflow"))
from benchmark.io_utils import load_video
from common import get_dataset,supported_files,load_optional_manifest,safe_name,load_experiment

def formal23(name,dcfg,videos):
    expected=int(dcfg.get("expected_count",23))
    if len(videos)!=expected: raise SystemExit(f"[DATA ERROR] {name}: expected {expected} source videos, found {len(videos)}")
    policy=dcfg.get("frame_policy_by_hw",{"144x176":100,"288x352":50}); hw_counts={k:0 for k in policy}; rows=[]
    for i,src in enumerate(videos,1):
        # Need at most the largest required number of frames for protocol validation.
        probe=max(int(v) for v in policy.values()); x=load_video(src,key=dcfg.get("input_key"),layout=dcfg.get("layout","HWT"),max_frames=probe); h,w,t=map(int,x.shape); key=f"{h}x{w}"
        if key not in policy: raise SystemExit(f"[DATA ERROR] {src.name}: unsupported size {key}; expected {list(policy)}")
        req=int(policy[key]); hw_counts[key]=hw_counts.get(key,0)+1
        if t<req: raise SystemExit(f"[DATA ERROR] {src.name}: {t} frames, need at least {req}")
        rows.append({"file":src.name,"enabled":1,"name":safe_name(src.stem),"layout":dcfg.get("layout","HWT"),"input_key":"","max_frames":req,"resize_height":"","resize_width":""})
        print(f"[{i:02d}/{len(videos)}] {src.name}: {key}, use first {req} frames",flush=True)
    expected_counts={k:int(v) for k,v in dcfg.get("expected_hw_counts",{}).items()}
    if expected_counts and hw_counts!=expected_counts: raise SystemExit(f"[DATA ERROR] size counts {hw_counts}, expected {expected_counts}")
    if dcfg.get("auto_manifest",True):
        p=ROOT/dcfg["root"]/"manifest.csv"; fields=["file","enabled","name","layout","input_key","max_frames","resize_height","resize_width"]
        with p.open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
        print("Generated:",p)

def generic(name,dcfg,videos):
    if not videos: raise SystemExit(f"[DATA ERROR] No supported videos in {ROOT/dcfg['root']}")
    exp_count=dcfg.get("expected_count")
    if exp_count is not None and len(videos)!=int(exp_count): raise SystemExit(f"[DATA ERROR] expected {exp_count}, found {len(videos)}")
    print(f"[PASS] Generic dataset '{name}': {len(videos)} source file(s).",flush=True)
    print("       Global defaults:", {k:dcfg.get(k) for k in ["layout","max_frames","resize_height","resize_width","recursive"]},flush=True)
    m=ROOT/dcfg["root"]/"manifest.csv"
    if m.exists(): print("       Per-video manifest:",m,flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); a=ap.parse_args(); dcfg=get_dataset(a.dataset); root=ROOT/dcfg["root"]; root.mkdir(parents=True,exist_ok=True); videos=supported_files(dcfg)
    if dcfg.get("protocol")=="formal23": formal23(a.dataset,dcfg,videos)
    else: generic(a.dataset,dcfg,videos)
    exp=load_experiment(); print("Sampling rates:",exp["sampling"]["sampling_rates"],"seeds:",exp["sampling"]["seeds"],flush=True)
if __name__=="__main__": main()
