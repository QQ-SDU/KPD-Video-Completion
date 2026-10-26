from __future__ import annotations
from pathlib import Path
import json, hashlib, re, csv, os, secrets

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
ART = ROOT / "artifacts"
DATA_ROOT = ROOT / "data"

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_experiment():
    cfg = load_json(CONFIG_DIR / "experiment.json")
    env_seed = os.environ.get("KPD_EXPERIMENT_SEED")
    if env_seed is not None and str(env_seed).strip() != "":
        cfg["sampling"]["seeds"] = [int(env_seed)]
    elif not cfg.get("sampling", {}).get("seeds"):
        # No seed is fixed in the package. Direct low-level use receives
        # a fresh process-local seed; top-level benchmark runners export
        # KPD_EXPERIMENT_SEED so all child processes share one run seed.
        cfg["sampling"]["seeds"] = [secrets.randbits(32)]
    return cfg

def load_method_defaults():
    return load_json(CONFIG_DIR / "method_parameters.json")

def _dataset_document():
    d=load_json(CONFIG_DIR / "datasets.json")
    if isinstance(d,list):
        return {"defaults":{},"datasets":d}
    return d

def dataset_defaults():
    base={"layout":"HWT","input_key":None,"max_frames":None,
          "resize_height":None,"resize_width":None,"recursive":False,
          "enabled":True,"protocol":"generic"}
    base.update(_dataset_document().get("defaults",{}))
    return base

def load_datasets():
    return _dataset_document().get("datasets",[])

def _folder_config(name):
    root=DATA_ROOT/name
    p=root/"dataset.json"
    if p.exists():
        cfg=load_json(p)
        if not isinstance(cfg,dict):
            raise ValueError(f"{p} must contain a JSON object")
        return cfg
    return {}

def get_dataset(name):
    cfg=dataset_defaults()
    cfg.update({"name":name,"root":f"data/{name}"})
    found=False
    for d in load_datasets():
        if d.get("name")==name:
            cfg.update(d); found=True; break
    root=ROOT/cfg["root"]
    if not found and not root.is_dir():
        raise KeyError(f"Unknown dataset '{name}'. Create {root} and put videos there.")
    # A folder-local dataset.json can override generic settings. For a centrally
    # defined strict protocol, only non-protocol fields should normally be used.
    local=_folder_config(name)
    cfg.update(local)
    cfg["name"]=name
    cfg.setdefault("root",f"data/{name}")
    return cfg

def available_dataset_names():
    names=[]
    for d in load_datasets():
        n=d.get("name")
        if n and n not in names: names.append(n)
    if DATA_ROOT.exists():
        for p in sorted(DATA_ROOT.iterdir(), key=lambda x:x.name.lower()):
            if p.is_dir() and not p.name.startswith("_") and p.name not in names:
                names.append(p.name)
    return names

def enabled_datasets():
    # Only centrally registered datasets are auto-run with no --dataset flag.
    return [get_dataset(d["name"]) for d in load_datasets() if d.get("enabled",True)]

def safe_name(s):
    s=re.sub(r"[^A-Za-z0-9_.-]+","_",str(s))
    return re.sub(r"_+","_",s).strip("_.") or "item"

def sr_tag(sr):
    return f"{int(round(float(sr)*100)):03d}"

def method_profile_payload():
    # Sampling-rate values and dataset definitions are case-level information;
    # they are intentionally NOT part of the method profile so V1.2 SR=.1-.3
    # outputs remain reusable when SR=.4/.5 are added.
    e=load_experiment()
    return {"profile_name":e["profile_name"],"sampling_type":e["sampling"]["type"],
            "methods":e["methods"],"method_parameters":load_method_defaults(),
            "evaluation":e["evaluation"],"fairness":e["fairness"]}

def profile_id():
    payload=json.dumps(method_profile_payload(),sort_keys=True,separators=(",",":"))
    h=hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"{load_experiment()['profile_name']}__{h}"

def sync_profile():
    ART.mkdir(parents=True,exist_ok=True)
    p=ART/"CURRENT_PROFILE.txt"; p.write_text(profile_id()+"\n",encoding="utf-8")
    return profile_id()

def case_dir(dataset): return ART/"cases"/safe_name(dataset)
def result_root(dataset): return ART/"results"/safe_name(dataset)/profile_id()
def recon_dir(dataset,method=None):
    base=result_root(dataset)/"reconstructions"; return base/method if method else base
def metrics_dir(dataset): return result_root(dataset)/"metrics"
def tables_dir(dataset): return result_root(dataset)/"tables"
def figures_dir(dataset): return result_root(dataset)/"figures"
def logs_dir(dataset): return ART/"logs"/safe_name(dataset)/profile_id()
def quarantine_dir(dataset): return ART/"quarantine"/safe_name(dataset)/profile_id()

def supported_files(dataset_cfg):
    root=ROOT/dataset_cfg["root"]
    if not root.exists(): return []
    exts={".y4m",".npy",".npz",".mat"}
    if dataset_cfg.get("recursive",False):
        xs=[p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        xs=[p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(xs,key=lambda p:p.name.lower())

def load_optional_manifest(dataset_cfg):
    root=ROOT/dataset_cfg["root"]; p=root/"manifest.csv"
    if not p.exists(): return {}
    out={}
    with p.open("r",encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            if row.get("file"): out[row["file"]]=row
    return out

def count_enabled_sources(dataset_cfg):
    manifest=load_optional_manifest(dataset_cfg)
    n=0
    for p in supported_files(dataset_cfg):
        row=manifest.get(p.name,{})
        if str(row.get("enabled","1")).strip().lower() not in {"0","false","no"}: n+=1
    return n
