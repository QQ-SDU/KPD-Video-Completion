from pathlib import Path
from common import available_dataset_names,get_dataset,supported_files
for n in available_dataset_names():
    try:
        c=get_dataset(n); print(f"{n:24s} files={len(supported_files(c)):3d} root={c['root']} protocol={c.get('protocol','generic')}")
    except Exception as e: print(n,'ERROR',e)
