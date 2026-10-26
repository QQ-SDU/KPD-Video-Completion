# -*- coding: utf-8 -*-
"""
Download the three Xiph development videos used in the current project.
If a mirror URL changes, place the corresponding .y4m files manually in data/.
"""
from pathlib import Path
import urllib.request

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    "akiyo_qcif.y4m": "https://media.xiph.org/video/derf/y4m/akiyo_qcif.y4m",
    "foreman_qcif.y4m": "https://media.xiph.org/video/derf/y4m/foreman_qcif.y4m",
    "mobile_qcif.y4m": "https://media.xiph.org/video/derf/y4m/mobile_qcif.y4m",
}

for name, url in URLS.items():
    path = OUT / name
    if path.exists() and path.stat().st_size > 1000:
        print("[exists]", path)
        continue
    print("[download]", url)
    try:
        urllib.request.urlretrieve(url, path)
        print("[saved]", path)
    except Exception as e:
        print("[FAILED]", name, e)

print("Done.")
