#!/usr/bin/env python3
"""Prepare an external DAVIS2017 validation subset for the frozen video-completion test.

No DAVIS labels are used. Do not alter data based on the held-out completion result.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import cv2
import numpy as np

DEFAULT_NAMES = ('blackswan','bmx-trees','breakdance','camel','car-roundabout','drift-chicane','kite-surf','parkour')

def _find_jpeg_root(root: Path, names) -> Path:
    paths = [root/'JPEGImages'/'480p', root/'DAVIS'/'JPEGImages'/'480p', root/'2017'/'JPEGImages'/'480p', root]
    for p in paths:
        if p.is_dir() and all((p/name).is_dir() for name in names):
            return p
    raise FileNotFoundError(f"DAVIS JPEGImages/480p not found below {root}; checked: {paths}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--davis-root',required=True,help='Extracted DAVIS-2017-trainval-480p DAVIS folder')
    p.add_argument('--out',default=None,help='Default benchmark_suite/data/davis2017_new')
    p.add_argument('--sequences',default=','.join(DEFAULT_NAMES))
    p.add_argument('--height',type=int,default=144)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--frames',type=int,default=50)
    p.add_argument('--append',action='store_true',help='Keep previously prepared video entries in manifest.csv')
    a=p.parse_args()
    names=[x.strip() for x in a.sequences.split(',') if x.strip()]
    if len(names)!=len(set(names)): raise ValueError('Duplicate sequence names')
    src=_find_jpeg_root(Path(a.davis_root).resolve(),names)
    out=Path(a.out).resolve() if a.out else Path(__file__).resolve().parents[2]/'benchmark_suite'/'data'/'davis2017_new'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in names:
        folder=src/name
        if not folder.is_dir(): raise FileNotFoundError(f'Missing DAVIS sequence: {folder}')
        imgs=sorted([*folder.glob('*.jpg'), *folder.glob('*.jpeg'), *folder.glob('*.png')])
        if len(imgs)<a.frames: raise ValueError(f'{name}: only {len(imgs)} frames; need {a.frames}')
        frames=[]
        for f in imgs[:a.frames]:
            rgb=cv2.imread(str(f),cv2.IMREAD_COLOR)
            if rgb is None: raise ValueError(f'Cannot open {f}')
            # OpenCV BGR->GRAY, matching input grayscale protocol of existing code.
            gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
            interpolation = cv2.INTER_AREA if gray.shape[0]>=a.height and gray.shape[1]>=a.width else cv2.INTER_LINEAR
            resized=cv2.resize(gray,(a.width,a.height),interpolation=interpolation)
            frames.append(resized.astype(np.float32)/255.0)
        arr=np.stack(frames,axis=2)
        file=out/f'{name}.npy'
        np.save(file,arr,allow_pickle=False)
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        rows.append(dict(name=name,file=file.name,width=a.width,height=a.height,frames=a.frames,group='DAVIS2017_val',sha256=digest))
        print(f'[OK] {name}: {arr.shape}, sha256={digest[:16]}')
    manifest=out/'manifest.csv'
    if a.append and manifest.exists():
        with manifest.open('r',encoding='utf-8-sig',newline='') as previous:
            existing={r['name']:r for r in csv.DictReader(previous)}
        for item in rows:
            # Existing files are reproducibly rewritten by the same image pipeline.
            if item['name'] in existing and existing[item['name']].get('sha256')!=item['sha256']:
                raise ValueError('Existing video has different SHA256: '+item['name'])
            existing[item['name']]=item
        rows=sorted(existing.values(),key=lambda q:q['name'])
    with manifest.open('w',encoding='utf-8-sig',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    info=dict(source='DAVIS 2017 trainval 480p, public validation sequences', source_root=str(src),
              source_url='https://davischallenge.org/davis2017/code.html', names=[r['name'] for r in rows],
              temporal_selection='first 50 frames; no skipping', spatial_resize=[a.height,a.width],
              color='cv2.COLOR_BGR2GRAY', array_layout='HWT', annotation_used=False)
    (out/'PREPROCESSING.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Prepared {len(rows)} sequences in {out}')

if __name__=='__main__':main()
