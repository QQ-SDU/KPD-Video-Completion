#!/usr/bin/env python3
"""DAVIS 2017 480p RGB -> uint-normalized H,W,T,3, Unicode-safe on Windows."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import cv2
import numpy as np

DEFAULT_NAMES=('blackswan','bmx-trees','breakdance','camel','car-roundabout','drift-chicane','kite-surf','parkour')

def _find_root(root, names):
    choices=[root/'JPEGImages'/'480p',root/'DAVIS'/'JPEGImages'/'480p',root/'2017'/'JPEGImages'/'480p',root]
    for p in choices:
        if p.is_dir() and all((p/name).is_dir() for name in names): return p
    raise FileNotFoundError('DAVIS JPEGImages/480p not found in '+str(choices))

def get_sha(file):
    h=hashlib.sha256()
    with Path(file).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--davis-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--sequences',default=','.join(DEFAULT_NAMES))
    p.add_argument('--height',type=int,default=144)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--frames',type=int,default=50)
    p.add_argument('--append',action='store_true')
    p.add_argument('--native-resolution',action='store_true',help='Use original DAVIS JPEG frame size without resizing')
    p.add_argument('--all-frames',action='store_true',help='Use every available frame of each video')
    a=p.parse_args()
    names=[x.strip() for x in a.sequences.split(',') if x.strip()]
    if len(names)!=len(set(names)) or not names:raise ValueError('Empty or duplicate sequences')
    source=_find_root(Path(a.davis_root).resolve(),names)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    old={}
    manifest=out/'manifest.csv'
    if a.append and manifest.exists():
        with manifest.open('r',encoding='utf-8-sig',newline='') as f:old={x['name']:x for x in csv.DictReader(f)}
    for name in names:
        folder=source/name
        imgs=sorted([*folder.glob('*.jpg'),*folder.glob('*.jpeg'),*folder.glob('*.png')])
        if len(imgs)<a.frames and not a.all_frames:raise ValueError(f'{name}: {len(imgs)} frames available, need {a.frames}')
        selected=imgs if a.all_frames else imgs[:a.frames]
        if not selected:raise ValueError(f'{name}: no frames found in {folder}')
        frames=[]
        native_hw=None
        for file in selected:
            # cv2.imread() can fail on Windows when the folder name has Chinese Unicode characters.
            encoded=np.frombuffer(file.read_bytes(),dtype=np.uint8)
            bgr=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
            if bgr is None:raise ValueError(f'Unable to decode {file}')
            if a.native_resolution:
                if native_hw is None: native_hw=bgr.shape[:2]
                elif bgr.shape[:2]!=native_hw:
                    raise ValueError(f'{name}: original frames have inconsistent resolutions: {native_hw} and {bgr.shape[:2]}')
                small=bgr
            else:
                interpolation=cv2.INTER_AREA if bgr.shape[0]>=a.height and bgr.shape[1]>=a.width else cv2.INTER_LINEAR
                small=cv2.resize(bgr,(a.width,a.height),interpolation=interpolation)
            frames.append(cv2.cvtColor(small,cv2.COLOR_BGR2RGB).astype(np.float32)/255.)
        # input frames: (T,H,W,C) => stored (H,W,T,C)
        arr=np.transpose(np.stack(frames,axis=0),(1,2,0,3))
        file=out/(name+'.npy')
        # Don't rewrite the exact same input file needlessly; protects incremental runs.
        if file.exists() and a.append:
            prior=np.load(file,mmap_mode='r')
            if prior.shape!=arr.shape or not np.array_equal(prior,arr):
                raise RuntimeError('Existing RGB data differs for '+name+'; do not mix experiments. Use a new --data directory.')
        else:np.save(file,arr,allow_pickle=False)
        hh,ww,tt,_=arr.shape
        entry=dict(name=name,file=file.name,width=ww,height=hh,frames=tt,
                   channels=3,layout='HWTC',group='DAVIS2017_val_RGB',sha256=get_sha(file))
        if name in old and old[name].get('sha256')!=entry['sha256']:
            raise RuntimeError('Existing manifest differs for '+name)
        old[name]=entry
        print(f'[RGB OK] {name}: {arr.shape}, sha256={entry["sha256"][:16]}',flush=True)
    rows=sorted(old.values(),key=lambda r:r['name'])
    with manifest.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'PREPROCESSING.json').write_text(json.dumps(dict(source=str(source),names=[r['name'] for r in rows],
       source_url='https://davischallenge.org/davis2017/code.html',
       spatial_resize=('native' if a.native_resolution else [a.height,a.width]),
       temporal_selection=('all available frames' if a.all_frames else f'first {a.frames} frames; no skipping'),
       original_frame_sizes={r['name']:[int(r['height']),int(r['width']),int(r['frames'])] for r in rows},
       color='RGB; no grayscale conversion',
       layout='H,W,T,3',annotation_used=False),indent=2,ensure_ascii=False),encoding='utf-8')
    print('Prepared RGB:',len(rows),'videos in',out,flush=True)
if __name__=='__main__':main()
