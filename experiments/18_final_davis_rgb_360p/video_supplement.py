#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create REAL RGB multi-method supplementary comparison MP4s, no synthetic frames.

The same fixed spatial ROI is used in all columns and on all video frames.
The codec is MP4V for broad Windows OpenCV compatibility; remux/transcode to
H.264 when required by a publisher (FFmpeg optional, not bundled here).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat
import frozen_benchmark as fb


def _frame_video_data(source,v12,out,case,columns):
    gt,obs,mask=fb.load_case(source,case)
    data={'GT':np.clip(gt*255+0.5,0,255).astype(np.uint8),
          'Observed':np.clip(obs*255+0.5,0,255).astype(np.uint8)}
    for name in columns:
        if name in data:continue
        file,where=fb.locate_recon(source,v12,out,case,name)
        if file is None:raise FileNotFoundError('RGB reconstruction missing: '+case+' / '+name)
        d=loadmat(file,variable_names=['recon'])
        arr=np.asarray(d['recon'],dtype=np.float32)
        if arr.shape!=fb.SHAPE or not np.isfinite(arr).all():
            raise ValueError('Bad RGB shape or nonfinite: '+str(file))
        # The same data consistency rule is applied without modifying original MATs.
        if np.max(np.abs(arr[mask]-obs[mask]))>0.002:
            raise ValueError('Incompatible source mask or observed pixels: '+str(file))
        arr=np.clip(arr,0,1)
        arr[mask]=obs[mask]
        data[name]=np.clip(arr*255+0.5,0,255).astype(np.uint8)
        del d,arr
    return gt,obs,mask,data


def _make_canvas(data,frame,columns,roi,tile_width,case,full_sr):
    src_h,src_w=fb.SHAPE[:2]
    x,y,rw,rh=roi
    title_h=36
    full_h=round(tile_width*src_h/src_w)
    zoom_h=round(tile_width*rh/rw)
    foot_h=36
    pad=6
    canvas=np.full((title_h+full_h+pad+zoom_h+foot_h,len(columns)*tile_width,3),255,np.uint8)
    for j,method in enumerate(columns):
        xoff=j*tile_width
        rgb=data[method][:,:,frame,:]
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        full=cv2.resize(bgr,(tile_width,full_h),interpolation=cv2.INTER_AREA)
        fx1=round(x*tile_width/src_w);fy1=round(y*full_h/src_h)
        fx2=round((x+rw)*tile_width/src_w);fy2=round((y+rh)*full_h/src_h)
        cv2.rectangle(full,(fx1,fy1),(fx2,fy2),(15,35,240),2)
        canvas[title_h:title_h+full_h,xoff:xoff+tile_width]=full
        crop=bgr[y:y+rh,x:x+rw]
        zoom=cv2.resize(crop,(tile_width,zoom_h),interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(zoom,(0,0),(tile_width-1,zoom_h-1),(15,35,240),2)
        canvas[title_h+full_h+pad:title_h+full_h+pad+zoom_h,xoff:xoff+tile_width]=zoom
        cv2.putText(canvas,method,(xoff+7,25),cv2.FONT_HERSHEY_SIMPLEX,
                    .65 if len(method)<15 else .52,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(canvas,f'frame {frame:02d}  SR {full_sr:.1f}',
                    (xoff+8,title_h+full_h+pad+zoom_h+24),
                    cv2.FONT_HERSHEY_SIMPLEX,.48,(45,45,45),1,cv2.LINE_AA)
        if j:cv2.line(canvas,(xoff,0),(xoff,canvas.shape[0]-1),(205,205,205),1)
    return canvas


def create_mp4(source,v12,out,video,sr,layout,main_columns,roi_spec,fps,cell_width,start,end,require_complete=False):
    if not 1<=fps<=120:raise ValueError('--fps must be between 1 and 120')
    if not 160<=cell_width<=640:raise ValueError('--video-cell-width must be in [160,640]')
    if not 0<=start<end<=fb.SHAPE[2]:raise ValueError('Video frame range must satisfy 0 <= start < end <= 50')
    columns=tuple(x.strip() for x in main_columns.split(',')) if layout=='main' else fb.SUPPLEMENT
    if len(set(columns))!=len(columns) or not {'Observed','GT','Ours-Frozen'}.issubset(columns):
        raise ValueError('Invalid movie column layout')
    if layout=='main' and len(columns)!=8:raise ValueError('Main movie requires exactly 8 columns')
    case=fb.case_id(video,sr)
    absent=[m for m in columns if m not in ('GT','Observed') and fb.locate_recon(source,v12,out,case,m)[0] is None]
    if absent:
        msg='MP4 missing reconstruction '+case+': '+','.join(absent)
        if require_complete:raise RuntimeError(msg)
        print('VIDEO SKIPPED',msg,flush=True)
        return None
    gt,obs,mask,data=_frame_video_data(source,v12,out,case,columns)
    # Deterministic ROI: method-neutral GT edge energy in missing area of frame 20.
    roi=fb._roi(gt,mask,20) if roi_spec=='auto' else tuple(map(int,roi_spec.split(',')))
    if len(roi)!=4:raise ValueError('ROI format x,y,w,h')
    x,y,rw,rh=roi
    if not (rw>0 and rh>0 and x>=0 and y>=0 and x+rw<=fb.SHAPE[1] and y+rh<=fb.SHAPE[0]):
        raise ValueError('ROI outside 360x640 RGB frame')
    filedir=out/'videos'/('main_8columns' if layout=='main' else 'supplement_allmethods')
    filedir.mkdir(parents=True,exist_ok=True)
    destination=filedir/(case+f'__frame{start:02d}-{end-1:02d}.mp4')
    first=_make_canvas(data,start,columns,roi,cell_width,case,sr)
    h,w=first.shape[:2]
    writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h),True)
    if not writer.isOpened():raise RuntimeError('OpenCV MP4V video codec unavailable; install FFmpeg-enabled OpenCV')
    try:
        writer.write(first)
        for t in range(start+1,end):writer.write(_make_canvas(data,t,columns,roi,cell_width,case,sr))
    finally:
        writer.release()
    if not destination.is_file() or destination.stat().st_size<1024:
        raise RuntimeError('MP4 output empty: '+str(destination))
    source_file=fb.orig.paths(source,case)
    metadata={'case':case,'resolution':[360,640,50,3],'frames':[start,end],
              'fps':fps,'codec':'mp4v','layout':layout,'columns':columns,'roi_xywh':roi,
              'mask_source_sha256':fb.sha256(source_file),
              'frozen_model':{'K':12,'hidden':32,'steps':3600,'patch':[16,96,96]},
              'roi_policy':'GT missing-edge geometry on frame 20, fixed across all methods and frames',
              'note':'Real saved RGB reconstructions only; no interpolation of missing method outputs.'}
    destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print('VIDEO READY',destination,f'({w}x{h}, {end-start} frames, {fps:g}fps)',flush=True)
    return destination
