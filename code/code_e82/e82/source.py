from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import torch
from .data import causal_dq
from .image import prepare

CAMERA_NAMES=('cam_high','cam_left_wrist','cam_right_wrist')
CROP_CAMERA=(0,1,1,2,2)
# Small in-memory numeric LRU only. No source writes or persistent pixel duplication.
_NUMERIC=OrderedDict()
def numeric(path):
    path=str(path)
    if path not in _NUMERIC:
        with np.load(path,allow_pickle=False) as f: _NUMERIC[path]={k:f[k] for k in f.files}
        if len(_NUMERIC)>4: _NUMERIC.popitem(last=False)
    _NUMERIC.move_to_end(path); return _NUMERIC[path]

def video_frame(path,index):
    import cv2
    capture=cv2.VideoCapture(str(path)); capture.set(cv2.CAP_PROP_POS_FRAMES,int(index))
    actual=capture.get(cv2.CAP_PROP_POS_FRAMES); ok,frame=capture.read(); capture.release()
    if not ok or abs(actual-int(index))>.1: raise IOError(f'Untraceable frame: {path} row {index}, actual {actual}')
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

def video_frames(path,indices):
    import cv2
    ids=sorted(set(map(int,indices))); result={}
    capture=cv2.VideoCapture(str(path)); capture.set(cv2.CAP_PROP_POS_FRAMES,ids[0])
    position=int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    if position!=ids[0]:capture.release();raise IOError(f'Untraceable video seek: {path}')
    wanted=set(ids)
    while position<=ids[-1]:
        ok,frame=capture.read()
        if not ok:capture.release();raise IOError(f'Failed original frame {path}:{position}')
        if position in wanted:result[position]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        position+=1
    capture.release();return result

def read_window(row,stage):
    if row.get('schema_version')=='e82-piper-next-feedback-v1': return read_D_window(row,stage)
    # Portable explicit sample references, also used for future data-publisher versions.
    if 'numeric_cache' not in row or 'O_video_rows' not in row: raise ValueError('Unsupported window schema; no guessed index mapping')
    return read_D_window(row,stage)

def read_D_window(row,stage):
    import h5py
    nums=numeric(row['numeric_cache']); i=int(row['anchor']); decision=float(row['decision_time'])
    source=Path(row['source_path']); camera_paths=row.get('camera_paths') or {c:str(source.parent/'video'/c/(source.stem+'.mp4')) for c in CAMERA_NAMES}
    transforms=row.get('approved_transforms',{})
    roi=transforms.get('wrist_roi',{})  # unknown ROI is structurally absent, never guessed
    top_box=transforms.get('top_black_bar_crop')
    oidx=np.asarray(row['O_video_rows']); widx=np.asarray(row['W_video_rows'])
    ct=nums['camera_exposure_timestamp']; recv=nums['camera_received_timestamp']
    for k in range(2):
        historical_decision=float(nums['state_available_time'][i-6 if k==0 else i])
        for c in range(3):
            j=int(oidx[k,c]); age=historical_decision-ct[j,c]
            if max(ct[j,c],recv[j,c])>historical_decision+1e-7 or not (0<=age<=.1+1e-7): raise ValueError('Future/stale O frame')
    for k,off in enumerate((10,25,50)):
        for c in range(3):
            if abs(ct[int(widx[k,c]),c]-nums['timestamp'][i+off])>1/60+1e-6: raise ValueError('W exposure outside tolerance')
    frame_cache={}
    for c in range(3):
        requested=np.concatenate((oidx[:,c],widx[:,c]))
        for j,rgb in video_frames(camera_paths[CAMERA_NAMES[c]],requested).items():frame_cache[c,j]=rgb
    def frame(c,j):
        key=(c,int(j))
        if key not in frame_cache: frame_cache[key]=video_frame(camera_paths[CAMERA_NAMES[c]],j)
        return frame_cache[key]
    def image_group(indices):
        images=[]; masks=[]; valids=[]; boxes=[]
        for crop,c in enumerate(CROP_CAMERA):
            zs=[]; ms=[]; vs=[]; bs=[]
            for j in indices[:,c]:
                rgb=frame(c,j); h,w=rgb.shape[:2]; box=top_box if crop==0 else roi.get('left' if c==1 else 'right') if crop in (2,4) else None
                missing=crop in (2,4) and box is None
                if missing:
                    image=torch.zeros(3,224,224); mask=torch.zeros(1,224,224); box_norm=[0,0,1,1]
                else:
                    image,mask,_=prepare(rgb,(252,448) if crop==0 else (224,224),box)
                    b=box if box is not None else [0,0,w,h]; box_norm=(np.asarray(b)/[w,h,w,h]).tolist()
                zs.append(image); ms.append(mask); vs.append(not missing); bs.append(box_norm)
            images.append(torch.stack(zs)[None]); masks.append(torch.stack(ms)[None]); valids.append(vs); boxes.append(bs)
        return images,masks,torch.tensor(valids,dtype=torch.bool).T[None],torch.tensor(boxes,dtype=torch.float32).permute(1,0,2)[None]
    oi,om,ov,boxes=image_group(oidx); fi,fm,fv,_=image_group(widx)
    # State history is measured q, with each row's own availability checked.
    with h5py.File(source,'r') as f:
        if f.attrs.get('sim',False) or f.attrs.get('simulation',False):raise ValueError('Simulation excluded')
        qs=f['observations/qpos'][max(0,i-7):i+1].astype(np.float32)
        qp=f['observations/qpos'][i-1].astype(np.float32)
        q=f['observations/qpos'][i].astype(np.float32)
    t=nums['timestamp']; dq,dqvalid=causal_dq(q,qp,t[i],t[i-1],nums['state_available_time'][i-1]<=decision)
    hist=np.zeros((8,16),np.float32)
    for k,j in enumerate(range(i-7,i+1)):
        if j>=0 and nums['state_available_time'][j]<=decision and nums['continuous_segment'][j]==nums['continuous_segment'][i]:
            hist[k,:14]=qs[j-max(0,i-7)]; hist[k,14]=1; hist[k,15]=decision-t[j]
    actions=nums['action'][i:i+50].copy()
    original=nums['original_action_index'][i:i+50]
    if not np.array_equal(original,np.arange(i+1,i+51)): raise ValueError('Original next-feedback pairing changed')
    def tensor(x,dtype=torch.float32):return torch.tensor(x,dtype=dtype)[None]
    ages=np.asarray(row['O_age_s'])[:,CROP_CAMERA]
    offsets=np.array([nums['state_available_time'][i-6],decision])-decision
    return dict(world_structure_valid=torch.tensor([True]*25+[True]*17+['left' in roi]*16+[True]*17+['right' in roi]*16,dtype=torch.bool)[None,None].expand(1,3,91),obs_images=oi,obs_pixel_valid=om,obs_crop_valid=ov,obs_boxes=boxes,obs_age=tensor(ages),obs_offsets=tensor(offsets),future_images=fi,future_pixel_valid=fm,future_crop_valid=fv,q=tensor(q),dq=tensor(dq),q_valid=torch.ones(1,14,dtype=torch.bool),dq_valid=torch.tensor([dqvalid]),geo=torch.zeros(1,1),state_valid=torch.tensor([[True,dqvalid,False,bool(hist[:,14].any())]]),history=tensor(hist),task_id=torch.tensor([row.get('task_id',0)]),action=tensor(actions),parent_component=[row['parent_component']],session_id=[row.get('session_id') or 'unknown'],window_id=[f"{row['canonical_id']}:{i}"],decision_wait_s=tensor([decision-t[i]]))
