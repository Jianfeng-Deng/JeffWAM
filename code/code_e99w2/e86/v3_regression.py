"""Actual training video/cache regressions across all four strata; no annotation."""
from pathlib import Path
import json,time,traceback
from .common import *
from .v3_train import DEFAULT_INPUT

def main():
 from .data import Package,CachedDataset,CAMS,collate,move
 from .resource_v3 import Guard
 A=DEFAULT_INPUT;dest=A/'real_cache_regression';dest.mkdir(exist_ok=False);p=Package();ready=json.loads((A/'READY_V3_cache_windows.json').read_text());ws=json.loads(Path(ready['windows']['P3']).read_text());stats=json.loads((A/'stats.json').read_text());selected=[next(r for r in ws if r['stratum']==st) for st in ('demo3','demo4','demo6early','demo6native')];ds=CachedDataset(selected,p.captures,ready['cache'],stats);guard=Guard('V3_actual_online_cache_regression',180)
 try:
  import torch,numpy as np,cv2
  from .vision import FrozenEncoder,preprocess
  from .model import WAM
  from .v3_train import configure
  configure();cv2.setNumThreads(1);e=FrozenEncoder('cuda:0');m=WAM().eval().to('cuda:0');records=[]
  for j,r in enumerate(selected):
   guard.check();offline=move(collate([ds[j]]),'cuda:0');online=dict(offline,obs=offline['obs'].clone());diffs=[];hw=[]
   for k,cam in enumerate(CAMS):
    route=next(c for c in p.routes[r['canonical_id']]['cameras'] if c['camera']==cam);rr=r['obs_rows'][-1][k];cap=cv2.VideoCapture(route['path']);assert cap.isOpened()
    try:
     # Decode sequentially to preserve exact source-frame identity.
     for _ in range(rr+1):ok=cap.grab();assert ok
     ok,bgr=cap.retrieve();assert ok;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);x,v,_=preprocess(rgb);z=e.images(x[None],v[None]);ref=offline['obs'][:,4,k];diffs.append(float((z-ref).abs().max()));online['obs'][:,4,k]=z;hw.append(list(rgb.shape[:2]))
    finally:cap.release()
   with torch.no_grad():
    pred1,_=m.infer(offline,steps=2);pred2,_=m.infer(online,steps=2)
   error=float((pred1-pred2).abs().max());record=dict(stratum=r['stratum'],canonical_id=r['canonical_id'],source_hw=hw,feature_max_abs_by_camera=diffs,policy_max_abs=error,scope='Current three cameras replayed from real RGB; same cached history and W; fixed untrained policy, 2 Euler steps; fp16 encoder batch-size rounding may differ')
   # fp16 convolution batching changes final ULPs; do not claim bit identity when not observed.
   if max(diffs)>.04 or error>.01:raise RuntimeError('Cache/online mismatch outside numerical tolerance: '+str(record))
   records.append(record);atomic(dest/'progress.json',records)
  atomic(dest/'READY_regression.json',dict(status='passed',strata=records,elapsed_s=guard.check(),feature_tolerance=.04,policy_tolerance=.01,formal_training_started=False))
 except BaseException as ex:atomic(dest/'status.json',dict(status='failed',reason=str(ex),traceback=traceback.format_exc()));raise
 finally:guard.finish()
if __name__=='__main__':main()
