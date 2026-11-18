import json,collections,numpy as np
from e88.audit import read_inputs,HorizonIndex,HORIZONS
from e88.common import ART,atomic
p,ready,ws,stats=read_inputs();ix=HorizonIndex(ready['cache'],p.captures);out={}
for h in HORIZONS:
 need=[set(),set(),set()];valid=[set(),set(),set()]
 for stage,rs in ws.items():
  for row in rs:
   ids,reason=ix.resolve(dict(row,_stage=stage),h,require_cached=False)
   if ids is not None:
    _,available=ix.get(row['canonical_id'])
    for cam,j in enumerate(ids):
     key=(row['canonical_id'],j);need[cam].add(key)
     if j in available[cam]:valid[cam].add(key)
 out[str(h)]={'required_unique_source_valid_camera_frames':[len(x) for x in need],'already_cached_camera_frames':[len(x) for x in valid],'missing_camera_frames':[len(a-b) for a,b in zip(need,valid)],'sample_required_rows':[sorted(x)[:4] for x in need],'added_bytes':0};atomic(ART/'cpu_audit_01/cache_reads.json',out)
print(json.dumps(out))
