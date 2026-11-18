import json,collections,numpy as np
from e88.audit import read_inputs,HorizonIndex,HORIZONS,labels_index,stage_target
from e88.common import ART,atomic,sha,C,OUT
p,ready,ws,stats=read_inputs();ix=HorizonIndex(ready['cache'],p.captures);labels,source=labels_index(p);bad=collections.Counter();prog={}
for stage,rs in ws.items():
 vals=[]
 for r in rs:
  y,pr,ok=stage_target(labels.get(r['canonical_id'],[]),r['anchor'])
  if ok:vals.append(pr)
  for h in HORIZONS:
   ids,reason=ix.resolve(dict(r,_stage=stage),h)
   if ids is not None:
    z,_=ix.get(r['canonical_id']);mask=z['p1_source_mask' if stage=='P1' else 'p3_source_mask']
    if any(not mask[j] for j in ids):bad[(stage,h)]+=1
 prog[stage]=dict(valid=len(vals),minimum=min(vals) if vals else None,maximum=max(vals) if vals else None,mean=float(np.mean(vals)) if vals else None,at_endpoint=sum(v==1 for v in vals))
atomic(ART/'cpu_audit_01/additional_checks.json',dict(camera_source_mask_failures={str(k):v for k,v in bad.items()},progress_distribution=prog,labels_source=source))
files={str(f.relative_to(C)):sha(f) for f in (C/'e86').glob('*.py')};same=all(sha(OUT/f)==v for f,v in files.items());atomic(ART/'legacy_source_comparison.json',dict(e86_unchanged_from_copy=same,files=files,model_behavior='exact copied e86 source used as legacy reference'))
print(json.dumps(dict(camera_source_mask_failures={str(k):v for k,v in bad.items()},progress=prog,legacy_sources_identical=same)))
