from common import *
from contract import sha
import numpy as np,os
records=[]
for arm in ['left','right']:
 p=P/'matched_frozen'/f'motion_robot_{arm}_h50_predictions.npz';before=sha(p)
 with np.load(p,allow_pickle=True) as f:arrays={k:f[k] for k in f.files}
 assert all(isinstance(x,str) for x in arrays['parents']);original={k:a.copy() for k,a in arrays.items()};arrays['parents']=arrays['parents'].astype(str)
 tmp=str(p)+'.unicode.tmp'
 with open(tmp,'wb') as f:np.savez_compressed(f,**arrays)
 os.replace(tmp,p)
 with np.load(p) as f:
  assert all(np.array_equal(f[k],a) for k,a in original.items())
 records.append(dict(path=str(p),before_copy_sha256=before,after_sha256=sha(p),only_change='object string metadata -> unicode ndarray; labels/scores/order exactly equal',original_source=str(R/'probes'/PILOT/'matched_frozen'/p.name)))
atomic(P/'matched_metadata_fix.json',dict(timestamp_brt=brt(),records=records,gpu_reruns=0,model_reruns=0))
p=C/'matched_head.py';s=p.read_text().replace('parents=ps,**scores','parents=ps.astype(str),**scores');p.write_text(s)
print('matched_metadata_unicode',len(records))
