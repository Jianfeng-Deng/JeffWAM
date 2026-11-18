import os,sys,json,collections
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='');sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';sys.path.insert(0,str(T/'code_e106c'))
from e86.data import Package
import numpy as np
p=Package();caps={c['canonical_id']:c for c in p.captures};full=json.loads((O/'FULL_CAPTURE_STATS.json').read_text());out={}
for cid,f in full.items():
 if 'interval_s' in f:continue
 z=np.load(caps[cid]['numeric_cache'],allow_pickle=False);t=z['timestamp'];seg=z['continuous_segment'];intervals=[]
 for k in range(2):
  ts=sorted(e[0] for e in f['events'] if e[1]==k)
  intervals.extend(float(t[b]-t[a]) for a,b in zip(ts[:-1],ts[1:]) if seg[a]==seg[b])
 out[cid]=intervals
q=O/'FULL_INTERVALS_EXTRA.json';assert not q.exists();q.write_text(json.dumps(out,indent=2));print('extra capture intervals',len(out))
