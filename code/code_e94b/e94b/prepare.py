from .common import *
import numpy as np
from e90.data import load_data
from e86.data import CAMS
from e88.audit import HorizonIndex
status('CPU_preflight')
p,ready,train,_,cov=load_data()
caps=json.loads((E93/'dev/captures_full.json').read_text());devrows=json.loads((E93/'dev/windows_full.json').read_text())
assert len(devrows)==211 and len({r['parent_component'] for r in devrows})==27
sel=json.loads((E93/'train64_selection.json').read_text());bykey={(r['canonical_id'],r['anchor']):r for r in train.rows};tr64=[bykey[(r['canonical_id'],r['anchor'])] for r in sel['windows']]
assert len(tr64)==507
by=collections.defaultdict(list)
for r in train.rows:by[r['parent_component']].append(r)
ridge=[rs[i] for p,rs in sorted(by.items()) for i in np.linspace(0,len(rs)-1,min(12,len(rs))).astype(int)]
# Keep all 2x4 rows from the original cache: every continuous-horizon/chain endpoint keeps original validity.
cs={c['canonical_id']:c for c in p.captures};cs.update({c['canonical_id']:c for c in caps});trainids={r['canonical_id'] for r in train.rows};devplans=json.loads((E93/'cache_plan.json').read_text())['captures']+json.loads((E93/'cache_plan_supplement.json').read_text())['captures'];dp={c['canonical_id']:c for c in devplans}
fine=collections.defaultdict(lambda:[set() for _ in CAMS]);ixtr=train.index;ixdev=HorizonIndex(E93/'features',caps)
for r in ridge+tr64+devrows:
 ix=ixtr if r['split']=='train' else ixdev;cid=r['canonical_id']
 for k in range(3):fine[cid][k].update(x[k] for x in r['obs_rows'] if x[k]>=0)
 for h in H:
  rr,_=ix.resolve(dict(r,_stage='P1'),h)
  if rr is not None:
   for k in range(3):fine[cid][k].add(rr[k])
plan=[]
for cid in sorted(trainids|set(dp)):
 c=cs[cid];assert cid not in p.excluded
 cams=[]
 for k,cam in enumerate(CAMS):
  if cid in trainids:
   roots=[Path(ready['cache'])];ov=roots[0]/'overlay.json'
   if ov.exists():roots.append(Path(json.loads(ov.read_text())['base_cache']))
   ids=sorted(set(int(v) for root in roots if (root/cid/(cam+'_rows.npy')).exists() for v in np.load(root/cid/(cam+'_rows.npy'))));path=next(x['path'] for x in p.routes[cid]['cameras'] if x['camera']==cam)
  else:ids=dp[cid]['cameras'][k]['rows'];path=dp[cid]['cameras'][k]['path']
  assert set(fine[cid][k])<=set(ids)
  cams.append(dict(camera=cam,path=path,rows=ids,fine_rows=sorted(fine[cid][k])))
 plan.append(dict(canonical_id=cid,parent_component=c['parent_component'],split=c['split'],cameras=cams))
f2=sum(len(m['rows']) for c in plan for m in c['cameras']);f7=sum(len(m['fine_rows']) for c in plan for m in c['cameras']);base=f2*8*768*2;detail=f7*49*768*2
estimate=1046032857+base+detail+3_000_000_000 # copied-and-deleted artifacts + checkpoints/eval/code/log allowance
receipt=dict(captures=plan,train_parents=len(by),dev_parents=27,train_frames=f2,detail_frames=f7,base_bytes=base,detail_bytes=detail,cumulative_write_estimate=estimate,limit=30_000_000_000,refs=p.refs,ready_cache=ready['cache'],windows_sha256=sha(ready['windows']['P1']),dev_windows_sha256=sha(E93/'dev/windows_full.json'),train64_selection_sha256=sha(E93/'train64_selection.json'),fine_extra='Full E93 B5 <=12 windows/each training parent additionally cached for exact alpha protocol, plus original train64 and dev. No future input to regressors.')
atomic(R/'cache_plan.json',receipt);atomic(R/'ridge_train_rows.json',ridge);atomic(R/'train64_rows.json',tr64)
report('## 1 CPU预审与容量估算\n'+json.dumps({k:v for k,v in receipt.items() if k not in ('captures','refs')},ensure_ascii=False,indent=2))
assert estimate<=30e9,'Storage forecast exceeds 30GB; stop before extraction'
status('preflight_complete',estimate_GB=estimate/1e9,frames=f2,fine_frames=f7);print(json.dumps(dict(estimate_GB=estimate/1e9,base_GB=base/1e9,detail_GB=detail/1e9,frames=f2,fine_frames=f7)))
