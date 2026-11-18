import json,numpy as np,collections
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';rows=json.loads((O/'ALIGNED_WINDOW_LOCK.json').read_text())['rows'];ready=json.loads((O/'INSPECT.json').read_text())['ready'];cache=Path(ready['cache']);base=Path(json.loads((cache/'overlay.json').read_text())['base_cache']);roots=[cache,base];extras=[T/'code_e86/artifacts'/s for s in ('features_v1','features_v2','features_v3')];extras=[p for p in extras if p not in roots];need=collections.defaultdict(set)
for r in rows:
 for k,cam in enumerate(('cam_high','cam_left_wrist','cam_right_wrist')):need[(r['canonical_id'],cam)].update(a[k] for a in r['obs_rows'] if a[k]>=0)
missing=[];recover=[]
for (cid,cam),ids in need.items():
 have=set()
 for root in roots:
  f=root/cid/(cam+'_rows.npy')
  if f.exists():have.update(map(int,np.load(f)))
 miss=ids-have
 if miss:
  record=dict(cid=cid,cam=cam,rows=sorted(miss));missing.append(record)
  for root in extras:
   f=root/cid/(cam+'_rows.npy')
   if f.exists():
    hit=miss&set(map(int,np.load(f)))
    if hit:recover.append(dict(root=str(root),cid=cid,cam=cam,rows=sorted(hit)))
print('required',sum(map(len,need.values())),'missing',sum(len(x['rows']) for x in missing),'captures',len({x['cid'] for x in missing}),'recoverable',sum(len(x['rows']) for x in recover));print('roots',list(map(str,roots)));print('extra_contracts',[(str(p),json.loads((p/'contract.json').read_text()).get('preprocess')) for p in extras if (p/'contract.json').exists()]);(O/'CACHE_AUDIT.json').write_text(json.dumps(dict(missing=missing,recover=recover,roots=list(map(str,roots))),indent=2))
