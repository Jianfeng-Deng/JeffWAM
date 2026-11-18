from .common import *
import numpy as np
from .data import Features
from e90.data import load_data
from e88.common import ART as E88ART
p,ready,tr,_,_=load_data();old=json.loads((T/'code_e88/artifacts/cpu_audit_01/residual_stats.json').read_text());bykey={(r['canonical_id'],r['anchor']):r for r in tr.rows};scales={};scales7={};features={g:Features(g) for g in (8,49)}
assert old['fitted_split']=='train'
for h in H:
 samples=old['samples'][str(h)];parents=collections.defaultdict(lambda:collections.defaultdict(list))
 for cid,a,parent in samples:
  r=bykey[(cid,a)];assert r['parent_component']==parent and r['split']=='train';rr,_=tr.index.resolve(r,h);assert rr is not None
  for grid in (8,49):
   z=features[8 if h>50 else grid];cur=np.stack([z.get(cid,k,[r['obs_rows'][-1][k]])[0] for k in range(3)]);fut=np.stack([z.get(cid,k,[rr[k]])[0] for k in range(3)]);d=fut-cur
   if h>50:d=d.mean(1,keepdims=True)
   parents[grid][parent].append((d**2).mean(-1))
 for grid,out in ((8,scales),(49,scales7)):
  out[str(h)]=np.maximum(np.sqrt(np.mean([np.mean(v,0) for v in parents[grid].values()],0)),.05).tolist()
result=dict(scale=scales,scale7=scales7,floor=.05,statistic=old['statistic'],fitted_split='train',stage='P1',samples={str(h):old['samples'][str(h)] for h in H},refs=p.refs,same_E90_selection=True,original_stats_sha256=sha(T/'code_e88/artifacts/cpu_audit_01/residual_stats.json'))
assert not (CACHE/'residual_stats.json').exists();atomic(CACHE/'residual_stats.json',result)
# Append-only completion extension; original extraction contract remains immutable for receipts.
atomic(CACHE/'normalization_contract.json',dict(base_contract_sha256=sha(CACHE/'contract.json'),residual_stats_path=str(CACHE/'residual_stats.json'),residual_stats_sha256=sha(CACHE/'residual_stats.json'),fitted_split='train',refs=p.refs))
report('## 2 训练归一化统计完成\n严格复用E90每训练parent每时域最多2窗的样本清单，按parent等权、通道均方后RMS，floor=.05；7×7短时域同法，+100复用2×4全局。统计与normalization_contract均已落盘。');status('stats_complete');print('stats complete')
