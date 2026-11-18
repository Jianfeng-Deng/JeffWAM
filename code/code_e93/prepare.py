import os,sys,json,collections
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');sys.path.insert(0,str(T/'code_e90'))
import numpy as np,h5py
from e86.data import Package,build_window,CAMS
from e86.common import atomic,sha,brt
from e88.audit import HorizonIndex
R=T/'results/e93_dev_diag_20261114_01'; D=R/'dev';D.mkdir(exist_ok=False)
p=Package(); oldcaps=json.loads((T/'code_e86/artifacts/dev_v1/captures.json').read_text()); oldrows=json.loads((T/'code_e86/artifacts/dev_v1/windows_v2.json').read_text()); caps=list(oldcaps);rows=[dict(w,_stage='P1',source='piper') for w in oldrows];seen={c['parent_component'] for c in caps}; failures=[]
for c in p.captures:
 if c['split']!='dev' or c['parent_component'] in seen or c['canonical_id'] in p.excluded:continue
 route=p.routes.get(c['canonical_id'],{})
 if not route.get('all_available'):failures.append([c['canonical_id'],'route_unavailable']);continue
 before=Path(c['source_path']).stat()
 with h5py.File(c['source_path'],'r') as f:
  assert not f.attrs.get('sim',False)
  t=f['timestamp'][:];ct=f['camera_timestamps'][:];rc=f['camera_received_timestamps'][:];br=f['state_source_brackets'][:]
 n=len(t);avail=np.maximum(t,np.max(br,axis=(1,2)));seg=np.cumsum(np.r_[0,(np.diff(t)<=0)|(np.diff(t)>.05)]);valid=np.isfinite(br).all((1,2)) & ((br[:,:,1]-br[:,:,0]).max(1)<=.075)
 z=dict(timestamp=t,camera_exposure_timestamp=ct,camera_received_timestamp=rc,state_available_time=avail,continuous_segment=seg,original_action_index=np.arange(1,n),p1_source_mask=np.ones(n,bool),p3_source_mask=np.ones(n,bool))
 ws=[]
 for a in np.unique(np.linspace(24,n-53,16).astype(int)//4*4):
  if a<24 or a+50>=n or seg[a-24]!=seg[a+50] or not valid[a-24:a+51].all():continue
  oi=np.empty((2,3),np.int64);wi=np.empty((3,3),np.int64)
  for k in range(3):
   ac=np.maximum.accumulate(np.maximum(ct[:,k],rc[:,k]));oi[:,k]=np.searchsorted(ac,avail[[a-6,a]],side='right')-1
   for j,off in enumerate((10,25,50)):
    rr=np.arange(max(0,a+off-2),min(n,a+off+3));wi[j,k]=rr[np.argmin(abs(ct[rr,k]-t[a+off]))]
  if any(abs(ct[wi[j,k],k]-t[a+off])>1/60+1e-6 for j,off in enumerate((10,25,50)) for k in range(3)):continue
  try:w=build_window(c,z,int(a),oi,wi,'P1')
  except ValueError:continue
  ws.append(dict(w,_stage='P1',source='piper'))
  if len(ws)==8:break
 if not ws:failures.append([c['canonical_id'],'no_valid_fixed_quantile_window']);continue
 num=D/(c['canonical_id']+'_numeric.npz');np.savez_compressed(num,**z)
 assert before.st_mtime_ns==Path(c['source_path']).stat().st_mtime_ns
 caps.append(dict(c,numeric_cache=str(num)));rows+=ws;seen.add(c['parent_component'])
atomic(D/'captures.json',caps);atomic(D/'windows.json',rows)
ix=HorizonIndex(R/'features',caps);plan=[];coverage=[]
for c in caps:
 cid=c['canonical_id'];ws=[w for w in rows if w['canonical_id']==cid]; sets=[set() for _ in range(3)];cnt=collections.Counter();reasons=collections.Counter()
 for w in ws:
  for k in range(3):sets[k].update(r[k] for r in w['obs_rows'] if r[k]>=0)
  for h in (10,25,50,100):
   ids,why=ix.resolve(w,h,require_cached=False)
   if ids is None:reasons[f'{h}:{why}']+=1
   else:
    cnt[h]+=1
    for k in range(3):sets[k].add(ids[k])
 route=p.routes[cid];cams=[dict(camera=cam,path=next(x['path'] for x in route['cameras'] if x['camera']==cam),rows=sorted(sets[k])) for k,cam in enumerate(CAMS)]
 plan.append(dict(canonical_id=cid,parent_component=c['parent_component'],split='dev',cameras=cams))
 demo=next(x for x in ('demo3','demo4','demo6') if x in c['source_collection'])
 for w in ws:w['stratum']=demo
 coverage.append(dict(parent=c['parent_component'],canonical_id=cid,demo=demo,windows=len(ws),valid={str(h):cnt[h] for h in (10,25,50,100)},reasons=dict(reasons)))
frames=sum(len(cam['rows']) for c in plan for cam in c['cameras']);estimate=frames*(8*768*2+8)+10_000_000
assert estimate<=30_000_000_000
atomic(D/'windows.json',rows);atomic(R/'cache_plan.json',dict(captures=plan,unique_camera_frames=frames,estimated_bytes=estimate,refs=p.refs,selection='Preserve old 48 windows exactly. New parents: same fixed 16 quantile candidates, first 8 valid; independent of model outcomes. One canonical per frozen parent.'))
atomic(R/'coverage_pre_cache.json',dict(frozen_dev_parents=sum(v=='dev' for v in p.split.values()),selected_parents=len(seen),windows=len(rows),parents=coverage,failures=failures,estimated_bytes=estimate,camera_frames=frames))
with (T/'notes/e93_dev_diag_20261114.md').open('a') as f:f.write('\n### A 预估与固定窗口方案\n'+f'{brt()}：冻结 dev {sum(v=="dev" for v in p.split.values())} parent，选中 {len(seen)} parent/{len(rows)} 窗；新增特征上界 {estimate/1e9:.4f} GB、{frames} 相机帧。原48窗逐字保留帧映射，新parent沿用E86的固定分位候选前8个合法窗口，不按模型结果选窗；非穷举全部anchor。元数据仅时间与帧映射，不加载状态/动作作输入。\n')
print(json.dumps(dict(parents=len(seen),windows=len(rows),frames=frames,estimated_GB=estimate/1e9,failures=len(failures))))
