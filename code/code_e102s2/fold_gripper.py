from common import *
import numpy as np,h5py
ss=json.loads((T/'results/e99b3_20261115_01/sources.json').read_text());parents=sorted(s['parent'] for s in ss if s['pool']=='train');perm=np.random.default_rng(10101).permutation(parents);fold={p:i%5 for i,p in enumerate(perm)};raw={}
for s in ss:
 if s['pool']=='train':
  with h5py.File(s['source'],'r') as h:raw[s['parent']]=h['observations/qpos_raw'][:s['n']][::30][:,[6,13]]
thresholds=[]
for k in range(5):
 vals=np.concatenate([v for p,v in raw.items() if fold[p]!=k]);th=[]
 for j in range(2):
  v=vals[:,j];a,b=np.quantile(v,[.1,.9])
  for _ in range(30):
   c=(a+b)/2;lo=v[v<=c];hi=v[v>c]
   if not len(lo) or not len(hi):break
   a,b=lo.mean(),hi.mean()
  th.append(float((a+b)/2))
 thresholds.append(th)
for s in ss:
 with h5py.File(s['source'],'r') as h:
  times=h['timestamp'][:s['n']];gr=h['observations/qpos_raw'][:s['n']][:,[6,13]];br=h['state_source_brackets'][:s['n']] if 'state_source_brackets' in h else None
 av=np.maximum.accumulate(np.maximum(times,br[:,:,1].max(1))) if br is not None else times;ix=np.minimum(np.arange(s['n']),np.searchsorted(av,times,side='right')-1);valid=np.load(R/'features'/s['parent']/'state.npy',mmap_mode='r')[:,-1]>0;out=np.zeros((5,s['n'],2),'u1')
 for k in range(5):out[k,valid]=gr[ix[valid]]>np.array(thresholds[k])
 path=R/'features'/s['parent']/'gripper_fold.npy';assert not path.exists();np.save(path,out)
put('GRIPPER_FOLD_LOCK.json',dict(brt=now(),thresholds_by_excluded_fold=thresholds,parent_to_fold=fold,method='same two-cluster raw-state mapping, each fold fits only other48 parents; no labels; all60 mapping for final dev'))
report('## 两态夹爪五折映射锁定\n本体state.npy的全60训练阈值仅用于最终全量头；H3五折训练和袋外评分改用各折其余48 parent原始状态拟合的独立二态映射，避免预处理跨折。没有读取dev标签或以标签定夹爪阈值。')
print('fold gripper COMPLETE',thresholds)
