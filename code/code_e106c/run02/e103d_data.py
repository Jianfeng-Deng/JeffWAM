import json,collections,hashlib
import numpy as np,torch
from e103d_runtime import *
from e86.data import collate
from e88.data import coarse
class External:
 def __init__(self,piper):
  self.plan=json.loads((CACHE/'plan.json').read_text());self.norm=piper.norm;self.piper=piper;self.rows=[];self.maps={};self.records={r['canonical_id']:r for r in self.plan['records']};self.pools=collections.defaultdict(lambda:collections.defaultdict(list))
  for rec in self.plan['records']:
   for w in rec['windows']:
    i=len(self.rows);self.rows.append(dict(w,canonical_id=rec['canonical_id'],source=rec['source'],task=rec['task']));self.pools[rec['source']][rec['task']].append(i)
  self.text=np.load(CACHE/'generic_text.npy').astype('f4')
  self.parent_pools={}
  for source,tasks in self.pools.items():
   table={}
   for task,ids in tasks.items():
    parents=collections.defaultdict(list)
    for i in ids:parents[self.rows[i]['canonical_id']].append(i)
    table[task]=dict(parents)
   self.parent_pools[source]=table
 def feature(self,cid,k,rows):
  key=(cid,k)
  if key not in self.maps:
   cam=['cam_high','cam_left_wrist','cam_right_wrist'][k];d=CACHE/'features'/cid;self.maps[key]=(np.load(d/(cam+'_rows.npy')),np.load(d/(cam+'.npy'),mmap_mode='r'))
  ids,arr=self.maps[key];ix=np.searchsorted(ids,rows);assert (ix<len(ids)).all() and np.array_equal(ids[ix],rows);return arr[ix].astype('f4')
 def get(self,i,horizons=(10,25,50,100)):
  r=self.rows[i];cid=r['canonical_id'];obs=np.stack([self.feature(cid,k,np.array(r['obs_rows'])[:,k]) for k in range(3)],axis=1);future=[];current=[];scales=[];hs=[];pos=[]
  for h in horizons:
   z=np.stack([self.feature(cid,k,[r['target_rows'][str(h)][k]])[0] for k in range(3)]);z=coarse(z,h);cur=coarse(obs[4],h);n=z.shape[1];future.append(z.reshape(-1,768));current.append(cur.reshape(-1,768));scales.append(np.broadcast_to(self.piper.scale(h)[:,:,None],z.shape).reshape(-1,768));hs.extend([h]*(3*n));pos.extend([k*8+j for k in range(3) for j in (range(8) if n==8 else [0])])
  f=np.concatenate(future);c=np.concatenate(current);s=np.concatenate(scales)
  return dict(obs=torch.tensor(obs),obs_meta=torch.tensor(r['obs_meta'],dtype=torch.float32),history_valid=torch.ones(4,dtype=torch.bool),current_valid=torch.ones(3,dtype=torch.bool),text=torch.tensor(self.text),text_valid=torch.ones(len(self.text),dtype=torch.bool),source_id=torch.tensor(['piper','yam','abc'].index(r['source'])),world=torch.tensor((f-c)/s),world_raw=torch.tensor(f),world_current=torch.tensor(c),world_scale=torch.tensor(s),world_valid=torch.ones(len(hs),dtype=torch.bool),world_horizons=torch.tensor(hs),world_positions=torch.tensor(pos))
 def sample(self,source,n,step):
  rng=np.random.default_rng(103000+step);tasks=sorted(self.parent_pools[source]);result=[]
  for j in range(n):
   task=tasks[(step*n+j)%len(tasks)];parents=self.parent_pools[source][task];cs=sorted(parents);cid=cs[int(rng.integers(len(cs)))];ids=parents[cid];result.append(ids[int(rng.integers(len(ids)))])
  return result
 def batch(self,ids):
  b=collate([self.get(i) for i in ids]);b['world_positions']=b['world_positions'][0];b['external_no_robot']=True;return b

def p1_only(b):
 keys=['obs','obs_meta','history_valid','current_valid','text','text_valid','source_id','world','world_raw','world_current','world_scale','world_valid','world_horizons','world_positions']
 return dict({k:b[k] for k in keys if k in b},external_no_robot=True)
