from .common import *
import numpy as np,torch,copy
from e90.data import Data as BaseData,ImageDataset,load_data as original_load
from e86.data import CAMS
class Features:
 def __init__(self,grid=8):self.grid=grid;self.memo={}
 def get(self,cid,cam,ids):
  key=(cid,cam)
  if key not in self.memo:
   stem=CAMS[cam]+('_7x7' if self.grid==49 else '');p=CACHE/cid
   self.memo[key]=(np.load(p/(stem+'_rows.npy')),np.load(p/(stem+'.npy'),mmap_mode='r'))
  rows,z=self.memo[key];ids=np.asarray(ids);j=np.searchsorted(rows,ids);ok=(j<len(rows)) & (rows[np.minimum(j,len(rows)-1)]==ids);assert (ok|(ids<0)).all(),(cid,cam,ids[~(ok|(ids<0))].tolist())
  out=np.zeros((len(ids),self.grid,768),np.float32);out[ok]=z[j[ok]];return out
class Data(BaseData):
 def __init__(self,original,grid=8):
  self.__dict__.update(original.__dict__);self.target=Features(grid);self.grid=grid
  self.norm=json.loads((CACHE/'residual_stats.json').read_text())['scale' if grid==8 else 'scale7']
 def get(self,i,horizons=H,language='instruction',stage='P1'):
  # Old reader establishes identical O/text and all validity masks from immutable SigLIP2 cache.
  b=self.base[i];r=self.rows[i];cid=r['canonical_id']
  if language=='short':
   b['text']=torch.tensor(self.base.text_cache['__human_default__']);b['text_valid']=torch.ones(len(b['text']),dtype=torch.bool)
  elif language=='none':b['text_valid']=torch.zeros_like(b['text_valid'])
  b.update(source_id=torch.tensor(0),phase_target=torch.tensor(-1),progress_target=torch.tensor(0.),phase_valid=torch.tensor(False))
  targetobs=np.stack([self.target.get(cid,k,np.array(r['obs_rows'])[:,k]) for k in range(3)],axis=1)
  cur=targetobs[4];fs=[];cs=[];sc=[];valid=[];hs=[];pos=[]
  for h in horizons:
   rr,_=self.index.resolve(dict(r,_stage='P1'),h);ok=rr is not None
   if h>50:
    # Global must match the canonical 2x4 contract even in 7x7 diagnostics.
    canonical=self.target if self.grid==8 else Features(8)
    current=np.stack([canonical.get(cid,k,[r['obs_rows'][4][k]])[0] for k in range(3)]).mean(1,keepdims=True)
    future=np.stack([canonical.get(cid,k,[rr[k]])[0] for k in range(3)]).mean(1,keepdims=True) if ok else np.zeros((3,1,768),np.float32)
   else:
    current=cur;future=np.stack([self.target.get(cid,k,[rr[k]])[0] for k in range(3)]) if ok else np.zeros_like(cur)
   n=current.shape[1];scale=np.broadcast_to(self.scale(h)[:,:,None],current.shape)
   fs.append(future.reshape(-1,768));cs.append(current.reshape(-1,768));sc.append(scale.reshape(-1,768));valid.extend([ok]*(3*n));hs.extend([h]*(3*n));pos.extend([k*8+j for k in range(3) for j in (range(8) if n==8 else [0]*n)])
  f=np.concatenate(fs);c=np.concatenate(cs);s=np.concatenate(sc)
  b.update(world=torch.tensor((f-c)/s),world_raw=torch.tensor(f),world_current=torch.tensor(c),world_scale=torch.tensor(s),world_valid=torch.tensor(valid),world_horizons=torch.tensor(hs),world_positions=torch.tensor(pos),target_obs=torch.tensor(targetobs))
  return b
 def scale(self,h):
  if str(h) in self.norm:return np.asarray(self.norm[str(h)],np.float32)
  keys=[x for x in H if (x<=50)==(h<=50)]
  if h<min(keys):return np.asarray(self.norm[str(min(keys))],np.float32)
  a=max((x for x in keys if x<=h),default=min(keys));bb=min((x for x in keys if x>=h),default=max(keys))
  if a==bb:return np.asarray(self.norm[str(a)],np.float32)
  t=(h-a)/(bb-a)
  return np.asarray(self.norm[str(a)],np.float32)*(1-t)+np.asarray(self.norm[str(bb)],np.float32)*t

def load(grid=8):
 p,ready,tr,old,cov=original_load();caps=json.loads((E93/'dev/captures_full.json').read_text());dr=json.loads((E93/'dev/windows_full.json').read_text())
 dev=BaseData(dr,caps,E93/'features',tr.base.stats,{'scale':tr.norm},{'__human_default__':old.base.text_cache['__human_default__']})
 rows=json.loads((R/'train64_rows.json').read_text());t64=BaseData(rows,p.captures,tr.base.cache,tr.base.stats,{'scale':tr.norm},{'__human_default__':old.base.text_cache['__human_default__']})
 return Data(tr,grid),Data(t64,grid),Data(dev,grid)
