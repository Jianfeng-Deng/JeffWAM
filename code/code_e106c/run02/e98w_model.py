import torch,numpy as np
from torch import nn
from e96a.model import WorldModel as Base
from e88.model import directional_mask,horizon_embedding
from e98w_data import OUT
class WorldModel(Base):
 def __init__(self,norm,variant='A',**kw):
  super().__init__(**kw);assert variant in ('A','B');self.variant=variant
  with torch.random.fork_rng(devices=[]):self.robot_proj=nn.Linear(124,self.c.width)
  for h in (10,25,50,100):
   z=dict(np.load(OUT/f'crossfit_h{h}.npz'))
   for k in ('w1','ym','zm','zs'):
    self.register_buffer(f's{h}_{k}',torch.tensor(np.stack([z[f'f{i}_{k}'] for i in range(5)]+[z[f'full_{k}']]),dtype=torch.float32))
   self.register_buffer(f'scale{h}',torch.tensor(np.asarray(norm[str(h)]),dtype=torch.float32).reshape(-1,1))
  hs=np.arange(101);nearest=np.array([min((10,25,50,100),key=lambda h:abs(h-int(x))) for x in hs]);means=[];scales=[]
  for h in nearest:means.append(getattr(self,f's{h}_zm')[5]);scales.append(getattr(self,f's{h}_zs')[5])
  self.register_buffer('robot_mean',torch.stack(means));self.register_buffer('robot_scale',torch.stack(scales))
 def robot_prediction(self,b):
  if 'robot_s' in b:return b['robot_s']
  result=torch.zeros_like(b['world']);fold=b['robot_fold'];hs=b['world_horizons'][0]
  for hh in hs.unique():
   h=int(hh);sel=hs==h;nt=24 if h<=50 else 3
   if h in (10,25,50,100):terms=[(h,1.)]
   elif h>50:terms=[(100,1.)]
   else:
    lo=max(x for x in (10,25,50) if x<h);hi=min(x for x in (10,25,50) if x>h);t=(h-lo)/(hi-lo);terms=[(lo,1-t),(hi,t)]
   for fixed,weight in terms:
    for fi in fold.unique():
     ix=torch.where(fold==fi)[0];j=int(fi);x=b['robot'][ix,h];mean=getattr(self,f's{fixed}_zm')[j];scale=getattr(self,f's{fixed}_zs')[j]
     pred=torch.einsum('br,krd->bkd',(x-mean)/scale,getattr(self,f's{fixed}_w1')[j])+getattr(self,f's{fixed}_ym')[j]
     raw=pred*getattr(self,f'scale{fixed}')[None]
     dest=torch.where(sel)[0];result[ix[:,None],dest[None,:]]+=weight*raw/b['world_scale'][ix[:,None],dest[None,:]]
  b['robot_s']=result;return result
 def mean_rollout(self,b,steps=4,seed=90):
  delta=self.forward_mean(b)['mean']
  if self.variant=='B':delta=delta+self.robot_prediction(b)
  return b['world_current']+delta*b['world_scale']
 def forward(self,b,a=None,w=None,ta=None,tw=None,branch='mean'):
  assert a is None and branch=='mean' and w is not None
  o,ov,hv=self.observation(b);no=o.shape[1];hs=b['world_horizons'];assert torch.equal(hs,hs[0].expand_as(hs));nw=w.shape[1]
  he=self.horizon_proj(horizon_embedding(hs.to(o.dtype),self.c.width));pos=self.grid_pos[b['world_positions']]
  rh=torch.arange(101,device=o.device);r=self.robot_proj((b['robot']-self.robot_mean)/self.robot_scale)+self.horizon_proj(horizon_embedding(rh.to(o.dtype),self.c.width))[None]+self.kind[1]
  query=self.mean_query.expand(len(o),nw,-1)+he+pos[None]+self.kind[2];nr=r.shape[1];start=no+nr
  x=torch.cat([o,r,query],1);m=torch.ones(x.shape[1],x.shape[1],dtype=torch.bool,device=x.device)
  m[:no,:no]=False;m[no:start,no:start]=~torch.eye(nr,dtype=torch.bool,device=x.device)
  m[start:,:no]=False;m[start:,no:start]=rh[None,:]>hs[0,:,None]
  m[start:,start:]=directional_mask(no,0,hs[0])[no:,no:]
  invalid=~torch.cat([ov,b['robot_valid'],b['world_valid']],1)
  # Native eval MHA emits NaNs on fully masked invalid R queries.
  # Use the training SDPA path for eval blocks; preserve the training path exactly.
  fast=torch.backends.mha.get_fastpath_enabled()
  if not self.training:torch.backends.mha.set_fastpath_enabled(False)
  try:
   for block in self.blocks:x=block(x,m,invalid)
  finally:
   if not self.training:torch.backends.mha.set_fastpath_enabled(fast)
  x=self.norm(x);causal=(x[:,:no]*ov[:,:,None]).sum(1)/ov.sum(1,keepdim=True).clamp_min(1)
  clean=causal[:,None]+he+pos[None]
  return dict(mean=self.mean_head(x[:,-nw:]),world_cos=self.world_cos_head(clean),observation=x[:,:no],history_kept=hv)
