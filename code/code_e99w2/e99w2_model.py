import math,numpy as np,torch
import torch.nn.functional as F
from e98w_model import WorldModel as Base
from e88.model import directional_mask,horizon_embedding
from e86.model import mask_oa
from e98w_core import CHAIN

def pose_torch(q):
 lead=q.shape[:-1];q=q.reshape(-1,14);p=[]
 for start in (0,7):
  T=torch.eye(4,device=q.device,dtype=q.dtype)[None].expand(len(q),-1,-1);j=0
  for _,typ,To,axis in CHAIN:
   T=T@torch.as_tensor(To,device=q.device,dtype=q.dtype)[None]
   if typ=='revolute':
    ax=axis/np.linalg.norm(axis);K=torch.tensor([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]],device=q.device,dtype=q.dtype)
    ang=q[:,start+j,None,None];R=torch.eye(3,device=q.device,dtype=q.dtype)[None]+ang.sin()*K+(1-ang.cos())*(K@K);j+=1
    top=torch.cat([R,torch.zeros(len(q),3,1,device=q.device,dtype=q.dtype)],-1);bot=torch.tensor([0,0,0,1],device=q.device,dtype=q.dtype)[None,None].expand(len(q),-1,-1);T=T@torch.cat([top,bot],1)
  p.append(torch.cat([T[:,:3,3]+T[:,:3,2]*.1358,T[:,:3,0],T[:,:3,1]],-1))
 return torch.cat(p,-1).reshape(*lead,18)

def trajectory(context,action):
 q0=context[:,:14];p0=context[:,84:102];seq=torch.cat([q0[:,None],action,action[:,-1:].expand(-1,50,-1)],1)
 return torch.cat([context[:,:70,None].transpose(1,2).expand(-1,101,-1),seq-q0[:,None],p0[:,None].expand(-1,101,-1),pose_torch(seq)-p0[:,None],context[:,120:124,None].transpose(1,2).expand(-1,101,-1)],-1)

class WorldModel(Base):
 def __init__(self,norm):super().__init__(norm,'A')
 def forward(self,b,a=None,w=None,ta=None,tw=None,branch='mean'):
  o,ov,hv=self.observation(b);no=o.shape[1];parts=[o];valids=[ov];na=0;nw=0;nr=0
  if a is not None:
   state=b['q_normalized'].clone();sv=b['state_valid'].clone()
   if self.training:state=state+torch.randn_like(state)*self.c.state_noise;sv=sv & (torch.rand(len(o),device=o.device)>=self.c.state_drop)
   st=(self.state_proj(state*sv[:,None])[:,None]+self.kind[1])*sv[:,None,None]
   at=self.action_proj(a)+self.a_pos[None]+self.ta(torch.stack([ta,torch.sin(ta*math.pi)],1))[:,None]+self.kind[1]
   parts.extend([st,at]);valids.extend([sv[:,None],torch.ones(a.shape[:2],dtype=torch.bool,device=o.device)]);na=51
  if w is not None:
   hs=b['world_horizons'];assert torch.equal(hs,hs[0].expand_as(hs));nw=w.shape[1];he=self.horizon_proj(horizon_embedding(hs.to(o.dtype),self.c.width));pos=self.grid_pos[b['world_positions']]
   if a is not None and not b.get('fixed_condition',False):
    absolute=a*b['action_std']+b['action_mean']+b['q_anchor'][:,None];robot=trajectory(b['context'],absolute)
   else:robot=b['robot']
   rh=torch.arange(101,device=o.device);r=self.robot_proj((robot-self.robot_mean)/self.robot_scale)+self.horizon_proj(horizon_embedding(rh.to(o.dtype),self.c.width))[None]+self.kind[1]
   query=self.mean_query.expand(len(o),nw,-1)+he+pos[None]+self.kind[2];parts.extend([r,query]);valids.extend([b['robot_valid'],b['world_valid']]);nr=101
  x=torch.cat(parts,1);n=len(x[0]);m=torch.ones(n,n,dtype=torch.bool,device=o.device);m[:no,:no]=False
  if na:m[no:no+na,:no+na]=False
  if nw:
   rs=no+na;ws=rs+nr;m[rs:ws,rs:ws]=~torch.eye(nr,dtype=torch.bool,device=o.device)
   m[ws:,:no+na]=False;m[ws:,rs:ws]=rh[None,:]>hs[0,:,None];m[ws:,ws:]=directional_mask(no,0,hs[0])[no:,no:]
   if na:
    # Preserve v3.1 long-W isolation; short W and A are bidirectional.
    m[no:no+na,ws:]=False;m[ws:,:][hs[0]>50,no:no+na]=True
  invalid=~torch.cat(valids,1);fast=torch.backends.mha.get_fastpath_enabled()
  if not self.training:torch.backends.mha.set_fastpath_enabled(False)
  try:
   for block in self.blocks:x=block(x,m,invalid)
  finally:
   if not self.training:torch.backends.mha.set_fastpath_enabled(fast)
  x=self.norm(x);ah=x[:,no+1:no+51] if na else None;wh=x[:,-nw:] if nw else None
  def mean(z,v):return (z*v[:,:,None]).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
  image=mean(x[:,:24],ov[:,:24]);hist=mean(x[:,24:56],ov[:,24:56]);causal=mean(x[:,:no],ov);clean=causal[:,None]+he+pos[None] if nw else None;aux=clean.mean(1) if nw else causal*0
  reps=[image,hist,ah.mean(1) if na else image*0]
  return dict(mean=self.mean_head(wh) if nw else None,action=self.action_head(ah) if na else None,world=None,world_cos=self.world_cos_head(clean) if nw else None,observation=x[:,:no],phase=self.phase_head(aux),progress=self.progress_head(aux).sigmoid().squeeze(-1),align=[F.normalize(h(z),dim=-1) for h,z in zip(self.align_heads,reps)],history_kept=hv,alignment_available=torch.stack([b['current_valid'].any(1),hv.any(1),torch.full((len(o),),bool(na),device=o.device,dtype=torch.bool)],1))
 @torch.no_grad()
 def infer(self,b,mode='action_only',steps=4,seed=90):
  assert mode in ('action_only','joint');device=b['obs'].device;a=torch.randn(len(b['obs']),50,14,device=device,generator=torch.Generator(device=device).manual_seed(seed));mean=None
  for i in range(steps):
   t=torch.full((len(a),),i/steps,device=device);out=self(b,a,torch.empty_like(b['world']) if mode=='joint' else None,t,t);a=a+out['action']/steps
   assert torch.isfinite(a).all(),'Nonfinite sampled action'
  absolute=b['action_mean']+a*b['action_std']+b['q_anchor'][:,None]
  if mode=='joint':mean=self.forward_mean(dict(b,robot=trajectory(b['context'],absolute)))['mean']
  return absolute,mean
