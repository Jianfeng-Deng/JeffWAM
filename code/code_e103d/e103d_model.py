import torch
from e99w2_model import WorldModel as C2Model
from e88.model import horizon_embedding,directional_mask
class WorldModel(C2Model):
 def forward(self,b,a=None,w=None,ta=None,tw=None,branch='mean'):
  if not b.get('external_no_robot',False):return super().forward(b,a,w,ta,tw,branch)
  assert a is None and w is not None,'P1 must physically omit A/state/R tokens'
  o,ov,hv=self.observation(b);no=o.shape[1];hs=b['world_horizons'];assert torch.equal(hs,hs[0].expand_as(hs));nw=w.shape[1];he=self.horizon_proj(horizon_embedding(hs.to(o.dtype),self.c.width));pos=self.grid_pos[b['world_positions']];query=self.mean_query.expand(len(o),nw,-1)+he+pos[None]+self.kind[2]
  x=torch.cat([o,query],1);mask=torch.ones(x.shape[1],x.shape[1],dtype=torch.bool,device=x.device);mask[:no,:no]=False;mask[no:,:no]=False;mask[no:,no:]=directional_mask(no,0,hs[0])[no:,no:];invalid=~torch.cat([ov,b['world_valid']],1)
  fast=torch.backends.mha.get_fastpath_enabled()
  if not self.training:torch.backends.mha.set_fastpath_enabled(False)
  try:
   for block in self.blocks:x=block(x,mask,invalid)
  finally:
   if not self.training:torch.backends.mha.set_fastpath_enabled(fast)
  x=self.norm(x);causal=(x[:,:no]*ov[:,:,None]).sum(1)/ov.sum(1,keepdim=True).clamp_min(1);clean=causal[:,None]+he+pos[None]
  return dict(mean=self.mean_head(x[:,-nw:]),world_cos=self.world_cos_head(clean),observation=x[:,:no],action=None,history_kept=hv)
