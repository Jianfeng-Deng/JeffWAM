import torch
import torch.nn.functional as F
from e90.model import WorldModel as Base
class WorldModel(Base):
 def adapter_loss(self,pair):
  raw,sig=pair
  target=self.current(sig.detach()).detach();pred=self.reentry(raw.detach())
  return (pred-target).square().mean()/target.square().mean().clamp_min(1e-6),F.cosine_similarity(pred,target,dim=-1).mean()
