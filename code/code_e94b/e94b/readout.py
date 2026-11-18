"""Import E94a v1 verbatim. Only cross-encoder adapter wiring is E94b-owned."""
from .common import *
import torch
import torch.nn.functional as F
from e94a.model import WorldModel as Unified
from e94a.objective import loss as unified_loss
from e90.objective import chain_loss as cross_encoder_chain
class WorldModel(Unified):
 def adapter_loss(self,pair):
  if isinstance(pair,tuple):raw,sig=pair
  else:
   if not hasattr(self,'_adapter_target'):raise RuntimeError('DINO/Sig adapter context missing')
   raw,sig=self._adapter_target,pair
  target=self.current(sig.detach()).detach();pred=self.reentry(raw.detach())
  return (pred-target).square().mean()/target.square().mean().clamp_min(1e-6),F.cosine_similarity(pred,target,dim=-1).mean()
def recipe():return json.loads((C/'E94b_readout_import_reviewed.json').read_text())['recipe']
def make_model():return WorldModel(recipe=recipe(),readout_hidden=1024)
def predict_mean(m,b):return m.mean_rollout(b)
def loss(model,b,*args,**kwargs):
 model._adapter_target=b['target_obs'][:,4]
 try:return unified_loss(model,b,*args,**kwargs)
 finally:del model._adapter_target
class MeanProxy:
 def __init__(self,m):self.m=m
 def __getattr__(self,name):return getattr(self.m,name)
 def rollout(self,*args,**kwargs):return self.m.mean_rollout(*args,**kwargs)
def chain(model,*args,**kwargs):return cross_encoder_chain(MeanProxy(model),*args,**kwargs)
def configure_readout(original,cfg):
 marker=json.loads((C/'E94b_readout_import_reviewed.json').read_text());rd=marker['recipe']
 for relative,expected in marker['copied_source_sha256'].items():assert sha(C/relative)==expected
 original.WorldModel=lambda:WorldModel(recipe=rd,readout_hidden=1024);original.loss=loss;original.chain_loss=chain;cfg.update(readout_recipe=rd,readout_hidden=1024,E94a_READY_sha256=marker['ready_sha256'],readout_source_sha256=marker['copied_source_sha256'])
