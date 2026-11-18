from .common import *
import torch,copy
from e86.fixtures import batch
from e90.train import configure
from .model import WorldModel as FM
from .readout import WorldModel as Mean,loss as mean_loss,chain as mean_chain
from e90.objective import loss as fm_loss
configure();b=batch(2)
for k in ('action','q_normalized','q_anchor','action_mean','action_std','state_valid'):b.pop(k)
b.update(world=torch.randn(2,24,768),world_raw=torch.randn(2,24,768),world_current=torch.randn(2,24,768),world_scale=torch.ones(2,24,768),world_valid=torch.ones(2,24,dtype=torch.bool),world_horizons=torch.full((2,24),25),world_positions=torch.arange(24),target_obs=torch.randn(2,5,3,8,768),phase_valid=torch.zeros(2,dtype=torch.bool),phase_target=torch.full((2,),-1),progress_target=torch.zeros(2),source_id=torch.zeros(2,dtype=torch.long))
checks=[]
for cls,func,recipe in [(FM,fm_loss,None),(Mean,mean_loss,'M1'),(Mean,mean_loss,'M2')]:
 m=cls() if recipe is None else cls(recipe=recipe);m.eval();val,metrics=func(m,b,'P1',1,seed=90,alignment_weight=0.);assert torch.isfinite(val);val.backward();assert any(x.grad is not None for x in m.reentry.parameters())
 target=m.current(b['obs'][:,4]).detach();raw=b['target_obs'][:,4];pred=m.reentry(raw);expected=(pred-target).square().mean()/target.square().mean().clamp_min(1e-6);actual,_=m.adapter_loss((raw,b['obs'][:,4]));assert torch.equal(actual,expected)
 if recipe:
  with torch.no_grad():
   p=m.forward_mean(b)['mean'];c=dict(b,world=b['world']+100,world_raw=b['world_raw']-100,target_obs=b['target_obs']+100);assert torch.equal(p,m.forward_mean(c)['mean'])
  shifted=copy.copy(b);direct=copy.copy(b);seen=[];hook=m.reentry.register_forward_pre_hook(lambda module,args:seen.append(args[0]));cl,_=mean_chain(m,b,shifted,direct,25,use_predicted=False);hook.remove();assert any(torch.equal(x,shifted['target_obs'][:,4]) for x in seen);assert torch.isfinite(cl)
 checks.append(dict(recipe=recipe or 'original_FM',finite_loss=True,adapter_cross_encoder_exact=True,teacher_reentry_DINO=True if recipe else None,target_not_used_by_mean=True if recipe else None))
atomic(R/'cross_encoder_checks.json',dict(passed=True,checks=checks,GPUh=0,brt=brt()));report('## 跨编码器接口CPU检查通过\n'+json.dumps(checks,ensure_ascii=False));print('cross encoder checks passed')
