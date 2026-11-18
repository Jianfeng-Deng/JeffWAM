from .common import *
import torch,numpy as np
from .data import load,Data
from .model import WorldModel
from e90.train import configure
configure();tr,t64,dev=load();m=WorldModel().eval();checks=[]
assert np.array_equal(tr.scale(51),tr.scale(100)) and np.array_equal(tr.scale(99),tr.scale(100))
for name,data in [('train',tr),('dev',dev)]:
 i=0;b=data.batch([i],H,'short','P1');o=data.base[i];assert torch.equal(b['obs'][0],o['obs']);assert not any(k in b for k in ('q_normalized','q_anchor','action'))
 assert b['world'].shape==(1,75,768) and b['target_obs'].shape==(1,5,3,8,768)
 for j,h in enumerate(H):
  ids,_=data.index.resolve(data.rows[i],h)
  if ids is None:continue
  row=data.rows[i];cid=row['canonical_id'];f=np.stack([data.target.get(cid,k,[ids[k]])[0] for k in range(3)]);c=b['target_obs'][0,4].numpy()
  if h>50:f=f.mean(1,keepdims=True);c=c.mean(1,keepdims=True)
  sl=slice(j*24,j*24+24) if h<=50 else slice(72,75)
  assert np.array_equal(f.reshape(-1,768),b['world_raw'][0,sl].numpy())
 with torch.no_grad():
  z=m.observation(b,False)[0];changed=dict(b,world=b['world']+99,world_raw=b['world_raw']+99,target_obs=b['target_obs']+99);assert torch.equal(z,m.observation(changed,False)[0]);nm,co=m.adapter_loss((b['target_obs'][:,4],b['obs'][:,4]));assert torch.isfinite(nm) and torch.isfinite(co)
 checks.append(dict(pool=name,observation_exact=True,no_action_state=True,target_read_exact=True,future_target_does_not_change_O=True))
# Fine reader dimensional contract, exact fixed E93 window selection and masks.
fine=Data(t64,49);b=fine.get(0,H,'short','P1');assert b['world'].shape==(444,768) #3*49*3 +3
assert len(t64.rows)==507 and len(dev.rows)==211
atomic(R/'implementation_checks.json',dict(passed=True,checks=checks,train64_windows=507,dev_windows=211,fine_world_shape=list(b['world'].shape)));report('## 代码与数据合同检查通过\nO逐元素等于原SigLIP2输入、无动作状态、DINO目标精确读取、改动未来目标不影响O；7×7短时域与+100全局维度正确，固定评估窗数不变。');print('implementation checks passed')
