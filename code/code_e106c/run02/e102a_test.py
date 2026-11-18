import json,torch,numpy as np,importlib
from e99w2_common import *
from e99w2_data import load,pose_np
from e99w2_model import WorldModel,trajectory,pose_torch
from e99w2_objective import loss
from e102a_contract import causal_features,contract
from e100c_chain import corrected_chain,open_second
from e96a.train import configure
configure();_,_,tr,dev=load();m=WorldModel(tr.norm);cp=torch.load(INIT,map_location='cpu',weights_only=False);missing=m.load_state_dict(cp['model'],strict=False);assert not missing.unexpected_keys;assert all(k.startswith(('grip_head.','proprio_proj.','delta_head.')) for k in missing.missing_keys);assert cp['step']==5000;del cp
b=tr.batch([0,100],(10,25,50,100),'short');m.eval()
assert b['proprio'].shape==(2,164);assert b['grip_label'].shape==(2,50,2);assert set(b['grip_label'].unique().tolist())<={0.,1.}
assert float((pose_torch(b['action'])-torch.tensor(pose_np(b['action'].numpy()))).abs().max())<2e-6
assert torch.allclose(trajectory(b['context'],b['plan_action']),b['robot'],atol=2e-6)
with torch.no_grad():
 a,w=m.infer(b);changed=dict(b,action=b['action']+999,robot=b['robot']+999,plan_action=b['plan_action']+999,world=b['world']+999);a2,_=m.infer(changed);assert torch.equal(a,a2)
 o=m.forward_mean(b)['mean'];o2=m.forward_mean(dict(b,world=b['world']+999))['mean'];assert torch.equal(o,o2)
 assert torch.isfinite(m.forward_mean(dict(b,robot_valid=torch.zeros_like(b['robot_valid'])))['mean']).all()
 # Predicted second hop has no real future observation; plan copied from original suffix only.
 second=open_second(b,25);assert not second['history_valid'].any();assert not second['state_valid'].any();assert torch.equal(second['action'][:,:25],b['plan_action'][:,25:])
 # Causal feature invariance to all feedback strictly after anchor.
 r=tr.rows[100];z,q,p,g=tr.states.get(tr,r['canonical_id']);anchor=r['anchor'];v=causal_features(q,p,anchor,r['history_valid'],z['continuous_segment'],g,contract());q=q.copy();p=p.copy();q[anchor+1:]+=10;p[anchor+1:]+=20;v2=causal_features(q,p,anchor,r['history_valid'],z['continuous_segment'],g,contract());assert np.array_equal(v,v2)
from e102a_eval import match
assert match([(2,1),(8,1)],[(4,1),(10,1)],5)==[-2,-2]
assert match([(2,0)],[(2,1)],10)==[]
assert match([],[(2,1)],10)==[]
# Regression: teacher shift near episode end consumes original suffix+hold labels only.
end_tests=[]
for i,r in enumerate(tr.rows):
 z,q,p,g=tr.states.get(tr,r['canonical_id']);a=r['anchor']
 if a+51<=len(q) and a+75>len(q) and tr.index.resolve(r,50)[0] is not None:
  shifted=tr.shifted(i,25)
  if shifted is None:continue
  orig=tr.get(i,(25,),'short');expected=torch.cat([orig['plan_action'][25:],orig['plan_action'][-1:].expand(25,-1)],0)
  from e99w2_data import planned_np
  assert torch.allclose(shifted['robot'],torch.tensor(planned_np(shifted['context'].numpy(),expected.numpy())),atol=2e-6)
  assert torch.equal(shifted['plan_action'],expected);assert torch.equal(shifted['grip_label'],torch.cat([orig['grip_label'][25:],orig['grip_label'][-1:].expand(25,-1)],0));assert shifted['grip_label'].shape==(50,2)
  end_tests.append(i)
  if len(end_tests)==2:break
assert len(end_tests)==2
seen={};m.train()
for step in range(1,80):
 from e86.objective import mode_for
 mode=mode_for(step,90,(.25,)*4)
 if mode in seen:continue
 m.zero_grad(set_to_none=True);v,mt=loss(m,b,step);v.backward();assert torch.isfinite(v) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
 if GROUP in ('A1','A2'):assert m.grip_head.weight.grad.abs().sum()>0
 if GROUP=='A2':assert m.delta_head.weight.grad.abs().sum()>0
 if mode!='action_only':assert m.mean_head[-1].weight.grad.abs().sum()>0
 seen[mode]=float(v.detach())
 if len(seen)==4:break
assert len(seen)==4
for name in ('e99w2_model','e99w2_objective','e99w2_train','e99w2_data','e100c_resource','e100c_chain'):
 mod=importlib.import_module(name);assert Path(mod.__file__).resolve().parent==ROOT,(name,mod.__file__)
assert not torch.cuda.is_initialized()
out=dict(status='passed',group=GROUP,modes=seen,missing_keys=missing.missing_keys,init_sha=sha(INIT),tail_shift_tests=end_tests,CPU_only=True,target_invariant=True,causal_future_invariant=True,source_hash=source_hash(),brt=brt());atomic(CONTROL/f'CPU_TEST_{GROUP}_03.json',out);print(json.dumps(out))
