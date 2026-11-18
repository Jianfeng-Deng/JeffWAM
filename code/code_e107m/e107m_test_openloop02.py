from e107m_eval03 import *
from e96a.train import configure
configure();torch.backends.mha.set_fastpath_enabled(False);_,_,tr,dev=datasets();lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());i=lock['entries'][0]['index'];results=[]
for g in ('BASE','M1'):
 cp=torch.load(Path(lock['checkpoints'][g]),map_location='cpu',weights_only=False);m=(StochasticWorld if g=='M1' else WorldModel)(tr.norm);m.load_state_dict(cp['model']);m.eval().requires_grad_(False);del cp
 with torch.no_grad():
  for k in (25,50):
   first=dev.batch([i],(k,),'short');first['plan_action']=first['plan_action'].double();z=m.mean_rollout(first);second=causal_second(m,first,k,z);assert second['robot'].dtype==next(m.parameters()).dtype
   out=m.mean_rollout(second);assert torch.isfinite(out).all();results.append(dict(group=g,k=k,original_plan_dtype=str(first['plan_action'].dtype),second_robot_dtype=str(second['robot'].dtype),finite=True))
assert not torch.cuda.is_initialized();atomic(CONTROL/'OPENLOOP_DTYPE_TEST02.json',dict(status='passed',results=results,CUDA_initialized=False,brt=brt()));print('CPU open-loop dtype/finite tests passed',results)
