import sys,traceback,json
sys.dont_write_bytecode=True
from e103d_runtime import *
try:
 import torch,numpy as np
 from e103d_train import *
 base.configure();_,_,tr,_=load();_,_,p1,_,_=load_data();ext=External(p1);m=WorldModel(tr.norm);m.load_state_dict(torch.load(INIT,map_location='cpu',weights_only=False)['model'],strict=True);m.eval();b=tr.batch([0,100],H,'short');pure=p1_only(b)
 with torch.no_grad():
  a=m.forward_mean(dict(b,robot=torch.zeros_like(b['robot']),robot_valid=torch.zeros_like(b['robot_valid'])))['mean'];z=m.forward_mean(pure)['mean'];diff=float(abs(a-z).max());assert diff<2e-5,diff
 def forbidden(*a):raise AssertionError('robot/action/state projection in P1')
 hooks=[x.register_forward_hook(forbidden) for x in [m.robot_proj,m.state_proj,m.action_proj]]
 items=[]
 for i in [0,100]:
  x=p1_only(p1.get(i,H,'instruction','P1'));x.pop('external_no_robot');items.append(x)
 items.append(ext.get(0));pb=collate(items);pb['world_positions']=pb['world_positions'][0];pb['external_no_robot']=True;m.train();v,mt=p1_loss(m,pb);v.backward();assert m.action_head.weight.grad is None and m.robot_proj.weight.grad is None;assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());assert not torch.cuda.is_initialized()
 for h in hooks:h.remove()
 d=dict(status='passed',P3_rows=len(tr.rows),P1_rows=len(p1.rows),P1_parents=len({r['parent_component'] for r in p1.rows}),no_traj_deletion_max_abs=diff,P1_mixed_Piper_ABC_forward_backward_finite=True,no_action_gradient=True,CUDA_initialized=False,loss=float(v.detach()),brt=brt());atomic(CONTROL/'CPU_PREFLIGHT.json',d);report('## 初步 CPU 核验（真实 Piper+ABC，yam 待缓存齐备再检）\n'+json.dumps(d,ensure_ascii=False));print(d,flush=True)
except BaseException:atomic(CONTROL/'CPU_PREFLIGHT_ERROR.json',dict(error=traceback.format_exc(),brt=brt()));raise
