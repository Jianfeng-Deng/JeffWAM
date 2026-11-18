import os,json,inspect,hashlib,random
import torch,numpy as np
from e103d_runtime import *
from e99w2_common import INIT
from e99w2_data import load
from e90.data import load_data
from e103d_model import WorldModel
from e103d_data import External,p1_only
from e104p_train import p1_loss
from e102x_setup import datasets
from e96a.train import configure

def main():
 configure();_,ready,tr,dev=load();_,_,p1,_,_=load_data();ext=External(p1);m=WorldModel(tr.norm);cp=torch.load(INIT,map_location='cpu',weights_only=False);m.load_state_dict(cp['model'],strict=True)
 assert cp['step']==2000;assert 'optimizer' in cp
 opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(cp['optimizer']);assert all(pg['lr']==3e-4 for pg in opt.param_groups)
 initial_steps=sorted({int(v['step']) for v in opt.state.values() if 'step' in v})
 frozen=[n for n,p in m.named_parameters() if n.startswith(('action_head.','action_proj.','state_proj.','robot_proj.','proprio_proj.','delta_head.','grip_head.'))]
 before={n:p.detach().clone() for n,p in m.named_parameters() if n in frozen};before_opt={n:{k:v.clone() if torch.is_tensor(v) else v for k,v in opt.state[p].items()} for n,p in m.named_parameters() if n in frozen}
 def forbidden(*args):raise AssertionError('robot/action projection invoked by external')
 hooks=[getattr(m,n).register_forward_hook(forbidden) for n in ['robot_proj','state_proj','action_proj','proprio_proj','action_head','delta_head','grip_head']]
 for n,p in m.named_parameters():
  if n in frozen:p.requires_grad_(False)
 for source in ['abc','yam']:
  m.train();opt.zero_grad(set_to_none=True);b=ext.batch(ext.sample(source,2,1));v=p1_loss(m,b);assert torch.isfinite(v);v.backward();assert all(p.grad is None for n,p in m.named_parameters() if n in frozen);opt.step()
 for h in hooks:h.remove()
 for n,p in m.named_parameters():
  if n in frozen:
   assert torch.equal(p,before[n]);assert all(torch.equal(v,before_opt[n][k]) if torch.is_tensor(v) else v==before_opt[n][k] for k,v in opt.state[p].items());p.requires_grad_(True)
 m.eval();b=tr.batch([0],(10,25,50,100),'short');no=dict(b,robot=torch.zeros_like(b['robot']),robot_valid=torch.zeros_like(b['robot_valid']))
 with torch.no_grad():diff=float(abs(m.forward_mean(no)['mean']-m.forward_mean(p1_only(b))['mean']).max())
 assert diff<2e-5,diff
 _,_,_,expanded=datasets();old=json.loads((T/'results/e102x_extended_s90_20261115_01/BEFORE_FORWARD_LOCK.json').read_text());entries=old['entries'];assert len(entries)==211 and sum(x['original48'] for x in entries)==48
 for d in (p1,tr,dev,expanded):assert all(r['canonical_id']!='piper_4dc107335389f5711a0f65e0' for r in d.rows)
 assert not {r['parent_component'] for r in p1.rows}&{r['parent_component'] for r in expanded.rows}
 paths=[INIT,CACHE/'plan.json',CACHE/'generic_text.npy',CONTROL/'GRIPPER_CONTRACT.json',T/'results/e102x_extended_s90_20261115_01/BEFORE_FORWARD_LOCK.json',T/'notes/e103d_external_data_20261115.md',T/'notes/e102a_action_fix_20261115.md',T/'notes/e102x_extended_eval_20261115.md']+[Path(v) for v in ready['windows'].values()]
 hashes={str(p):sha(p) for p in paths};atomic(CONTROL/'INPUT_HASHES.json',hashes)
 out=dict(status='passed',initial_optimizer_steps=initial_steps,external_frozen_model_and_optimizer_unchanged=True,no_robot_action_projection=True,no_traj_equivalence_max_abs=diff,P3_pool=len(tr.rows),P1_pool=len(p1.rows),external_windows=len(ext.rows),expanded_windows=len(entries),expanded_parents=27,original48=48,CUDA_initialized=torch.cuda.is_initialized(),brt=brt());assert not out['CUDA_initialized'];atomic(CONTROL/'CPU_TESTS.json',out)
 lock=dict(brt=brt(),initial_checkpoint=str(INIT),initial_sha=sha(INIT),plan_sha=sha(CACHE/'plan.json'),seed=90,lr=3e-4,optimizer='restore A2 AdamW; never reset including external->Piper; frozen grads=None skip momentum and decay',F2='Piper10000; save5000=F0 and10000=F2',F1='external5000 then Piper5000; save external end=F1-pre',external='batch32 ABC16 yam16; task round robin, parent equal; fixed generic E103-D instruction; no_traj physically deletes all robot/state/A tokens; world MSE+.1cos+.1adapter; no action/grip/chain loss; action/robot/state/proprio/grip/delta projections frozen',adapter_policy='No extra calibration optimizer or updates; check train-heldout thresholds only; continuous restored AdamW',Piper='same A2 P3 batch32 + independent Piper P1 batch32 weight.1 from E103-D; P3 pool34777 P1pool107057 unchanged; parent equal demo3 cap.45; P3 seed90 P1seed10390; paired sampler prefixes and per-step RNG; horizons/global schedule2000+Piper step; C2 chain .1 predicted reentry .8',recipe_disclosure='A2 source P3-only; per explicit current P1+P3 request added paired full Piper P1 auxiliary .1 matching E103-D, disclosed before training; no substitution/reduction of Piper',evaluation='one inherited E102-X evaluator, checkpoint own A2 deterministic actions for pred_N1; all4 methods/all4 horizons; fixed human short language; main expanded27, original48 appendix; parent bootstrap2000 seed90',criterion='F1-F0 AND F1-F2: at least TWO SAME no_traj horizons among25/50/100 with lower CI>0; both comparisons all gt_action25/50/100 lower CI>-.01; joint L1 at8/50 and event F1 at5/10 point relative degradation <=5%; F1>F0 only: gain explainable by extra steps',GPU_seconds_cap=10800,disk_bytes_cap=10_000_000_000,peak_mib=6144,entries=entries,checkpoints={'A2':str(INIT),'F1-pre':str(T/'results/e104p_f1_s90_20261115_03/ckpt_pre_05000.pt'),'F0':str(T/'results/e104p_f2_s90_20261115_03/ckpt_05000.pt'),'F1':str(T/'results/e104p_f1_s90_20261115_03/ckpt_05000.pt'),'F2':str(T/'results/e104p_f2_s90_20261115_03/ckpt_10000.pt')},code_hashes={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')},input_hashes=hashes)
 assert not (CONTROL/'PRETRAIN_LOCK.json').exists();atomic(CONTROL/'PRETRAIN_LOCK.json',lock);report('## CPU合同核验与结果前锁定\n'+json.dumps(out,ensure_ascii=False)+'\n'+json.dumps({k:v for k,v in lock.items() if k not in ('entries','code_hashes','input_hashes')},ensure_ascii=False));print(json.dumps(out))
if __name__=='__main__':main()
