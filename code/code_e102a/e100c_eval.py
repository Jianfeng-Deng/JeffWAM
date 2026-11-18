import json,torch,numpy as np
from e99w2_common import *
from e99w2_data import load
from e99w2_model import WorldModel,trajectory
from e99w2_complete_h100 import UnionImage,EXTRA
from e88.audit import HorizonIndex
from e86.data import move,collate
from e96a.train import configure
from e100c_chain import reentered
from e100c_resource import Guard
H=(10,25,50,100)
def finite(x):
 assert torch.isfinite(x).all() if torch.is_tensor(x) else np.isfinite(x).all(),'Nonfinite refused'
 return x

def evaluate():
 configure();torch.backends.mha.set_fastpath_enabled(False);_,_,tr,dev=load()
 c0=json.loads((dev.base.cache/'contract.json').read_text());c1=json.loads((EXTRA/'contract.json').read_text());assert c0['preprocess']==c1['preprocess'] and c0['encoder_files']==c1['encoder_files'];assert json.loads((T/'results/e93_dev_diag_20261114_01/cache_replay.json').read_text())['max_abs_difference']==0
 dev.base.__class__=UnionImage;dev.index=HorizonIndex(EXTRA,list(dev.index.captures.values()))
 cp=torch.load(RUN/'ckpt_05000.pt',map_location='cpu',weights_only=False);assert cp['step']==5000;m=WorldModel(tr.norm);m.load_state_dict(cp['model']);del cp
 guard=Guard('eval',300);records=[];chains=[];diag=[];coverage={}
 try:
  m=m.to('cuda');m.eval()
  with torch.inference_mode():
   for i,r in enumerate(dev.rows):
    guard.check();b=move(dev.batch([i],H,'short'),'cuda');actions=[finite(m.infer(b,'action_only',4,90+n)[0]) for n in range(8)]
    for h in H:
     bh=move(dev.batch([i],(h,),'short'),'cuda')
     if not bh['world_valid'].all():continue
     coverage.setdefault(str(h),[]).append(i);Y=finite(bh['world']);no=dict(bh,robot=torch.zeros_like(bh['robot']),robot_valid=torch.zeros_like(bh['robot_valid']))
     pp={'gt_action':m.forward_mean(bh)['mean'],'no_traj':m.forward_mean(no)['mean']};samples=[m.forward_mean(dict(bh,robot=trajectory(bh['context'],a)))['mean'] for a in actions];pp.update(pred_N1=samples[0],pred_N8=torch.stack(samples).mean(0))
     for method,P in pp.items():
      finite(P);e=(P-Y).double();y=Y.double();records.append(dict(index=i,parent=r['parent_component'],h=h,method=method,model=float(e.square().mean()),copy=float(y.square().mean()),cosine=float(torch.nn.functional.cosine_similarity(P.flatten(),Y.flatten(),dim=0))))
    for k in (25,50):
     first=move(dev.batch([i],(k,),'short'),'cuda');direct=move(dev.batch([i],(2*k,),'short'),'cuda')
     if not direct['world_valid'].all():continue
     # True open-loop needs no shifted real observation/history/state.
     # Real middle frame is used exclusively as a diagnostic reference.
     teacher=dict(first,obs=first['obs'].clone());teacher['obs'][:,4]=first['world_raw'].reshape(1,3,8,-1)
     second,z=reentered(m,first,teacher,k,True,steps=2,seed=90);end=m.mean_rollout(second,2,91);dp=m.mean_rollout(direct,2,92)
     if k==50:end=end.reshape(1,3,8,-1).mean(2)
     truth=direct['world_raw'];scale=direct['world_scale'];err=finite(((end-truth)/scale).square().mean());derr=finite(((dp-truth)/scale).square().mean())
     chains.append(dict(index=i,parent=r['parent_component'],k=k,model=float(err),direct=float(derr)))
     pred=finite(m.reentry(z));real=finite(m.reentry(teacher['obs'][:,4]));nmse=finite((pred-real).square().mean()/real.square().mean().clamp_min(1e-6));cos=finite(torch.nn.functional.cosine_similarity(pred,real,dim=-1).mean());ad,ac=m.adapter_loss(z)
     diag.append(dict(index=i,parent=r['parent_component'],k=k,nmse=float(nmse),cosine=float(cos),predicted_adapter_self_nmse=float(finite(ad)),predicted_adapter_self_cosine=float(finite(ac))))
    if i%12==0:print('eval',GROUP,SEED,i,flush=True)
   torch.cuda.synchronize()
 finally:guard.finish()
 atomic(ART/'evaluation.json',dict(group=GROUP,seed=SEED,records=records,chains=chains,diagnostics=diag,coverage=coverage,checkpoint_sha=sha(RUN/'ckpt_05000.pt'),finite=True,MHA_fastpath=False,brt=brt()))
 print('eval complete',len(records),len(chains),flush=True)
if __name__=='__main__':evaluate()
