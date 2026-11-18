import os,sys,time,json,hashlib,collections,random,traceback
import torch,numpy as np
from e103d_runtime import *
from e106c_train import reset,p1_loss,digest_params
from e99w2_data import load
from e90.data import load_data,sample_horizons
from e103d_data import p1_only
from e86.data import move
from e86.common import seed_for
from e99w2_train import make_chain
from e99w2_objective import loss
from e96a import train as base
from e107m_model import StochasticWorld,WorldModel,p3_energy,energy
from e107m_test import samplers,CK

def main(group):
 lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());assert json.loads((CONTROL/'CPU_TESTS.json').read_text())['status']=='passed';assert sha(CK)==lock['resume_sha'];run=T/f'results/e107m_{group.lower()}_s90_20261116_01';run.mkdir(exist_ok=False);guard=None
 try:
  base.configure();torch.backends.mha.set_fastpath_enabled(False);reset();_,_,tr,_=load();_,_,p1,_,_=load_data();cp=torch.load(CK,map_location='cpu',weights_only=False);m=(StochasticWorld if group=='M1' else WorldModel)(tr.norm);mis=m.load_state_dict(cp['model'],strict=group=='M0');s,p=samplers(tr,p1,cp)
  frozen={n for n,_ in m.named_parameters() if not n.startswith('noise_branch.')};before=digest_params(m,frozen)
  guard=Guard('train_'+group,4000,6144);m=m.to('cuda')
  if group=='M0':opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(cp['optimizer'])
  else:opt=torch.optim.AdamW(m.noise_branch.parameters(),lr=3e-4,weight_decay=1e-4)
  del cp;raws,_=base.calibration_raws(tr);reset();hh=hashlib.sha256();counts=collections.Counter()
  with (run/'updates.jsonl').open('x') as log:
   for step in range(20001,23001):
    guard.check();globalstep=step+2000
    if step%16==0 and group=='M0':
     with torch.no_grad():av,ac=m.adapter_loss(raws[64:]);m.adapter_ready.fill_(base.adapter_pass(av,ac))
    torch.manual_seed(seed_for(90,'e104p_piper',step));ids=s.sample(32);hs=sample_horizons(globalstep,90);b=move(tr.batch(ids,hs),'cuda');m.train();opt.zero_grad(set_to_none=True)
    if group=='M0':ch=make_chain(tr,ids,(25,50)[globalstep%2],globalstep) if bool(m.adapter_ready) else None;v,mt=loss(m,b,globalstep,ch)
    else:v,mt=p3_energy(m,b,globalstep)
    assert torch.isfinite(v);v.backward()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
     torch.manual_seed(seed_for(90,'e104p_p1',step));pids=p.sample(32);pb=move(p1_only(p1.batch(pids,(10,25,50,100),'instruction','P1')),'cuda')
     if group=='M0':pv=p1_loss(m,pb)
     else:
      with torch.no_grad():o=m.forward_mean(pb)['mean']
      pv=energy(m,pb,o,globalstep+1900000)
     assert torch.isfinite(pv);(.1*pv).backward()
    gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step();torch.cuda.synchronize();hh.update(json.dumps([ids,pids,list(hs)],default=int).encode());counts[mt['mode']]+=1
    if step%50==0:
     last=dict(step=step,loss=float(v.detach()+.1*pv.detach()),p1=float(pv.detach()),elapsed_s=guard.check(),**mt);log.write(json.dumps(last)+'\n');log.flush();atomic(run/'status.json',last)
    if step%500==0:print(group,step,round(guard.check(),1),flush=True)
   assert hh.hexdigest()==lock['flow_sha'] and dict(counts)==lock['modes']
   if group=='M1':assert digest_params(m,frozen)==before
   path=run/'ckpt_23000.pt';assert not path.exists();torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),sampler=s.state_dict(),p1sampler=p.state_dict(),step=23000,group=group,seed=90),path)
   out=dict(status='complete',step=23000,updates=3000,flow_sha=hh.hexdigest(),modes=dict(counts),frozen_unchanged=group=='M1' and digest_params(m,frozen)==before,elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,ckpt_sha=sha(path),brt=brt());atomic(run/'summary.json',out);report('## '+group+'固定3000步训练完成\n'+json.dumps(out,ensure_ascii=False))
 finally:
  if guard:guard.finish()
if __name__=='__main__':main(sys.argv[1])
