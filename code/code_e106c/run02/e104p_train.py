import os,sys,time,json,hashlib,collections,random,traceback
import numpy as np,torch
from e103d_runtime import *
from e99w2_common import INIT
from e103d_model import WorldModel
from e103d_data import External,p1_only
from e99w2_data import load
from e90.data import load_data,sample_horizons
from e90.sampling import MultiSourceSampler
from e86.common import seed_for
from e86.data import move,collate
from e96a import train as base
from e99w2_train import make_chain
from e99w2_objective import loss
from e90.objective import weighted_mean
H=(10,25,50,100)
def p1_loss(m,b):
 o=m.forward_mean(b);w=weighted_mean((o['mean']-b['world']).square().mean(-1),b['world_valid'],b['world_horizons']);c=weighted_mean(1-torch.nn.functional.cosine_similarity(o['world_cos'],b['world'],dim=-1),b['world_valid'],b['world_horizons'],True);ad,_=m.adapter_loss(b['obs'][:,4]);return w+.1*c+.1*ad

def digest_params(m,names):
 h=hashlib.sha256()
 for n,p in m.named_parameters():
  if n in names:h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def reset():torch.manual_seed(90);np.random.seed(90);random.seed(90)
def train(group):
 lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());assert sha(INIT)==lock['initial_sha'];assert sha(CACHE/'plan.json')==lock['plan_sha'];run=T/f'results/e104p_{group.lower()}_s90_20261115_03';run.mkdir(exist_ok=False);guard=None
 try:
  base.configure();reset();_,_,tr,_=load();_,_,p1,_,_=load_data();ext=External(p1)
  guard=Guard('train_'+group,8000,6144);m=WorldModel(tr.norm);cp=torch.load(INIT,map_location='cpu',weights_only=False);m.load_state_dict(cp['model'],strict=True);m=m.to('cuda');opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(cp['optimizer']);del cp
  for pg in opt.param_groups:assert pg['lr']==3e-4 and pg['weight_decay']==1e-4
  # Both Piper phases start from the exact same RNG state; optimizer is NEVER reset.
  def save(name,step,phase):
   path=run/name;assert not path.exists();torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step,phase=phase,group=group,seed=90,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path);atomic(run/(name+'.json'),dict(sha256=sha(path),step=step,phase=phase,optimizer_steps={str(int(v['step'])) for v in []} if False else sorted({int(v['step']) for v in opt.state.values() if 'step' in v}),brt=brt()))
  if group=='F1':
   frozen=[n for n,_ in m.named_parameters() if n.startswith(('action_head.','action_proj.','state_proj.','robot_proj.','proprio_proj.','delta_head.','grip_head.'))];before=digest_params(m,frozen);count=collections.Counter()
   # grad=None prevents both Adam momentum updates and decoupled decay for frozen parameters.
   for n,p in m.named_parameters():
    if n in frozen:p.requires_grad_(False)
   with (run/'external_updates.jsonl').open('x') as log:
    for step in range(1,5001):
     guard.check();ids=ext.sample('abc',16,step)+ext.sample('yam',16,step);b=move(ext.batch(ids),'cuda');m.train();opt.zero_grad(set_to_none=True);v=p1_loss(m,b);assert torch.isfinite(v);v.backward();assert all(p.grad is None for n,p in m.named_parameters() if n in frozen);gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step();torch.cuda.synchronize();count.update(abc=16,yam=16)
     if step==1 or step%50==0:log.write(json.dumps(dict(step=step,loss=float(v.detach())))+'\n');log.flush();atomic(run/'status.json',dict(status='external_pretrain',step=step,loss=float(v.detach()),elapsed_s=guard.check(),brt=brt()))
     if step%500==0:print(group,'pre',step,round(guard.check(),1),flush=True)
   assert digest_params(m,frozen)==before
   atomic(run/'EXTERNAL_FREEZE_AUDIT.json',dict(status='passed',frozen_names=frozen,hash_before=before,hash_after=digest_params(m,frozen),counts=dict(count),no_action_grip_chain_loss=True))
   save('ckpt_pre_05000.pt',5000,'external')
   for n,p in m.named_parameters():p.requires_grad_(True)
   report('## F1 外部阶段完成\n'+json.dumps(dict(counts=dict(count),frozen_unchanged=True,checkpoint=str(run/'ckpt_pre_05000.pt'),elapsed_s=guard.check()),ensure_ascii=False))
  reset();sampler=MultiSourceSampler(tr.rows,seed=90,weights={'piper':1.},caps={},demo3_cap=.45);p1sampler=MultiSourceSampler(p1.rows,seed=10390,weights={'piper':1.},caps={},demo3_cap=.45);raws,_=base.calibration_raws(tr)
  # Calibration consumes model-dependent compute; isolate RNG so the paired Piper stream stays equal.
  with torch.no_grad():
   av,ac=m.adapter_loss(raws[64:]);passed=base.adapter_pass(av,ac);m.adapter_ready.fill_(passed)
  disabled=not passed;hash3=hashlib.sha256();hash1=hashlib.sha256();hashflow=hashlib.sha256();modes=collections.Counter();steps=10000 if group=='F2' else 5000
  with (run/'piper_updates.jsonl').open('x') as log:
   for step in range(1,steps+1):
    guard.check();globalstep=2000+step
    if step%16==0:
     with torch.no_grad():av,ac=m.adapter_loss(raws[64:])
     m.adapter_ready.fill_(base.adapter_pass(av,ac) and not disabled)
    # Per-step global RNG pairing survives differing adapter gates across models.
    torch.manual_seed(seed_for(90,'e104p_piper',step));ids=sampler.sample(32);hs=sample_horizons(globalstep,90);hash3.update(np.asarray(ids,dtype='<i8').tobytes());hashflow.update(json.dumps([ids,list(hs)],default=int).encode());b=move(tr.batch(ids,hs),'cuda');ch=make_chain(tr,ids,(25,50)[globalstep%2],globalstep) if bool(m.adapter_ready) else None
    m.train();opt.zero_grad(set_to_none=True);v,mt=loss(m,b,globalstep,ch);assert torch.isfinite(v);v.backward()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
     torch.manual_seed(seed_for(90,'e104p_p1',step));pids=p1sampler.sample(32);hash1.update(np.asarray(pids,dtype='<i8').tobytes());pb=p1_only(p1.batch(pids,H,'instruction','P1'));pv=p1_loss(m,move(pb,'cuda'));assert torch.isfinite(pv);(.1*pv).backward()
    gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step();torch.cuda.synchronize();modes[mt['mode']]+=1;last=dict(step=step,global_step=globalstep,loss=float(v.detach()+.1*pv.detach()),p1_loss=float(pv.detach()),horizons=list(hs),**mt)
    if step==1 or step%50==0:log.write(json.dumps(last)+'\n');log.flush();atomic(run/'status.json',dict(status='piper_training',elapsed_s=guard.check(),brt=brt(),**last))
    if step%500==0:print(group,'Piper',step,round(guard.check(),1),flush=True)
    if step in (5000,10000):
     save(f'ckpt_{step:05d}.pt',step,'piper');out=dict(status='completed' if step==steps else 'midpoint_saved',step=step,P3_samples_sha256=hash3.hexdigest(),P1_samples_sha256=hash1.hexdigest(),P3_flow_sha256=hashflow.hexdigest(),modes=dict(modes),P3_draws=step*32,P1_draws=step*32,P3_pool=len(tr.rows),P1_pool=len(p1.rows),elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,adapter_initial=passed,last=last,brt=brt());atomic(run/f'summary_{step:05d}.json',out);report('## '+group+' Piper '+str(step)+'步\n'+json.dumps(out,ensure_ascii=False))
  atomic(run/'status.json',out)
 finally:
  if guard:guard.finish()
if __name__=='__main__':
 try:train(sys.argv[1])
 except BaseException:atomic(CONTROL/('train_'+sys.argv[1]+'_error.json'),dict(traceback=traceback.format_exc(),brt=brt()));raise
