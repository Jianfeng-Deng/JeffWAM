import os,sys,time,json,traceback,hashlib
from pathlib import Path
import torch,numpy as np
from e96a.common import T,ART,atomic,brt,sha,report,source_hash
from e96a import train as base
from e96a.model import WorldModel as E96
from e98w_model import WorldModel
from e98w_data import augment
from e90.data import load_data
from e90.sampling import MultiSourceSampler
from e86.data import move

def contracts(data):
 base.configure();m=WorldModel(data.norm,'A');state=torch.get_rng_state().clone();base.configure();ref=E96();same=all(torch.equal(v,m.state_dict()[k]) for k,v in ref.state_dict().items());assert same;assert torch.equal(state,torch.get_rng_state())
 b=data.batch([0,100],(10,25,50,100),'short');m.eval()
 # Nonzero random readout is essential: otherwise a zero-initialized head conceals leaks.
 with torch.no_grad():m.mean_head[-1].weight.normal_(std=.01);out=m.forward_mean(b);changed=dict(b,robot=b['robot']+7);out2=m.forward_mean(changed)
 assert torch.equal(out['observation'],out2['observation']);delta=float((out['mean']-out2['mean']).abs().max());assert delta>1e-6
 with torch.no_grad():s=m.robot_prediction(b)
 fixed=[]
 from e98w_core import R,States,predict
 st=States()
 for h in (10,25,50,100):
  z=dict(np.load(R/f'R1_h{h}.npz'));feat=np.stack([st.features(data,data.rows[i],h)[0] for i in [0,100]]);f=dict(b,robot_fold=torch.full((2,),5));f.pop('robot_s',None)
  with torch.no_grad():pred=m.robot_prediction(f)[:,b['world_horizons'][0]==h].numpy()
  want=predict(z,None,(feat-z['zm'])/z['zs'],'R1');err=float(abs(pred-want).max());assert err<5e-5,err;fixed.append(dict(h=h,max_abs_full_R1_diff=err))
 residual=b['world']-s;err=((s+residual)-b['world']).abs().max();assert float(err)<2e-6
 # Every original sampler and horizon draw stays unchanged across A and B.
 samplers=[MultiSourceSampler(data.rows,seed=90,weights={'piper':1.},caps={},demo3_cap=.45) for _ in range(2)]
 assert samplers[0].sample(320)==samplers[1].sample(320)
 result=dict(status='passed',brt=brt(),old_parameter_initialization_exact=same,O_max_robot_perturbation=0,W_nonzero_robot_response=delta,full_R1_fixed_h=fixed,addback_max_error=float(err),A_B_parameter_count=sum(p.numel() for p in m.parameters()),trajectory_rows=101,no_dev=True)
 atomic(ART/'contracts.json',result);report('\n### 实现合同通过\n'+json.dumps(result,ensure_ascii=False));return result

def train(variant,data):
 run=T/'results'/f'e98w_{variant.lower()}_20261114_04';run.mkdir(exist_ok=False);guard=None
 try:
  base.configure();cfg=base.config_for();cfg.update(schema='e98w-v1',variant=variant,checkpoint_interval=5000,trajectory_rows=101,crossfit='5-fold parent, seed98090',s_rule='locked CROSSFIT_LOCK.json')
  atomic(run/'config.json',cfg)
  from e96a.resource import Guard
  guard=Guard('E98W_'+variant,3400);m=WorldModel(data.norm,variant).to('cuda:0');opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=cfg['weight_decay']);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
  sampler=MultiSourceSampler(data.rows,seed=90,weights=cfg['source_weights'],caps=cfg['source_caps'],demo3_cap=cfg['demo3_cap']);raws,_=base.calibration_raws(data)
  extra={'initial_adapter_pass':base.calibrate(m,raws,guard,run,'initial')};gate_disabled=not extra['initial_adapter_pass'];values=[];chain_updates=0;samplehash=hashlib.sha256();orig_sample=sampler.sample
  def sample(n):
   ids=orig_sample(n);samplehash.update(np.asarray(ids,dtype='<i8').tobytes());return ids
  sampler.sample=sample;m.train();profile=[]
  with (run/'updates.jsonl').open('x') as log:
   for step in range(1,5001):
    if step==cfg['chain_warmup']:
     passed=base.calibrate(m,raws,guard,run,'chain_boundary');gate_disabled|=not passed;extra['boundary_adapter_pass']=passed
    if step%16==0:
     with torch.no_grad():av,ac=m.adapter_loss(raws[64:])
     passed=base.adapter_pass(av,ac);m.adapter_ready.fill_(passed and not gate_disabled);extra['last_adapter_check']=dict(step=step,nmse=float(av),cosine=float(ac),passed=passed)
    metrics=base.update(m,opt,scheduler,sampler,data,step,cfg,guard);values.append(metrics['loss']);chain_updates+=int(metrics['chain_weight']>0);log.write(json.dumps(metrics)+'\n')
    if step<=100:profile.append(metrics['seconds'])
    if step==100:
     p=dict(steps=100,median_update_s=float(np.median(profile[-20:])),projected_5000_s=float(np.median(profile[-20:])*5000),peak_mib=torch.cuda.max_memory_allocated()/2**20,GPU_seconds=guard.check());atomic(run/'profile.json',p)
     # Profile uses the first100 authorized training steps; never restarts or changes recipe.
     if p['projected_5000_s']>3150:raise RuntimeError('Profile projects run beyond reserved GPU budget')
    if step==1 or step%50==0:
     log.flush();atomic(run/'status.json',dict(status='training',step=step,loss=metrics['loss'],elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,brt=brt()))
    if step%500==0:print(variant,step,round(metrics['loss'],5),round(guard.check(),1),flush=True)
  # Only the predeclared fixed endpoint is saved.
  cp=run/'ckpt_05000.pt';torch.save(dict(model=m.state_dict(),step=5000,config=cfg,source_hash=source_hash(),sampler=sampler.state_dict(),extra=extra),cp)
  result=dict(status='completed',steps=5000,checkpoint=str(cp),sha256=sha(cp),sampler=sampler.state_dict(),sample_indices_sha256=samplehash.hexdigest(),trainable_parameters=sum(p.numel() for p in m.parameters()),first32=float(np.mean(values[:32])),last32=float(np.mean(values[-32:])),chain_updates=chain_updates,adapter_gate_disabled=gate_disabled,extra=extra,peak_mib=torch.cuda.max_memory_allocated()/2**20,elapsed_s=guard.check(),brt=brt())
  atomic(run/'train_summary.json',result);atomic(run/'status.json',result);report('\n### '+variant+'固定5000步完成\n'+json.dumps(result,ensure_ascii=False));del m,opt,scheduler;torch.cuda.empty_cache()
 except BaseException as e:
  atomic(run/'status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc(),brt=brt()));raise
 finally:
  if guard:guard.finish()

def main():
 assert (ART/'CROSSFIT_LOCK.json').exists();_,_,data,_,_=load_data();augment(data,True)
 if not (ART/'contracts.json').exists():contracts(data)
 # State/FK I/O is CPU-only and prewarmed before GPU admission.
 for cid in sorted({r['canonical_id'] for r in data.rows}):data.robot_states.get(data,cid)
 atomic(ART/'STATUS.json',dict(status='training',brt=brt()))
 train('A',data);train('B',data)
 a=json.loads((T/'results/e98w_a_20261114_04/train_summary.json').read_text());b=json.loads((T/'results/e98w_b_20261114_04/train_summary.json').read_text());assert a['sample_indices_sha256']==b['sample_indices_sha256'] and a['sampler']==b['sampler'] and a['trainable_parameters']==b['trainable_parameters']
 atomic(ART/'TRAIN_COMPLETE.json',dict(status='completed',brt=brt(),sample_sha=a['sample_indices_sha256'],parameter_count=a['trainable_parameters']))
 atomic(ART/'STATUS.json',dict(status='trained_awaiting_evaluation',brt=brt()))
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(ART/'STATUS.json',dict(status='stopped',traceback=traceback.format_exc(),brt=brt()));raise
