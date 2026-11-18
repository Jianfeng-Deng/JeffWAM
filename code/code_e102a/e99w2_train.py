import os,time,json,traceback,hashlib,collections,numpy as np,torch
from e99w2_common import *
from e99w2_data import load,audit,pose_np,planned_np
from e99w2_model import WorldModel,pose_torch,trajectory
from e99w2_objective import loss,chain_loss
from e100c_chain import corrected_chain,open_second
from e96a import train as base
from e90.data import sample_horizons
from e90.sampling import MultiSourceSampler
from e86.data import move,collate
from e86.common import generator

def make_chain(data,ids,k,step):
 chosen=[];shifted=[]
 for i in ids:
  if data.index.resolve(data.rows[i],2*k)[0] is None:continue
  b=data.shifted(i,k)
  if b is not None and b['world_valid'].all():chosen.append(i);shifted.append(b)
  if len(chosen)==2:break
 if not chosen:return None
 first=move(data.batch(chosen,(k,)),'cuda');direct=move(data.batch(chosen,(2*k,)),'cuda');second=collate(shifted);second['world_positions']=second['world_positions'][0];second=move(second,'cuda')
 check=second['world_raw'];check=check.reshape(len(chosen),3,8,-1).mean(2) if k==50 else check;assert torch.allclose(check,direct['world_raw'],atol=1e-6)
 predicted=bool(torch.rand((),generator=generator(SEED,step,'reentry')).item()<.2+.6*min(1.,max(0.,(step+5000-3000)/2000)))
 if GROUP in ('A0','A1','A2'):return lambda m:corrected_chain(m,first,second,direct,k,predicted,steps=2,seed=SEED+step)
 return lambda m:chain_loss(m,first,second,direct,k,predicted,steps=2,seed=SEED+step)

def train(tr):
 RUN.mkdir(exist_ok=False);guard=None
 try:
  from e100c_resource import Guard
  base.configure();torch.manual_seed(SEED);np.random.seed(SEED);__import__('random').seed(SEED);guard=Guard('E102A_'+GROUP,1900);m=WorldModel(tr.norm);cp=torch.load(INIT,map_location='cpu',weights_only=False);missing=m.load_state_dict(cp['model'],strict=False);assert not missing.unexpected_keys;assert all(k.startswith(('grip_head.','proprio_proj.','delta_head.')) for k in missing.missing_keys);del cp;m=m.to('cuda');opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
  cfg=base.config_for();cfg.update(group=GROUP,seed=SEED,steps=STEPS,schema='e102a-c2-continue-v1',no_action_training=False,alignment_weight=.01,phase_weight=0.,init=str(INIT),init_sha=sha(INIT),condition='C2 fixed-two-state world plan; A2 causal proprio always in A; BCE target-free t0; A2 normalized delta MSE; FM joints auxiliary weight1',plan_extension='hold action49 through100',world_loss='conditional deterministic MSE',checkpoint_interval=STEPS,dev='P3 V3 fixed48 windows/6parent; no selection',eval_N=1)
  atomic(RUN/'config.json',cfg);atomic(RUN/'provenance.json',dict(source_hash=source_hash(),brt=brt()))
  sampler=MultiSourceSampler(tr.rows,seed=SEED,weights={'piper':1.},caps={},demo3_cap=.45);raws,_=base.calibration_raws(tr);extra={'initial_adapter_pass':base.calibrate(m,raws,guard,RUN,'initial')};disabled=not extra['initial_adapter_pass'];samplehash=hashlib.sha256();values=[];modes=collections.Counter();chain_updates=0;m.train();profile=[]
  with (RUN/'updates.jsonl').open('x') as log:
   for step in range(1,STEPS+1):
    guard.check();started=time.monotonic()
    if step==3000:
     extra['boundary_adapter_pass']=base.calibrate(m,raws,guard,RUN,'chain_boundary');disabled|=not extra['boundary_adapter_pass']
    if step%16==0:
     with torch.no_grad():av,ac=m.adapter_loss(raws[64:])
     passed=base.adapter_pass(av,ac);m.adapter_ready.fill_(passed and not disabled);extra['last_adapter_check']=dict(step=step,nmse=float(av),cosine=float(ac),passed=passed)
    ids=sampler.sample(32);samplehash.update(np.asarray(ids,dtype='<i8').tobytes());hs=sample_horizons(step,SEED);b=move(tr.batch(ids,hs),'cuda');ch=make_chain(tr,ids,(25,50)[step%2],step) if True and step>0 and bool(m.adapter_ready) else None
    opt.zero_grad(set_to_none=True);v,mt=loss(m,b,step,ch);assert torch.isfinite(v),'Nonfinite loss';v.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn),'Nonfinite gradient';opt.step();torch.cuda.synchronize();mt.update(step=step,loss=float(v.detach()),seconds=time.monotonic()-started,horizons=list(hs));log.write(json.dumps(mt)+'\n');values.append(mt['loss']);modes[mt['mode']]+=1;chain_updates+=int(mt['chain_weight']>0);profile.append(mt['seconds'])
    if step==100:
     estimate=float(np.median(profile[-20:])*STEPS*1.45);atomic(RUN/'profile.json',dict(estimated_total_s_with_chain_margin=estimate,peak_mib=torch.cuda.max_memory_allocated()/2**20));assert estimate<1900,'Projected training exceeds reserved GPU budget'
    if step==1 or step%50==0:
     log.flush();atomic(RUN/'status.json',dict(status='training',step=step,elapsed_s=guard.check(),loss=mt['loss'],brt=brt()))
    if step%500==0:print('train',step,round(mt['loss'],5),round(guard.check(),1),flush=True)
  cp=RUN/'ckpt_02000.pt';torch.save(dict(model=m.state_dict(),step=STEPS,config=cfg,extra=extra,sampler=sampler.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),cp)
  result=dict(status='completed',steps=STEPS,checkpoint=str(cp),sha256=sha(cp),sampler=sampler.state_dict(),sample_sha256=samplehash.hexdigest(),modes=dict(modes),chain_updates=chain_updates,gate_disabled=disabled,extra=extra,first32=float(np.mean(values[:32])),last32=float(np.mean(values[-32:])),peak_mib=torch.cuda.max_memory_allocated()/2**20,elapsed_s=guard.check(),brt=brt())
  atomic(RUN/'train_summary.json',result);atomic(RUN/'status.json',result);atomic(ART/'TRAIN_COMPLETE.json',result);report('## 固定2000步训练完成\n'+json.dumps(result,ensure_ascii=False));del m,opt;torch.cuda.empty_cache()
 finally:
  if guard:guard.finish()
