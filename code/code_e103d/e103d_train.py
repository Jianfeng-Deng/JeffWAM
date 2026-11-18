import os,json,time,collections,hashlib,traceback
import numpy as np,torch
from e103d_runtime import *
from e99w2_common import INIT
from e103d_model import WorldModel
from e103d_data import External,p1_only
from e99w2_data import load
from e90.data import load_data
from e90.sampling import MultiSourceSampler
from e86.data import move,collate
from e86.common import seed_for
from e96a import train as base
from e99w2_train import make_chain
from e99w2_objective import loss
from e90.objective import weighted_mean
H=(10,25,50,100)

def p1_batch(piper,external,sampler,group,step):
 ids=sampler.sample(32);n=32 if group=='D0' else 20;items=[];counts={'piper':n}
 for i in ids[:n]:
  b=p1_only(piper.get(i,H,'instruction','P1'));b.pop('external_no_robot');items.append(b)
 if group=='D1':
  for source in ['yam','abc']:
   # Missing source count is explicitly returned to Piper.
   if external.pools[source]:
    exids=external.sample(source,6,step);counts[source]=6;items.extend(external.get(i) for i in exids)
   else:
    counts['piper']+=6
    for i in ids[len(items):len(items)+6]:
     b=p1_only(piper.get(i,H,'instruction','P1'));b.pop('external_no_robot');items.append(b)
 assert len(items)==32
 b=collate(items);b['world_positions']=b['world_positions'][0];b['external_no_robot']=True;return b,counts,ids

def p1_loss(model,b):
 out=model.forward_mean(b);world=weighted_mean((out['mean']-b['world']).square().mean(-1),b['world_valid'],b['world_horizons']);cos=weighted_mean(1-torch.nn.functional.cosine_similarity(out['world_cos'],b['world'],dim=-1),b['world_valid'],b['world_horizons'],True);ad,_=model.adapter_loss(b['obs'][:,4]);v=world+.1*cos+.1*ad
 return v,dict(p1_world=float(world.detach()),p1_cos=float(cos.detach()),p1_adapter=float(ad.detach()))

def contracts():
 assert json.loads((CONTROL/'EARLY_DECODE_AUDIT.json').read_text())['status']=='passed'
 base.configure();_,_,tr,_=load();_,_,p1,_,_=load_data();ext=External(p1);m=WorldModel(tr.norm);m.load_state_dict(torch.load(INIT,map_location='cpu',weights_only=False)['model'],strict=True);m.eval();b=tr.batch([0,100],H,'short');no=dict(b,robot=torch.zeros_like(b['robot']),robot_valid=torch.zeros_like(b['robot_valid']));pure=p1_only(b)
 with torch.no_grad():
  original=m.forward_mean(no)['mean'];deleted=m.forward_mean(pure)['mean'];diff=float(abs(original-deleted).max());assert diff<2e-5,(diff,'R deletion changed valid W outputs')
 # Any robot/state/action projection invocation is a hard failure in P1.
 def forbidden(*a):raise AssertionError('Forbidden robot/action/state projection in P1')
 hooks=[x.register_forward_hook(forbidden) for x in [m.robot_proj,m.state_proj,m.action_proj]]
 for source in ['abc','yam']:
  ids=ext.sample(source,2,1);eb=ext.batch(ids);m.zero_grad(set_to_none=True);m.train();v,_=p1_loss(m,eb);assert torch.isfinite(v);v.backward();assert m.action_head.weight.grad is None;assert m.robot_proj.weight.grad is None;assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
 for h in hooks:h.remove()
 mods={name:Path(__import__(name).__file__).resolve() for name in ['e99w2_train','e99w2_objective','e103d_model','e103d_data','e98w_core']};assert all(p.parent==ROOT for p in mods.values())
 coverage={};receipts={}
 for rec in ext.plan['records']:
  source=rec['source'];d=CACHE/'features'/rec['canonical_id'];rc=d/'receipt.json';assert rc.exists();receipts[rec['canonical_id']]=sha(rc);c=coverage.setdefault(source,dict(episodes=0,windows=0,camera_frames=0,hours=0.,hours_scope='ABC common-camera recorded span' if source=='abc' else 'E87 nominal frames/30, not measured exposure duration',payload_bytes=0));c['episodes']+=1;c['windows']+=len(rec['windows']);c['hours']+=rec['original'].get('duration_s',rec['original'].get('nominal_seconds',0))/3600
  for k,cam in enumerate(['cam_high','cam_left_wrist','cam_right_wrist']):
   ids=np.load(d/(cam+'_rows.npy'));arr=np.load(d/(cam+'.npy'),mmap_mode='r');assert np.array_equal(ids,rec['rows'][k]);assert arr.shape==(len(ids),8,768) and arr.dtype==np.dtype('float16');c['camera_frames']+=len(ids);c['payload_bytes']+=arr.nbytes
 cov=dict(status='passed',by_source=coverage,receipt_sha256=receipts,brt=brt());atomic(CONTROL/'CACHE_COVERAGE.json',cov);report('## 最终缓存覆盖核验\n'+json.dumps(coverage,ensure_ascii=False))
 out=dict(status='passed',physical_robot_action_state_deletion=True,no_action_gradient=True,finite_abc_yam=True,no_traj_equivalence_max_abs=diff,initial_sha256=sha(INIT),modules={k:str(v) for k,v in mods.items()},cuda_initialized=torch.cuda.is_initialized(),brt=brt());assert not torch.cuda.is_initialized();atomic(CONTROL/'CPU_TESTS.json',out);report('## CPU 合同验证\n'+json.dumps(out,ensure_ascii=False));print(out,flush=True)

def train(group):
 assert (CONTROL/'PRETRAIN_LOCK_02.json').exists() and json.loads((CONTROL/'CPU_TESTS.json').read_text())['status']=='passed';assert sha(CACHE/'plan.json')==json.loads((CONTROL/'PRETRAIN_LOCK_02.json').read_text())['plan_sha256']
 run=T/f'results/e103d_{group.lower()}_s90_20261115_02';run.mkdir(exist_ok=False);guard=None
 try:
  base.configure();_,_,tr,_=load();_,_,p1,_,_=load_data();ext=External(p1);guard=Guard('train_'+group+'_02',5400,6000);m=WorldModel(tr.norm);cp=torch.load(INIT,map_location='cpu',weights_only=False);m.load_state_dict(cp['model'],strict=True);m=m.to('cuda');opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(cp['optimizer']);del cp
  sampler=MultiSourceSampler(tr.rows,seed=90,weights={'piper':1.},caps={},demo3_cap=.45);p1sampler=MultiSourceSampler(p1.rows,seed=10390,weights={'piper':1.},caps={},demo3_cap=.45);counts=collections.Counter();modes=collections.Counter();hs_hash=hashlib.sha256();raws,_=base.calibration_raws(tr);passed=base.calibrate(m,raws,guard,run,'initial');disabled=not passed;last={}
  atomic(run/'config.json',dict(json.loads((CONTROL/'PRETRAIN_LOCK_02.json').read_text()),group=group,optimizer='restored C2 AdamW state; same lr3e-4, wd1e-4',P1_world_weight=.1))
  with (run/'updates.jsonl').open('x') as log:
   for step in range(1,5001):
    guard.check();globalstep=5000+step
    if step%16==0:
     with torch.no_grad():av,ac=m.adapter_loss(raws[64:])
     m.adapter_ready.fill_(base.adapter_pass(av,ac) and not disabled)
    ids=sampler.sample(32);hs_hash.update(np.asarray(ids,dtype='<i8').tobytes());from e90.data import sample_horizons
    hs=sample_horizons(globalstep,90);b=move(tr.batch(ids,hs),'cuda');ch=make_chain(tr,ids,(25,50)[globalstep%2],globalstep) if bool(m.adapter_ready) else None
    m.train();opt.zero_grad(set_to_none=True);v,mt=loss(m,b,globalstep,ch);assert torch.isfinite(v);v.backward()
    # Independent RNG stream keeps all Piper P3 stochastic draws equal across arms.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
     torch.manual_seed(seed_for(90,'e103d_p1',step));pb,ct,_=p1_batch(p1,ext,p1sampler,group,step);pb=move(pb,'cuda');pv,pm=p1_loss(m,pb);assert torch.isfinite(pv);(.1*pv).backward();counts.update(ct)
    gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step();torch.cuda.synchronize();last=dict(step=step,global_step=globalstep,loss=float(v.detach()+.1*pv.detach()),**mt,**pm);modes[mt['mode']]+=1;log.write(json.dumps(last)+'\n')
    if step==1 or step%50==0:log.flush();atomic(run/'status.json',dict(status='training',elapsed_s=guard.check(),brt=brt(),**last))
    if step%500==0:print(group,step,round(last['loss'],5),round(guard.check(),1),flush=True)
   ck=run/'ckpt_05000.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=5000,global_step=10000,group=group,seed=90),ck)
  out=dict(status='completed',steps=5000,checkpoint=str(ck),checkpoint_sha256=sha(ck),elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,P1_counts=dict(counts),P3_sampler=sampler.state_dict(),P3_samples_sha256=hs_hash.hexdigest(),modes=dict(modes),adapter_initial=passed,last=last,brt=brt());atomic(run/'status.json',out);atomic(run/'train_summary.json',out);report('## '+group+' 训练完成\n'+json.dumps(out,ensure_ascii=False))
 finally:
  if guard:guard.finish()
if __name__=='__main__':
 import sys
 try:
  if sys.argv[1]=='contracts':contracts()
  else:train(sys.argv[1])
 except BaseException:atomic(CONTROL/('error_'+sys.argv[1]+'_02.json'),dict(error=traceback.format_exc(),brt=brt()));raise
