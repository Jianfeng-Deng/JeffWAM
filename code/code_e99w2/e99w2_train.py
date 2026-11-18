import os,time,json,traceback,hashlib,collections,numpy as np,torch
from e99w2_common import *
from e99w2_data import load,audit,pose_np,planned_np
from e99w2_model import WorldModel,pose_torch,trajectory
from e99w2_objective import loss,chain_loss
from e96a import train as base
from e90.data import sample_horizons
from e90.sampling import MultiSourceSampler
from e86.data import move,collate
from e86.common import generator

def contracts(tr):
 base.configure();m=WorldModel(tr.norm);cp=torch.load(INIT,map_location='cpu',weights_only=False);assert cp['step']==5000 and sha(INIT)=='df0ed0f5fa905501ef458713e15e8945abeada2e845e5defe6e1e6cd4afadbcc';m.load_state_dict(cp['model'],strict=True);del cp
 b=tr.batch([0,100],(10,25,50,100),'short');m.eval();q=b['action'];err=float((pose_torch(q)-torch.tensor(pose_np(q.numpy()))).abs().max());assert err<2e-6
 feat=trajectory(b['context'],q);assert torch.allclose(feat,b['robot'],atol=2e-6)
 with torch.no_grad():
  o=m.forward_mean(b);c=m.forward_mean(dict(b,robot=b['robot']+1));assert torch.equal(o['observation'],c['observation']);assert float((o['mean']-c['mean']).abs().max())>0
  empty=m.forward_mean(dict(b,robot_valid=torch.zeros_like(b['robot_valid'])))['mean'];assert torch.isfinite(empty).all()
  # No GT actions or trajectory in action_only forward.
  a=torch.zeros_like(q);t=torch.ones(2)*.5;v=m(b,a,None,t,t)['action'];v2=m(dict(b,robot=b['robot']+999,action=q+999),a,None,t,t)['action'];assert torch.equal(v,v2)
  # At clean time actual action features reproduce fixed GT conditions.
  ya=(q-b['q_anchor'][:,None]-b['action_mean'])/b['action_std'];x=m(b,ya,torch.empty_like(b['world']),torch.ones(2),t)['mean'];y=m(dict(b,fixed_condition=True),ya,torch.empty_like(b['world']),torch.ones(2),t)['mean'];assert torch.allclose(x,y,atol=2e-4)
  # Target values cannot enter the deterministic head.
  z=m.forward_mean(dict(b,world=b['world']+999))['mean'];assert torch.equal(o['mean'],z)
 # All four real losses/backward must exercise finite A and W gradients.
 from e86.objective import mode_for
 seen={};m.train()
 for step in range(1,80):
  mode=mode_for(step,90,(.25,)*4)
  if mode in seen:continue
  m.zero_grad(set_to_none=True);v,mt=loss(m,b,step);assert torch.isfinite(v);v.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());assert m.action_head.weight.grad.abs().sum()>0
  if mode!='action_only':assert m.mean_head[-1].weight.grad.abs().sum()>0
  seen[mode]=float(v.detach())
  if len(seen)==4:break
 assert len(seen)==4
 out=dict(status='passed',init_sha256=sha(INIT),strict_initialization=True,FK_max_abs=err,trajectory_mapping=True,finite_no_traj=True,action_only_no_GT_dependency=True,deterministic_no_target_dependency=True,modes=seen,parameters=sum(p.numel() for p in m.parameters()),CUDA_initialized=torch.cuda.is_initialized())
 atomic(ART/'contracts.json',out);report('## CPU实现合同\n'+json.dumps(out,ensure_ascii=False));del m

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
 predicted=bool(torch.rand((),generator=generator(90,step,'reentry')).item()<.2+.6*min(1.,max(0.,(step-3000)/2000)))
 return lambda m:chain_loss(m,first,second,direct,k,predicted,steps=2,seed=90+step)

def train(tr):
 RUN.mkdir(exist_ok=False);guard=None
 try:
  from e96a.resource import Guard
  base.configure();guard=Guard('E99W2_P3_full_v31',11500);m=WorldModel(tr.norm);cp=torch.load(INIT,map_location='cpu',weights_only=False);m.load_state_dict(cp['model'],strict=True);del cp;m=m.to('cuda');opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
  cfg=base.config_for();cfg.update(schema='e99w2-full-v31-p3-v1',no_action_training=False,alignment_weight=.01,phase_weight=0.,init=str(INIT),init_sha=sha(INIT),condition='current noised action iterate FK in sync/independent, GT in clean; action_only physically deletes W/R',plan_extension='hold action49 through100',world_loss='conditional deterministic MSE',checkpoint_interval=5000,dev='P3 V3 fixed48 windows/6parent; no selection',eval_N=8)
  atomic(RUN/'config.json',cfg);atomic(RUN/'provenance.json',dict(source_hash=source_hash(),brt=brt()))
  sampler=MultiSourceSampler(tr.rows,seed=90,weights={'piper':1.},caps={},demo3_cap=.45);raws,_=base.calibration_raws(tr);extra={'initial_adapter_pass':base.calibrate(m,raws,guard,RUN,'initial')};disabled=not extra['initial_adapter_pass'];samplehash=hashlib.sha256();values=[];modes=collections.Counter();chain_updates=0;m.train();profile=[]
  with (RUN/'updates.jsonl').open('x') as log:
   for step in range(1,5001):
    guard.check();started=time.monotonic()
    if step==3000:
     extra['boundary_adapter_pass']=base.calibrate(m,raws,guard,RUN,'chain_boundary');disabled|=not extra['boundary_adapter_pass']
    if step%16==0:
     with torch.no_grad():av,ac=m.adapter_loss(raws[64:])
     passed=base.adapter_pass(av,ac);m.adapter_ready.fill_(passed and not disabled);extra['last_adapter_check']=dict(step=step,nmse=float(av),cosine=float(ac),passed=passed)
    ids=sampler.sample(32);samplehash.update(np.asarray(ids,dtype='<i8').tobytes());hs=sample_horizons(step);b=move(tr.batch(ids,hs),'cuda');ch=make_chain(tr,ids,(25,50)[step%2],step) if step>3000 and bool(m.adapter_ready) else None
    opt.zero_grad(set_to_none=True);v,mt=loss(m,b,step,ch);assert torch.isfinite(v),'Nonfinite loss';v.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn),'Nonfinite gradient';opt.step();torch.cuda.synchronize();mt.update(step=step,loss=float(v.detach()),seconds=time.monotonic()-started,horizons=list(hs));log.write(json.dumps(mt)+'\n');values.append(mt['loss']);modes[mt['mode']]+=1;chain_updates+=int(mt['chain_weight']>0);profile.append(mt['seconds'])
    if step==100:
     estimate=float(np.median(profile[-20:])*5000*1.45);atomic(RUN/'profile.json',dict(estimated_total_s_with_chain_margin=estimate,peak_mib=torch.cuda.max_memory_allocated()/2**20));assert estimate<10800,'Projected training exceeds reserved GPU budget'
    if step==1 or step%50==0:
     log.flush();atomic(RUN/'status.json',dict(status='training',step=step,elapsed_s=guard.check(),loss=mt['loss'],brt=brt()))
    if step%500==0:print('train',step,round(mt['loss'],5),round(guard.check(),1),flush=True)
  cp=RUN/'ckpt_05000.pt';torch.save(dict(model=m.state_dict(),step=5000,config=cfg,extra=extra,sampler=sampler.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),cp)
  result=dict(status='completed',steps=5000,checkpoint=str(cp),sha256=sha(cp),sampler=sampler.state_dict(),sample_sha256=samplehash.hexdigest(),modes=dict(modes),chain_updates=chain_updates,gate_disabled=disabled,extra=extra,first32=float(np.mean(values[:32])),last32=float(np.mean(values[-32:])),peak_mib=torch.cuda.max_memory_allocated()/2**20,elapsed_s=guard.check(),brt=brt())
  atomic(RUN/'train_summary.json',result);atomic(RUN/'status.json',result);atomic(ART/'TRAIN_COMPLETE.json',result);report('## 固定5000步训练完成\n'+json.dumps(result,ensure_ascii=False));del m,opt;torch.cuda.empty_cache()
 finally:
  if guard:guard.finish()

def main():
 atomic(ART/'STATUS.json',dict(status='CPU_preparing',brt=brt()));base.configure();_,_,tr,_=load();audit(tr);contracts(tr)
 report('### 训练前冻结实现口径\nH50之后保持末计划值；训练sync/independent用当前噪声动作迭代构造FK条件，clean用GT，action_only删除W及机器人token。W-only链用原计划后缀+保持，不延长读取实测动作。短W与A双向、长W沿用v3.1禁止读A；O不读A/R/W。原3000/1000/.1链日程，只有含W模式执行链；phase无合法监督保持0，既有训练文本向量作alignment辅助。dev固定P3 V3 48窗/6parent（不借用E93新增窗口）；+100仅原缓存有效窗口评分，不能将E98-W 27parent直接作本轮配对基线。')
 atomic(ART/'STATUS.json',dict(status='training',brt=brt()));train(tr);atomic(ART/'STATUS.json',dict(status='trained',brt=brt()))
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(ART/'STATUS.json',dict(status='stopped',traceback=traceback.format_exc(),brt=brt()));raise
