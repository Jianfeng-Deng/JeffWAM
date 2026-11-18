import sys
sys.dont_write_bytecode=True
import collections,json,time,traceback,numpy as np,torch
from e99w2_common import *
from e99w2_data import load
from e99w2_model import WorldModel,trajectory
from e96a.train import configure
from e86.data import move,JOINTS,GRIPS
H=(10,25,50,100);METHODS=('oracle','gt_action','pred_N1','pred_N8','no_traj','noisy_gt');CAMS=('top','left_wrist','right_wrist')
def finite(x):
 if torch.is_tensor(x):assert torch.isfinite(x).all(),'Nonfinite tensor refused'
 else:assert np.isfinite(x).all(),'Nonfinite array refused'
 return x

def aggregate(records):
 groups=collections.defaultdict(list)
 for r in records:
  finite([r['model'],r['copy'],r['cosine']]);groups[r['h'],r['camera'],r['method']].append(r)
 arrays={};summary=[];paired=[];retention=[]
 for (h,camera,method),rr in sorted(groups.items()):
  by=collections.defaultdict(list)
  for r in rr:by[r['parent']].append([r['model'],r['copy'],r['cosine']])
  ps=sorted(by);a=np.array([np.mean(by[p],0) for p in ps]);arrays[h,camera,method]=(ps,a);v=a.mean(0);draw=a[np.random.default_rng(90).integers(len(a),size=(2000,len(a)))].mean(1)
  assert v[1]>0 and np.all(draw[:,1]>0),'Zero copy energy cannot be scored'
  finite(1-draw[:,0]/draw[:,1]);finite(v)
  summary.append(dict(h=h,camera=camera,method=method,windows=len(rr),parents=len(ps),mse=float(v[0]),copy_mse=float(v[1]),skill=float(1-v[0]/v[1]),ci95=np.quantile(1-draw[:,0]/draw[:,1],[.025,.975]).tolist(),cosine=float(v[2]),degenerate_parent_CI=len(ps)<2))
 for h,cam in sorted({(k[0],k[1]) for k in arrays}):
  ps,b=arrays[h,cam,'no_traj'];_,o=arrays[h,cam,'oracle'];ix=np.random.default_rng(90).integers(len(b),size=(2000,len(b)));den=(b[:,0]-o[:,0]).mean()
  for method in METHODS:
   if method=='no_traj':continue
   pp,a=arrays[h,cam,method];assert ps==pp and np.array_equal(a[:,1],b[:,1]);v=(b[:,0]-a[:,0]).mean()/b[:,1].mean();draw=(b[ix,0].mean(1)-a[ix,0].mean(1))/b[ix,1].mean(1)
   paired.append(dict(h=h,camera=cam,method=method,comparison=method+'-no_traj',delta_skill=float(v),ci95=np.quantile(draw,[.025,.975]).tolist(),parents=len(ps),degenerate_parent_CI=len(ps)<2))
   if method.startswith('pred'):
    dden=b[ix,0].mean(1)-o[ix,0].mean(1);dnum=b[ix,0].mean(1)-a[ix,0].mean(1)
    retention.append(dict(h=h,camera=cam,method=method,percent=float(100*(b[:,0]-a[:,0]).mean()/den) if den>0 else None,oracle_gain_positive=bool(den>0),ci95_percent=np.quantile(100*dnum/dden,[.025,.975]).tolist() if np.all(dden>0) else None))
 return summary,paired,retention

def action_record(pred,target,q,parent,method,index):
 rows=[]
 for h in (8,50):
  e=pred[:h]-target[:h];finite(e)
  rows.append(dict(parent=parent,index=index,method=method,h=h,joint_L1_rad=float(np.abs(e[:,JOINTS]).mean()),joint_RMSE_rad=float(np.sqrt((e[:,JOINTS]**2).mean())),gripper_L1_mm=float(np.abs(e[:,GRIPS]).mean()*1000),gripper_RMSE_mm=float(np.sqrt((e[:,GRIPS]**2).mean())*1000),left_joint_L1_rad=float(np.abs(e[:,:6]).mean()),right_joint_L1_rad=float(np.abs(e[:,7:13]).mean()),left_gripper_L1_mm=float(np.abs(e[:,6]).mean()*1000),right_gripper_L1_mm=float(np.abs(e[:,13]).mean()*1000)))
 return rows

def main():
 assert (ART/'TRAIN_COMPLETE.json').exists();out=ART/'evaluation_01';out.mkdir(exist_ok=False)
 with (out/'DEV_ONCE_LOCK.json').open('x') as f:json.dump(dict(brt=brt(),steps=5000,flow_steps=4,N=8,seed=90,world_only_queries=True,no_checkpoint_selection=True),f)
 configure();torch.backends.mha.set_fastpath_enabled(False);_,_,tr,dev=load();cp=torch.load(RUN/'ckpt_05000.pt',map_location='cpu',weights_only=False);assert cp['step']==5000;m=WorldModel(tr.norm);m.load_state_dict(cp['model']);del cp;guard=None;records=[];ar=[];timing=collections.defaultdict(list);coverage={};snap={}
 try:
  from e96a.resource import Guard
  guard=Guard('E99W2_fixed5000_eval',2400);m=m.to('cuda');m.eval()
  with torch.inference_mode():
   # Error amplitude fitted only on the predeclared 16 training parents used by adapter holdout.
   ids=[];seen=set()
   for i,r in enumerate(tr.rows):
    if r['parent_component'] not in seen:ids.append(i);seen.add(r['parent_component'])
    if len(ids)==80:break
   errors=[]
   for i in ids[64:]:
    guard.check();b=move(tr.batch([i],(10,25,50,100),'short'),'cuda');pred=m.infer(b,'action_only',4,90)[0];errors.append((pred-b['action'])[0].cpu().numpy())
   rms=np.sqrt(np.mean(np.square(errors),0));finite(rms);np.save(out/'train_only_noise_rms.npy',rms);atomic(out/'noise_calibration.json',dict(training_indices=ids[64:],train_parents=16,per_step_dimension_RMS=True,seed=90,no_dev_fitting=True,Gaussian_independent=True))
   for i,r in enumerate(dev.rows):
    guard.check();b=move(dev.batch([i],H,'short'),'cuda');target=b['action'][0].cpu().numpy();q=b['q_current'][0].cpu().numpy();parent=r['parent_component'];actions=[]
    for n in range(8):
     torch.cuda.synchronize();st=time.perf_counter();a,_=m.infer(b,'action_only',4,90+n);torch.cuda.synchronize();timing['action_only_N1'].append(time.perf_counter()-st);finite(a);actions.append(a)
    torch.cuda.synchronize();st=time.perf_counter();joint,jw=m.infer(b,'joint',4,90);torch.cuda.synchronize();timing['joint'].append(time.perf_counter()-st);finite(joint);finite(jw)
    predictions={'copy_current_state':np.broadcast_to(q,(50,14)),'action_only_N1':actions[0][0].cpu().numpy(),'action_only_N8_action_mean':torch.stack(actions).mean(0)[0].cpu().numpy(),'joint':joint[0].cpu().numpy()}
    for method,p in predictions.items():ar.extend(action_record(p,target,q,parent,method,i))
    oracle,ov=dev.oracle(i);z,qstate,_,good=dev.states.get(dev,r['canonical_id']);anchor=r['anchor'];idx=np.minimum(anchor+np.arange(101),len(qstate)-1);ok=(anchor+np.arange(101)<len(qstate)) & z['p1_source_mask'][idx];dt=np.diff(z['timestamp'][anchor:min(anchor+101,len(qstate))]);dtgood=np.ones(101,bool);dtgood[1:len(dt)+1]=(dt>0)&(dt<=.050001);ov=ov & torch.tensor(np.logical_and.accumulate(ok & dtgood));oracle[~ov]=0;noise=np.random.default_rng(990000+i).normal(size=(50,14))*rms;noisy=b['action']+torch.tensor(noise[None],device='cuda',dtype=torch.float32)
    for h in H:
     bh=move(dev.batch([i],(h,),'short'),'cuda')
     if not bool(bh['world_valid'].all()):continue
     assert bool(ov[h]),'Oracle missing at valid image horizon';coverage.setdefault(str(h),[]).append(dict(index=i,parent=parent));Y=bh['world'].cpu().numpy();finite(Y)
     sources={'oracle':dict(bh,robot=oracle[None].to('cuda'),robot_valid=ov[None].to('cuda')),'gt_action':bh,'no_traj':dict(bh,robot=torch.zeros_like(bh['robot']),robot_valid=torch.zeros_like(bh['robot_valid'])),'noisy_gt':dict(bh,robot=trajectory(bh['context'],noisy))}
     pp={method:finite(m.forward_mean(bb)['mean']).cpu().numpy() for method,bb in sources.items()};samples=[]
     for a in actions:samples.append(finite(m.forward_mean(dict(bh,robot=trajectory(bh['context'],a)))['mean']).cpu().numpy())
     pp['pred_N1']=samples[0];pp['pred_N8']=np.mean(samples,0);nt=Y.shape[1]//3
     snap[f'{i}_h{h}_target']=Y.astype('f4')
     for method,P in pp.items():
      finite(P);snap[f'{i}_h{h}_{method}']=P.astype('f4')
      for ci,cam in [(None,'all'),*enumerate(CAMS)]:
       sl=slice(None) if ci is None else slice(ci*nt,(ci+1)*nt);y=Y[:,sl].astype('f8');p=P[:,sl].astype('f8');err=float(((p-y)**2).mean());copy=float((y*y).mean());cos=float((p*y).sum()/max(np.sqrt((p*p).sum()*(y*y).sum()),1e-12));records.append(dict(index=i,parent=parent,h=h,camera=cam,method=method,model=err,copy=copy,cosine=cos))
    if i%8==0:print('eval',i,'/',len(dev.rows),flush=True)
   torch.cuda.synchronize()
  del m;torch.cuda.empty_cache();guard.finish();guard=None
 except BaseException:
  if guard:guard.finish()
  raise
 summary,paired,retention=aggregate(records);atomic(out/'world_records.json',records);atomic(out/'world_summary.json',summary);atomic(out/'world_paired.json',paired);atomic(out/'retention.json',retention);atomic(out/'coverage.json',{h:dict(windows=len(rs),parents=len({r['parent'] for r in rs}),indices=[r['index'] for r in rs]) for h,rs in coverage.items()});np.savez_compressed(out/'vectors.npz',**snap)
 action=[];metrics=[k for k in ar[0] if k not in ('parent','index','method','h')]
 for method in sorted({r['method'] for r in ar}):
  for h in (8,50):
   rs=[r for r in ar if r['method']==method and r['h']==h];ps=sorted({r['parent'] for r in rs});row=dict(method=method,h=h,windows=len(rs),parents=len(ps))
   for k in metrics:row[k]=float(np.mean([np.mean([r[k] for r in rs if r['parent']==p]) for p in ps]))
   action.append(row)
 atomic(out/'action_records.json',ar);atomic(out/'action_summary.json',action);atomic(out/'latency.json',{k:dict(count=len(v),median_ms=float(np.median(v)*1000),p95_ms=float(np.quantile(v,.95)*1000),includes='batch1 CUDA synchronized model inference including FK and final W prediction for joint; excludes data I/O') for k,v in timing.items()})
 decisions={str(h):next(r for r in paired if r['h']==h and r['camera']=='all' and r['method']=='pred_N1') for h in (25,50,100) if str(h) in coverage};passed=len(decisions)==3 and all(r['ci95'][0]>0 for r in decisions.values());decision=dict(deployable_condition_effective=bool(passed),by_h=decisions,limited_parent_coverage=any(r['parents']<2 for r in decisions.values()),no_selection=True,checkpoint_sha256=sha(RUN/'ckpt_05000.pt'),brt=brt());atomic(out/'decision.json',decision);atomic(out/'COMPLETE.json',dict(status='completed',brt=brt(),finite_rejection=True,MHA_fastpath=False));atomic(ART/'STATUS.json',dict(status='evaluated',brt=brt()));report('## 固定终点评估完成\n'+json.dumps(decision,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(ART/'evaluation_failure.json',dict(brt=brt(),traceback=traceback.format_exc()));raise
