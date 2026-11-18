import os,sys,inspect,json,traceback,numpy as np,torch
from e103d_runtime import *
from e102x_setup import datasets
import e102a_eval as ev
from e107m_model import WorldModel,StochasticWorld
from e100c_chain import open_second
from e86.data import move

def sample_stats(S,Y):
 # S[K,B,T,D], one window and one horizon; standardized feature space.
 a=S.flatten(1).double();y=Y.flatten().double();dist=lambda v:v.square().mean(-1).clamp_min(1e-12).sqrt()
 energy=dist(a-y).mean()-.5*dist(a[:,None]-a[None,:]).mean()
 crps=(a-y).abs().mean()-.5*(a[:,None]-a[None,:]).abs().mean()
 mse=(a-y).square().mean(-1);mu=a.mean(0)
 return dict(energy=float(energy),crps=float(crps),best_mse=float(mse.min()),mean_mse=float((mu-y).square().mean()),spread2=float((a-mu).square().mean()),coverage=float(((y>=a.min(0).values)&(y<=a.max(0).values)).double().mean()))
def evaluate(group):
 lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());CKPT=Path(lock['checkpoints'][group]);ART=CONTROL/('eval_'+group+'_02');ART.mkdir(exist_ok=False);arrays={};extra=[];chains=[]
 stochastic=group=='M1'
 class EvalModel(StochasticWorld if stochastic else WorldModel):
  def __init__(self,norm):super().__init__(norm)
 def capture(m,bb,bh,i,h,method,parent):
  z=m.forward_mean(bb)['mean'];S=m.samples(z,4,90+i*101+h).transpose(0,1) if stochastic else z[None].expand(4,-1,-1,-1)
  assert torch.isfinite(S).all() and torch.isfinite(z).all();P=S.mean(0);Y=bh['world'];c=bh['world_current'];scale=bh['world_scale'];key=f'{i}_{h}_{method}'
  stats=sample_stats(S,Y);stats.update(index=i,parent=parent,h=h,method=method,zero_mse=float((z.double()-Y.double()).square().mean()),copy=float(Y.double().square().mean()));extra.append(stats)
  if method in ('no_traj','gt_action'):
   pool=lambda A:A.reshape(-1,3,(8 if h<=50 else 1),768).mean(2).flatten(1).cpu().numpy()
   arrays[key+'_samples_pool']=pool(c[None]+S[:,0]*scale[None])
   if stochastic:arrays[key+'_samples_raw']=(c[None]+S[:,0]*scale[None]).reshape(4,*c.shape).cpu().numpy()
   arrays[key+'_zero']=(c+z*scale)[0].cpu().numpy();arrays[key+'_mean']=(c+P*scale)[0].cpu().numpy()
   arrays[f'{i}_{h}_Y']=bh['world_raw'][0].cpu().numpy();arrays[f'{i}_{h}_C']=c[0].cpu().numpy()
  return P.double()
 def chain_eval(m,dev,entries):
  for e in entries:
   i=e['index']
   for k in (25,50):
    if 2*k not in e['horizons']:continue
    first=move(dev.batch([i],(k,),'short'),'cuda');direct=move(dev.batch([i],(2*k,),'short'),'cuda');Y=direct['world_raw'];scale=direct['world_scale']
    def raw(bb,seed,sample=None):
     zz=m.forward_mean(bb)['mean']
     if stochastic:
      ss=m.samples(zz,4,seed);zz=ss.mean(1) if sample is None else ss[:,sample]
     return bb['world_current']+zz*bb['world_scale']
    # E100C open_second uses only original observed <=t context and original plan.
    def second_from(z):
     b=open_second(first,k);b['reentry_tokens']=m.reentry(z.reshape(1,3,8,768));b['world_current']=z.reshape(1,24,768);return b
    z=raw(first,90);end=raw(second_from(z),91)
    if k==50:end=end.reshape(1,3,8,768).mean(2)
    deterministic=float(((end-Y)/scale).double().square().mean());ends=[];errs=[]
    for j in range(4):
     z=raw(first,90+i*101+k,j);ep=raw(second_from(z),91+i*101+k,j)
     if k==50:ep=ep.reshape(1,3,8,768).mean(2)
     assert torch.isfinite(ep).all();ends.append(ep);errs.append(float(((ep-Y)/scale).double().square().mean()))
    endmean=torch.stack(ends).mean(0)
    chains.append(dict(index=i,parent=e['parent'],k=k,mean_rollout_mse=deterministic,sample_rollout_mse=float(np.mean(errs)),sample_ensemble_endpoint_mse=float(((endmean-Y)/scale).double().square().mean())))
 src=inspect.getsource(ev.evaluate).replace("assert len(f['train64'])==64 and len(f['dev'])==48","assert len(f['dev'])==len(lock['entries'])")
 start=src.index(' c0=json.loads(');end=src.index(' cp=torch.load',start);src=src[:start]+src[end:]
 src=src.replace("RUN/'ckpt_02000.pt'",'CKPT').replace("assert cp['step']==STEPS;",'')
 src=src.replace("[('dev',dev,f['dev']),('train64',tr,f['train64'])]","[('dev',dev,f['dev'])]")
 src=src.replace('for h in (25,50,100):',"for h in (10,25,50,100):\n       if h not in entry['horizons']:continue")
 src=src.replace('seq=qq[a+1:a+101];assert len(seq)==100','seq=qq[np.minimum(a+np.arange(1,101),len(qq)-1)]')
 src=src.replace('       sources=dict(gt_action','       feat[~valid.numpy()]=0\n       sources=dict(gt_action')
 src=src.replace("P=m.forward_mean(bb)['mean'].double()","P=capture(m,bb,bh,i,h,method,r['parent_component'])")
 start=src.index("   b=move(dev.batch([f['dev'][0]");end=src.index(' finally:guard.finish()',start);src=src[:start]+"   chain_eval(m,dev,lock['entries'])\n"+src[end:]
 src=src.replace("f=json.loads((CONTROL/'fixed_windows.json').read_text())","f=dict(dev=lock['entries'])")
 ns=dict(ev.__dict__);ns.update(CONTROL=CONTROL,ART=ART,Guard=lambda *a:Guard('eval_'+group+'_02',1800,6144),lock=lock,load=datasets,CKPT=CKPT,GROUP=group,WorldModel=EvalModel,capture=capture,chain_eval=chain_eval)
 (ART/'evaluator.py').write_text(src);exec(compile(src,str(ART/'evaluator.py'),'exec'),ns);ns['evaluate']()
 np.savez_compressed(ART/'vectors.npz',**arrays);atomic(ART/'probabilistic.json',extra);atomic(ART/'chains.json',chains);report('## '+group+'固定终点评估前向完成\n四条件四时域、动作、分布向量与开环两跳已保存；能量/CRPS使用K4有限样本V统计，best-of-K仅诊断；MHA fastpath禁用，非有限值拒绝。\n')
if __name__=='__main__':evaluate(sys.argv[1])
