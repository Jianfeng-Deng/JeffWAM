import json,sys,collections
import numpy as np
from e103d_runtime import *
from e104p_metrics import worlds,action_summary,bootstrap,quant
GROUPS=('A2','F0','F2','S15000','S20000');STEPS=(2000,5000,10000,15000,20000);METHODS=('gt_action','pred_N1','no_traj','oracle')
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return not isinstance(x,float) or np.isfinite(x)
def main():
 lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());mapping={e['index']:e for e in lock['entries']};raw={g:json.loads((CONTROL/('eval_'+g)/'evaluation.json').read_text()) for g in GROUPS};summary={};pairs={}
 for g,x in raw.items():
  assert finite(x);assert x['checkpoint_sha']==sha(Path(lock['checkpoints'][g]));assert len(x['actions'])==211 and len(x['world'])==3364;summary[g]={}
  for scope in ('all27','original48'):
   keep=lambda r:scope=='all27' or mapping[r['index']]['original48'];aa=list(filter(keep,x['actions']));ww=list(filter(keep,x['world']));summary[g][scope]=dict(actions=action_summary(aa),world=worlds(ww))
 # Parent-resampled L1 and relative-copy intervals for every fixed point.
 for g in GROUPS:
  for scope in ('all27','original48'):
   aa=[r for r in raw[g]['actions'] if scope=='all27' or mapping[r['index']]['original48']];ps=sorted({r['parent'] for r in aa});idx=bootstrap(len(ps))
   for h in ('8','50'):
    l=np.array([np.mean([r['joints'][h]['l1'] for r in aa if r['parent']==p]) for p in ps]);c=np.array([np.mean([r['joints'][h]['copy_l1'] for r in aa if r['parent']==p]) for p in ps]);summary[g][scope]['actions']['joints'][h].update(l1_ci95=quant(l[idx].mean(1)),relative_copy=float(1-l.mean()/c.mean()),relative_copy_ci95=quant(1-l[idx].mean(1)/c[idx].mean(1)))
 # Include A0 reference, evaluation is inherited immutable E102-X output; never checkpoint selection.
 a0path=T/'results/e102x_extended_s90_20261115_01/A0/evaluation.json'
 if not a0path.exists():
  candidates=list((T/'results/e102x_extended_s90_20261115_01').rglob('evaluation.json'));a0path=next(p for p in candidates if json.loads(p.read_text())['group']=='A0')
 raw['A0']=json.loads(a0path.read_text());assert finite(raw['A0'])
 comparisons=list(zip(GROUPS[1:],GROUPS[:-1]))+[(g,'A0') for g in GROUPS]
 for target,base in comparisons:
  key=target+'-'+base;pairs[key]={}
  for scope in ('all27','original48'):
   keep=lambda r:scope=='all27' or mapping[r['index']]['original48'];p=dict(world={},actions={})
   for h in (10,25,50,100):
    p['world'][str(h)]={}
    for method in METHODS:
     rr={g:[r for r in raw[g]['world'] if keep(r) and r['h']==h and r['method']==method] for g in (target,base)};ps=sorted({r['parent'] for r in rr[base]});idx=bootstrap(len(ps));a={g:np.array([[np.mean([r[k] for r in rr[g] if r['parent']==pa]) for k in ('model','copy')] for pa in ps]) for g in rr};assert np.array_equal(a[target][:,1],a[base][:,1]);d=a[base][:,0]-a[target][:,0];copy=a[base][:,1];p['world'][str(h)][method]=dict(delta=float(d.mean()/copy.mean()),ci95=quant(d[idx].mean(1)/copy[idx].mean(1)),parents=len(ps))
   aa={g:list(filter(keep,raw[g]['actions'])) for g in (target,base)};ps=sorted({r['parent'] for r in aa[base]});idx=bootstrap(len(ps))
   def values(g):
    result={}
    for h in ('8','50'):
     l=np.array([np.mean([r['joints'][h]['l1'] for r in aa[g] if r['parent']==pa]) for pa in ps]);c=np.array([np.mean([r['joints'][h]['copy_l1'] for r in aa[g] if r['parent']==pa]) for pa in ps]);result['l1_'+h]=l;result['relative_copy_'+h]=(l,c)
    result['state_accuracy']=np.array([np.mean([r['state_accuracy'] for r in aa[g] if r['parent']==pa]) for pa in ps]);act=action_summary(aa[g])
    for tol in ('5','10'):
     pm=np.array(act['events'][tol]['parent_metrics']);result['f1_'+tol]=pm[:,2];result['false_switches_'+tol]=pm[:,3]
    return result
   a=values(target);b=values(base)
   for metric in a:
    if isinstance(a[metric],tuple):
     l,c=a[metric];bl,bc=b[metric];delta=bl.mean()/bc.mean()-l.mean()/c.mean();bs=bl[idx].mean(1)/bc[idx].mean(1)-l[idx].mean(1)/c[idx].mean(1)
    else:d=a[metric]-b[metric];delta=d.mean();bs=d[idx].mean(1)
    p['actions'][metric]=dict(delta=float(delta),ci95=quant(bs))
   pairs[key][scope]=p
 atomic(CONTROL/'SUMMARY.json',summary);atomic(CONTROL/'PAIRED.json',pairs)
 # Full curves with confidence intervals; static scientific artifact.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':9})
 for scope in ('all27','original48'):
  fig,axes=plt.subplots(3,4,figsize=(17,11));axes=axes.ravel()
  for j,h in enumerate(('10','25','50','100')):
   ax=axes[j]
   for method in METHODS:
    y=np.array([summary[g][scope]['world'][h]['skill'][method] for g in GROUPS])*100;ci=np.array([summary[g][scope]['world'][h]['skill_ci95'][method] for g in GROUPS])*100;ax.plot(STEPS,y,'o-',label=method);ax.fill_between(STEPS,ci[:,0],ci[:,1],alpha=.12)
   ax.set_title('World skill +'+h);ax.set_ylabel('%');ax.legend(fontsize=7)
  metrics=['l1_8','l1_50','relative_copy_8','relative_copy_50','state_accuracy','f1_5','f1_10','false_switches']
  for ax,metric in zip(axes[4:],metrics):
   if metric=='false_switches':
    for tol in ('5','10'):ax.plot(STEPS,[summary[g][scope]['actions']['events'][tol]['false_switches_per_minute'] for g in GROUPS],'o-',label=tol)
    ax.legend()
   else:
    yy=[];cc=[]
    for g in GROUPS:
     act=summary[g][scope]['actions']
     if metric.startswith('l1'):v=act['joints'][metric.split('_')[1]];y=v['l1'];ci=v['l1_ci95']
     elif metric.startswith('relative'):v=act['joints'][metric.split('_')[-1]];y=v['relative_copy'];ci=v['relative_copy_ci95']
     elif metric.startswith('f1'):v=act['events'][metric.split('_')[1]];y=v['f1'];ci=v['f1_ci95']
     else:v=act['state_accuracy'];y=v['mean'];ci=v['ci95']
     yy.append(y);cc.append(ci)
    ci=np.array(cc);ax.plot(STEPS,yy,'o-');ax.fill_between(STEPS,ci[:,0],ci[:,1],alpha=.2)
   ax.set_title(metric)
  for ax in axes:ax.grid(alpha=.25);ax.set_xlabel('Reported training step (F2 Piper local)')
  fig.suptitle(scope+'; parent bootstrap 2000, seed90; fixed endpoints');fig.tight_layout();fig.savefig(CONTROL/('convergence_'+scope+'.png'),dpi=150);fig.savefig(CONTROL/('convergence_'+scope+'.pdf'));plt.close(fig)
 # Loss history is on-policy mixed modes; do not compare individual noisy steps to dev directly.
 oldrun=T/'results/e104p_f2_s90_20261115_03';newrun=T/'results/e106c_f2_s90_20261115_01';updates=[json.loads(l) for p in (oldrun/'piper_updates.jsonl',newrun/'piper_updates.jsonl') for l in p.read_text().splitlines()];lossblocks=[]
 for end in (5000,10000,15000,20000):
  rr=[r for r in updates if end-1000<r['step']<=end];lossblocks.append(dict(end=end,n=len(rr),loss_mean=float(np.mean([r['loss'] for r in rr])),p1_mean=float(np.mean([r['p1_loss'] for r in rr]))))
 atomic(CONTROL/'TRAIN_LOSS.json',lossblocks);fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].plot([r['step'] for r in updates],[r['loss'] for r in updates],alpha=.2,label='logged loss');axes[0].plot([r['end'] for r in lossblocks],[r['loss_mean'] for r in lossblocks],'o-',label='last 1000 steps mean');axes[0].legend();axes[0].set_title('Mixed-mode train loss')
 for h in ('10','25','50','100'):axes[1].plot(STEPS,[summary[g]['all27']['world'][h]['skill']['no_traj']*100 for g in GROUPS],'o-',label='+'+h)
 axes[1].legend();axes[1].set_title('Frozen dev no_traj skill %');fig.tight_layout();fig.savefig(CONTROL/'train_vs_dev.png',dpi=150);plt.close(fig)
 lines=['## 五个固定点结果与收敛曲线','计步说明：A2=原2000步；F0/F2/15000/20000为从A2开始的Piper阶段累计步数（对应连续AdamW动作头总步数7000/12000/17000/22000）。横轴遵照用户指定2000/5000/10000/15000/20000，首段不是统一累计优化器步数。主27parent；+100为208窗/26parent。附48窗/6parent。无dev模型语言，默认仍为A2。','图：'+str(CONTROL/'convergence_all27.png')+'；附表图 convergence_original48.png；训练/dev图 train_vs_dev.png。完整全部指标CI及相邻点配对CI：SUMMARY.json / PAIRED.json。']
 for scope in ('all27','original48'):
  lines+=['\n### '+scope,'|步数|h|gt_action / pred / no_traj / oracle skill %|','|---|---|---|']
  for step,g in zip(STEPS,GROUPS):
   for h in ('10','25','50','100'):lines.append(f'|{step}|{h}|'+ '/'.join(f"{100*summary[g][scope]['world'][h]['skill'][m]:.3f}" for m in METHODS)+'|')
  lines+=['|步数|前8/H50 L1|复制相对改善% 前8/H50|状态准确率|事件F1 ±5/±10|假切换/分钟 ±5/±10|','|---|---|---|---|---|---|']
  for step,g in zip(STEPS,GROUPS):
   a=summary[g][scope]['actions'];lines.append(f'|{step}|'+ '/'.join(f"{a['joints'][h]['l1']:.6f}" for h in ('8','50'))+'|'+ '/'.join(f"{100*a['joints'][h]['relative_copy']:.2f}" for h in ('8','50'))+f"|{a['state_accuracy']['mean']:.4f}|"+ '/'.join(f"{a['events'][h]['f1']:.4f}" for h in ('5','10'))+'|'+ '/'.join(f"{a['events'][h]['false_switches_per_minute']:.3f}" for h in ('5','10'))+'|')
 lines+=['## 相邻点与A0参考（主集，百分点pp及95%CI）','|比较|h|no_traj差 [CI]|gt_action差 [CI]|','|---|---|---|---|']
 for key,p in pairs.items():
  for h in ('10','25','50','100'):
   fmt=lambda m:f"{100*p['all27']['world'][h][m]['delta']:+.3f} ["+','.join(f'{100*v:+.3f}' for v in p['all27']['world'][h][m]['ci95'])+']';lines.append(f'|{key}|{h}|{fmt("no_traj")}|{fmt("gt_action")}|')
 lines+=['## 训练损失与dev描述',json.dumps(lossblocks,ensure_ascii=False),'训练loss为固定每50步日志的末1000步均值，混合模式比例及样本难度有噪声，不能单凭总loss认定过拟合。仅观察固定终点dev变化，不据此选点。','## 完整性与资源终审']
 # Exact evaluator source equality with inherited run and five points.
 evalhash={g:sha(CONTROL/('eval_'+g)/'evaluator.py') for g in GROUPS};oldhash=sha(T/'results/e104p_control_20261115_03/eval_A2/evaluator.py');assert len(set(evalhash.values()))==1 and next(iter(evalhash.values()))==oldhash
 for k in ('A2','F0','F2'):
  previous=json.loads((T/'results/e104p_control_20261115_03'/('eval_'+k)/'evaluation.json').read_text());assert previous['actions']==raw[k]['actions'] and previous['world']==raw[k]['world']
 for name,h in lock['code_hashes'].items():assert sha(ROOT/name)==h,name
 assert sha(Path(lock['checkpoints']['F2']))==lock['resume_sha'];assert all(json.loads(p.read_text())['status']=='closed' for p in CONTROL.glob('gpu_*.json'));assert used_seconds()<=7200
 import torch
 audit={}
 for g,step in [('S15000',17000),('S20000',22000)]:
  cp=torch.load(lock['checkpoints'][g],map_location='cpu',weights_only=False);assert cp['step']==step-2000;assert cp['sampler']['count']==(step-2000)*32 and cp['p1sampler']['count']==(step-2000)*32;names=list(cp['model']);# parameter IDs ordered by model.named_parameters, use model on CPU
  from e103d_model import WorldModel
  from e99w2_data import load
  _,_,tr,_=load();m=WorldModel(tr.norm);m.load_state_dict(cp['model'],strict=True);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(cp['optimizer']);assert int(opt.state[m.delta_head.weight]['step'])==step
  assert all(torch.isfinite(x).all() for x in cp['model'].values());audit[g]=dict(sha=sha(Path(lock['checkpoints'][g])),delta_head_optimizer_step=step,p3_draws=cp['sampler']['count'],p1_draws=cp['p1sampler']['count'])
 size=sum(p.stat().st_size for root in (ROOT,CONTROL,newrun) for p in root.rglob('*') if p.is_file());assert size<9_900_000_000;out=dict(status='complete',GPUh_upper=used_seconds()/3600,new_bytes=size,evaluator_identical=True,evaluator_sha=oldhash,old_points_replay_exact=True,checkpoints=audit,default='A2',brt=brt());atomic(CONTROL/'FINAL_AUDIT.json',out);lines+=[json.dumps(out,ensure_ascii=False),'命令：nohup e29/bin/python -B -u code_e106c/e106c_pipeline.py（setsid/start_new_session=True）；launcher.pid与PIPELINE.json记录状态。新增特征0字节；旧路径仅只读。未完成：无。需要用户决定：无；默认A2由Claude决定是否更改，本轮不更改。'];report('\n'.join(lines))
 for path in (T/'docs/experiment_log/EXPERIMENT_LOG.md',T/'notes/worklog.md'):
  with path.open('a') as f:f.write('\n- E106-C | '+brt()+' BRT | COMPLETE | F2同流续至20000，15000/20000固定ckpt及五点曲线；code_e106c / results/e106c_control_20261115_01；GPUh='+str(out['GPUh_upper'])+'；新字节='+str(size)+'；默认A2未改；详见 notes/e106c_longer_training_20261115.md。\n')
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
