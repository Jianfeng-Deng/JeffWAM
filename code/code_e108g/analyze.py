import os,sys,json,time,collections,traceback
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR='/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e108g_diag_20261116_01/mplconfig',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.dont_write_bytecode=True
import numpy as np
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e108g_diag_20261116_01';R=T/'notes/e108g_gripper_diagnosis_20261116.md'
sys.path.insert(0,str(T/'code_e106c'))
from e102a_eval import events,match

def put(n,x):
 p=O/n;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,p)
def report(s):
 with R.open('a') as f:f.write('\n'+s+'\n')
def pairs(pred,true,tol):
 # Same traversal, objective, and stable ties as original DP. Return indices for attribution.
 n,m=len(pred),len(true);dp=[[None]*(m+1) for _ in range(n+1)];dp[0][0]=(0,0,[])
 def better(old,new):return new if old is None or (new[0],-new[1])>(old[0],-old[1]) else old
 for i in range(n+1):
  for j in range(m+1):
   v=dp[i][j]
   if v is None:continue
   if i<n:dp[i+1][j]=better(dp[i+1][j],v)
   if j<m:dp[i][j+1]=better(dp[i][j+1],v)
   if i<n and j<m and pred[i][1]==true[j][1] and abs(pred[i][0]-true[j][0])<=tol:
    dp[i+1][j+1]=better(dp[i+1][j+1],(v[0]+1,v[1]+abs(pred[i][0]-true[j][0]),v[2]+[(i,j)]))
 out=dp[n][m][2];assert [pred[i][0]-true[j][0] for i,j in out]==match(pred,true,tol)
 return out

def decode(p,initial,cfg):
 family=cfg['family'];low=cfg.get('low',.5);high=cfg.get('high',.5);d=cfg.get('d',0);alpha=cfg.get('alpha',1.)
 s=np.array(initial,dtype='i1');last=np.full(2,-100);ema=s.astype(float);out=[]
 for t,v in enumerate(p,1):
  if family=='ema':ema=alpha*v+(1-alpha)*ema;v=ema
  candidate=np.where(s==0,v>high,v>low).astype('i1')
  change=(candidate!=s)&(t-last>=d);s=np.where(change,candidate,s).astype('i1');last=np.where(change,t,last);out.append(s.copy())
 return np.array(out)

def eventrow(r,state,tol,detail=False):
 gt=np.array(r['labels']);pr=[];truth=[];errs=[];details=[]
 for k in range(2):
  pe=events(state[:,k],r['current'][k]);te=events(gt[:,k],r['label_init'][k]);mm=pairs(pe,te,tol) if detail else None
  ee=[pe[i][0]-te[j][0] for i,j in mm] if detail else match(pe,te,tol)
  pr+=pe;truth+=te;errs+=ee
  if detail:details.append((pe,te,mm))
 return dict(tp=len(errs),pred=len(pr),true=len(truth),errors=errs,details=details)

def score(rows,states,tols=(5,10,20)):
 parents=sorted({r['parent'] for r in rows});raw={tol:{p:dict(tp=0,pred=0,true=0,minutes=0,errors=[]) for p in parents} for tol in tols}
 for r,s in zip(rows,states):
  for tol in tols:
   e=eventrow(r,s,tol);v=raw[tol][r['parent']]
   for key in ('tp','pred','true'):v[key]+=e[key]
   v['minutes']+=50/30/60;v['errors']+=e['errors']
 out={}
 for tol,pp in raw.items():
  metrics=[];errors=[]
  for par,v in pp.items():
   tp,pr,gt=v['tp'],v['pred'],v['true'];metrics.append([2*tp/(pr+gt) if pr+gt else 0,(pr-tp)/v['minutes'],tp/gt if gt else 0]);errors+=v['errors']
  arr=np.array(metrics);out[str(tol)]=dict(f1=float(arr[:,0].mean()),false_per_min=float(arr[:,1].mean()),recall=float(arr[:,2].mean()),offset_median=float(np.median(errors)) if errors else None,offset_q25_q75=np.quantile(errors,[.25,.75]).tolist() if errors else None,matched=len(errors),true=sum(v['true'] for v in pp.values()),pred=sum(v['pred'] for v in pp.values()),parent_metrics=metrics,parent_raw=pp)
 return out

def test_causality(configs):
 p=np.random.default_rng(90).random((50,2));initial=[0,1]
 for cfg in configs:
  full=decode(p,initial,cfg)
  for t in (1,5,13,27,49):
   q=p.copy();q[t:]=1-q[t:];assert np.array_equal(full[:t],decode(q,initial,cfg)[:t]);assert np.array_equal(full[:t],decode(p[:t],initial,cfg))

def bootstrap(new,base):
 out={}
 for tol,v in new.items():
  a=np.array(v['parent_metrics']);b=np.array(base[tol]['parent_metrics']);assert list(v['parent_raw'])==list(base[tol]['parent_raw']);idx=np.random.default_rng(90).integers(len(a),size=(2000,len(a)));diff=a-b;di={}
  for j,k in enumerate(('f1','false_per_min','recall')):di[k]=dict(delta=float(diff[:,j].mean()),ci95=np.quantile(diff[idx,j].mean(1),[.025,.975]).tolist())
  # Overall matched-event median, with parents jointly resampled; undefined if either sample has no match.
  pars=list(v['parent_raw']);tim=[]
  for ids in idx:
   ne=sum([v['parent_raw'][pars[i]]['errors'] for i in ids],[]);be=sum([base[tol]['parent_raw'][pars[i]]['errors'] for i in ids],[])
   if ne and be:tim.append(float(np.median(ne)-np.median(be)))
  di['offset_median']=dict(delta=v['offset_median']-base[tol]['offset_median'] if v['offset_median'] is not None and base[tol]['offset_median'] is not None else None,ci95=np.quantile(tim,[.025,.975]).tolist() if tim else None,valid_bootstraps=len(tim));out[tol]=di
 return out

def g0(rows,states,subset):
 baseline=score(rows,states,(3,5,10,20));cells=collections.defaultdict(lambda:dict(tp=0,fn=0,fp=0,parents=set(),event_parents=set()));fps=[];offsets={str(t):[] for t in (3,5,10,20)};boundary=0
 for r,s in zip(rows,states):
  width=np.array(r['width']);p=np.array(r['prob']);boundary+=int(np.any(np.array(r['current'])!=np.array(r['label_init'])))
  for tol in (3,5,10,20):offsets[str(tol)]+=eventrow(r,s,tol)['errors']
  for tol in (5,10):
   for k,(pe,te,mm) in enumerate(eventrow(r,s,tol,True)['details']):
    usedp={i for i,j in mm};usedt={j for i,j in mm}
    for threshold in (.1,.5,1.):
     delta=np.abs(np.diff(width[:,k]))*1000
     for segment in ('all','static','moving'):
      key=(tol,k,threshold,segment);v=cells[key];v['parents'].add(r['parent'])
      accept=lambda t:segment=='all' or (delta[t-1]<=threshold if segment=='static' else delta[t-1]>threshold)
      for j,(t,d) in enumerate(te):
       if accept(t):v['tp' if j in usedt else 'fn']+=1;v['event_parents'].add(r['parent'])
      for i,(t,d) in enumerate(pe):
       if i not in usedp and accept(t):v['fp']+=1
    if tol==10:
     for i,(t,d) in enumerate(pe):
      if i in usedp:continue
      nxt=pe[i+1][0] if i+1<len(pe) else None;lo=max(1,t-5);hi=min(50,t+10);local=p[lo-1:hi,k];ls=(local>.5).astype(int)
      crossings=np.flatnonzero(ls[1:]!=ls[:-1]);near_cross=sum(abs(local[j]-.5)<=.1 and abs(local[j+1]-.5)<=.1 for j in crossings)
      fps.append(dict(parent=r['parent'],index=r['index'],gripper=k,time=t,direction=d,return_frames=nxt-t if nxt else None,censored=nxt is None,static_mm={str(th):bool(abs(width[t,k]-width[t-1,k])*1000<=th) for th in (.1,.5,1.)},boundary=t==1,initial_label_mismatch=r['current'][k]!=r['label_init'][k],prob_before=float(p[t-2,k]) if t>1 else float(r['current'][k]),prob_at=float(p[t-1,k]),near_crossings_local=int(near_cross),crossings_local=len(crossings),trace_start=lo,trace=local.tolist()))
 cc=[]
 for (tol,k,th,seg),v in sorted(cells.items()):cc.append(dict(tol=tol,gripper=k,static_threshold_mm=th,segment=seg,tp=v['tp'],fn=v['fn'],fp=v['fp'],true=v['tp']+v['fn'],parents=len(v['parents']),event_parents=len(v['event_parents']),small_sample=v['tp']+v['fn']<20 or len(v['event_parents'])<5))
 short={str(d):sum(f['return_frames'] is not None and f['return_frames']<=d for f in fps) for d in (2,3,5,10)}
 summary=dict(n=len(fps),return_within=short,censored=sum(f['censored'] for f in fps),boundary_first=sum(f['boundary'] for f in fps),boundary_and_mismatch=sum(f['boundary'] and f['initial_label_mismatch'] for f in fps),static={str(th):sum(f['static_mm'][str(th)] for f in fps) for th in (.1,.5,1.)},near_threshold_crossing=sum(abs(f['prob_before']-.5)<=.1 and abs(f['prob_at']-.5)<=.1 for f in fps),window_initial_label_mismatch=boundary)
 put('G0_'+subset+'.json',dict(baseline=baseline,cells=cc,offsets=offsets,false_events=fps,false_summary=summary))
 lines=['## G0 '+subset+' 拆分、容差与偏移','阶段：原冻结窗口无人工任务阶段字段；不以模型阶段预测分层。静止/运动为该爪实测相邻帧开度差，非整臂任务阶段。TP/FN归于真事件时刻，FP归于预测时刻，先做完整窗口匹配再归类。各静止阈值仅描述并列，互不择优。','|容差|F1 parent均值|召回|假检/分钟|TP/真事件/预测|偏移Q25/median/Q75帧|','|---|---|---|---|---|---|']
 for tol,v in baseline.items():lines.append(f"|±{tol}|{v['f1']:.6f}|{v['recall']:.6f}|{v['false_per_min']:.3f}|{v['matched']}/{v['true']}/{v['pred']}|{v['offset_q25_q75']}/{v['offset_median']}|")
 lines+=['|tol|侧|静止阈值mm|分层|TP|FN|FP|真事件|暴露parent/有真事件parent|小样本|','|---|---|---|---|---|---|---|---|---|---|']
 for v in cc:
  if v['segment']=='all' and v['static_threshold_mm']!=.5:continue
  lines.append('|'+ '|'.join(map(str,[v['tol'],'左' if v['gripper']==0 else '右',v['static_threshold_mm'],v['segment'],v['tp'],v['fn'],v['fp'],v['true'],f"{v['parents']}/{v['event_parents']}",v['small_sample']]))+'|')
 lines+=['小样本标志：真事件<20或有真事件parent<5；这些格子只报数量，不推断阶段或侧别优劣。左右/静止表是事件计数，主F1仍按parent等权；重叠窗口事件可重复。', '假检诊断（±10）：'+json.dumps(summary,ensure_ascii=False),'局部概率轨迹使用[t−5,t+10]仅作G0回顾诊断，绝不输入G1解码。寿命仅能看到50帧窗口内下一次回切；尾部截尾不当作长寿命，也不当作短寿命。']
 report('\n'.join(lines));return baseline,offsets,fps

def labelcheck(rows):
 # Use measured feedback states as a crossing-vs-onset diagnostic, preserving original onset test truth.
 obs=[];lags=[];alltrue=[]
 c=json.loads((T/'results/e106c_control_20261115_02/GRIPPER_CONTRACT.json').read_text())
 for r in rows:
  q=np.array(r['width']);obs.append((q[1:]>np.array(c['threshold_m'])).astype(int))
  te=[events(np.array(r['labels'])[:,k],r['label_init'][k]) for k in range(2)];pe=[events(obs[-1][:,k],r['current'][k]) for k in range(2)]
  for k in range(2):
   lags += match(pe[k],te[k],50)
   for t,d in te[k]:alltrue.append((r['parent'],r['cid'],r['anchor']+t,k,d))
 crossing=score(rows,obs,(3,5,10,20));unique=list(set(alltrue));rng=np.random.default_rng(90);sensitivity=[]
 # For an isolated event with bounded uniform integer annotation jitter, exact within-tolerance probability.
 # MC uses shared jitter for each unique parent/cid/absolute-time/side/direction, so overlapping windows are coherent.
 for jitter in (1,3,5,10):
  vals=[]
  for repeat in range(500):
   noise=rng.integers(-jitter,jitter+1,len(unique));vals.append([float(np.mean(abs(noise)<=tol)) for tol in (3,5,10,20)])
  sensitivity.append(dict(uniform_jitter_frames=jitter,isolated_event_f1_mean=np.mean(vals,0).tolist(),tol=[3,5,10,20],unique_events=len(unique),repeats=500))
 put('LABEL_SENSITIVITY.json',dict(crossing_vs_onset=crossing,crossing_minus_onset_quantiles=np.quantile(lags,[0,.25,.5,.75,1]).tolist(),isolated_jitter=sensitivity))
 lines=['## G0 标签精度与±5可达性检查','真值仍是E102-A由实测反馈回溯得到的运动起点代理，未移动原标签。以下是敏感性诊断，既非替换测试真值，也不是已识别的真实命令F1上限。先用实测开度自身阈值切换与原起点代理匹配，量化定义差；再对唯一真实事件起点加独立有界整数均匀抖动±1/3/5/10帧（seed90，500次），假定孤立事件且忽略窗边界，F1即抖动落在容差内的比例。重叠窗共享事件身份；该简化无错配/边界损失，只是乐观敏感性上界，不能估计人的真实标注噪声。','|参照|F1±3|F1±5|F1±10|F1±20|','|---|---|---|---|---|']
 lines.append('|实测阈值crossing对原onset|'+'|'.join(f"{crossing[str(t)]['f1']:.4f}" for t in (3,5,10,20))+'|')
 for s in sensitivity:lines.append('|孤立事件抖动±'+str(s['uniform_jitter_frames'])+'|'+'|'.join(f'{v:.4f}' for v in s['isolated_event_f1_mean'])+'|')
 lines.append('crossing−onset匹配偏移min/Q25/median/Q75/max='+str(np.quantile(lags,[0,.25,.5,.75,1]).tolist())+'帧。真实实发命令与人工复标均缺失，无法识别真实±5上限；小于等于±5帧的有界误差在孤立事件假设下不会单独造成±5漏匹配。')
 report('\n'.join(lines))

def main():
 start=time.monotonic();assert (O/'BASELINE_REPRODUCED.json').exists() and (O/'COLLECT_COMPLETE.json').exists()
 rows=json.loads((O/'predictions.json').read_text());train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];assert len(dev)==211
 configs=[]
 for lo,hi in ((.3,.7),(.4,.6),(.45,.55)):configs.append(dict(family='hysteresis',low=lo,high=hi))
 for d in (2,3,5,8):configs.append(dict(family='dwell',d=d))
 for lo,hi in ((.3,.7),(.4,.6),(.45,.55)):
  for d in (2,3,5,8):configs.append(dict(family='combined',low=lo,high=hi,d=d))
 for alpha in (.2,.4,.6,.8):configs.append(dict(family='ema',alpha=alpha))
 test_causality(configs);put('CAUSALITY_TEST.json',dict(passed=True,checks='perturb all future probabilities and truncate prefixes; identical prefixes for all grid configs',forbids='no centered average, no backdating, no future confirmation',grid=configs))
 grid=[];selected={}
 for j,cfg in enumerate(configs):
  ss=[decode(np.array(r['prob']),r['current'],cfg) for r in train];v=score(train,ss,(10,))['10'];record=dict(config=cfg,train_f1=v['f1'],train_false_per_min=v['false_per_min'],train_recall=v['recall'],train_true=v['true'],train_pred=v['pred']);grid.append(record)
  key=(v['f1'],-v['false_per_min']);fam=cfg['family']
  if fam not in selected or key>selected[fam][0]:selected[fam]=(key,cfg)
  print('grid',j+1,len(configs),record,flush=True);put('status.json',dict(status='train_grid',done=j+1,total=len(configs),elapsed_s=time.monotonic()-start))
 chosen={fam:val[1] for fam,val in selected.items()};put('TRAIN_SELECTED_LOCK.json',dict(selected=chosen,grid=grid,selection='train-only parent F1@10 max; false switches min; fixed grid order',locked_before_dev_decode=True))
 report('## G1 训练parent选参完成（dev解码前锁定）\n'+json.dumps(chosen,ensure_ascii=False)+'\n|家族及参数|train F1@10|train 假切换/分钟|train 召回|真事件/预测事件|\n|---|---|---|---|---|\n'+'\n'.join('|'+str(v['config'])+f"|{v['train_f1']:.6f}|{v['train_false_per_min']:.3f}|{v['train_recall']:.6f}|{v['train_true']}/{v['train_pred']}|" for v in grid))
 basestates=[(np.array(r['logits'])>0).astype(int) for r in dev];base,offsets,fps=g0(dev,basestates,'all27');original=[(r,s) for r,s in zip(dev,basestates) if r['original48']];g0([r for r,s in original],[s for r,s in original],'original48');labelcheck(dev)
 result={}
 for subset,rr in [('all27',dev),('original48',[r for r in dev if r['original48']])]:
  bb=score(rr,[(np.array(r['logits'])>0).astype(int) for r in rr]);result[subset]={'baseline':bb};lines=['## G1 '+subset+' 冻结dev结果','|规则|tol|F1|假检/分钟|召回|偏移median帧|ΔF1 [95%CI]|Δ假检 [95%CI]|Δ召回 [95%CI]|','|---|---|---|---|---|---|---|---|---|']
  for tol,v in bb.items():lines.append(f"|原样|{tol}|{v['f1']:.6f}|{v['false_per_min']:.3f}|{v['recall']:.6f}|{v['offset_median']}|0|0|0|")
  for fam,cfg in chosen.items():
   vv=score(rr,[decode(np.array(r['prob']),r['current'],cfg) for r in rr]);ci=bootstrap(vv,bb);v=vv['10'];b=bb['10'];reduction=1-v['false_per_min']/b['false_per_min'];rec_drop=1-v['recall']/b['recall'];effective=ci['10']['f1']['ci95'][0]>0 and reduction>=.3 and rec_drop<=.05
   result[subset][fam]=dict(config=cfg,metrics=vv,paired=ci,false_reduction=reduction,recall_relative_drop=rec_drop,effective=bool(effective))
   for tol,v in vv.items():
    d=ci[tol];lines.append(f"|{fam}|{tol}|{v['f1']:.6f}|{v['false_per_min']:.3f}|{v['recall']:.6f}|{v['offset_median']}|{d['f1']['delta']:+.6f} {d['f1']['ci95']}|{d['false_per_min']['delta']:+.3f} {d['false_per_min']['ci95']}|{d['recall']['delta']:+.6f} {d['recall']['ci95']}|")
  lines+=['|规则|F1@10 CI下界>0|假切换减少|召回相对降幅|满足全部预注册|','|---|---|---|---|---|']
  for fam in chosen:
   v=result[subset][fam];lines.append(f"|{fam}|{v['paired']['10']['f1']['ci95'][0]>0}|{100*v['false_reduction']:.2f}%|{100*v['recall_relative_drop']:.2f}%|{v['effective']}|")
  report('\n'.join(lines))
 put('G1_RESULTS.json',result)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,3.6));ax[0].hist(offsets['20'],bins=np.arange(-20.5,21.5),edgecolor='white');ax[0].set(xlabel='Predicted minus true onset (frames), matches at ±20',ylabel='Event-window count');ax[1].plot([3,5,10,20],[base[str(t)]['f1'] for t in (3,5,10,20)],'o-',label='baseline')
 for fam in chosen:ax[1].plot([5,10,20],[result['all27'][fam]['metrics'][str(t)]['f1'] for t in (5,10,20)],'o-',label=fam)
 ax[1].set(xlabel='Tolerance (frames)',ylabel='Parent mean F1');ax[1].legend();fig.tight_layout();fig.savefig(O/'offset_tolerance.png',dpi=160);plt.close(fig)
 # Predefined first twelve false detections in window/side/time order, never chosen by apparent severity.
 fig,axes=plt.subplots(3,4,figsize=(13,7))
 for ax,f in zip(axes.flat,fps[:12]):
  t=np.arange(f['trace_start'],f['trace_start']+len(f['trace']));ax.plot(t,f['trace'],'o-',ms=2);ax.axhline(.5,c='grey',ls='--');ax.axvline(f['time'],c='red');ax.set_ylim(-.02,1.02);ax.set_title(f"index {f['index']}, {'L' if f['gripper']==0 else 'R'}, t={f['time']}",fontsize=9)
 fig.tight_layout();fig.savefig(O/'false_event_traces.png',dpi=150);plt.close(fig)
 effective=any(result['all27'][fam]['effective'] for fam in chosen)
 conclusion='至少一个训练侧选定因果解码规则满足预注册有效条件；不自动部署。' if effective else '瓶颈不在因果解码，需要在原架构内通过目标/数据层面解决'
 report('## 预注册判定与局限\n'+conclusion+'\n这些规则是模型外的推理时解码层，没有训练、没有新头、没有改变模型；顺序处理每窗概率，仅访问≤t，当前实测状态初始化；确认当帧输出、不回填，因果前缀扰动测试全网格通过。这里t为预测窗内偏移，不声称已评估闭环逐时重规划；模型于anchor产生原生50步预测，解码不读取未来观测/标签。\n候选仅建议、不实施：①在现有夹爪BCE中按训练parent统计对运动起点邻域/开合少数帧重加权，同时保留静止负例以约束误触发；②按真实事件/稳定保持片段平衡训练窗口采样，尤其覆盖跨anchor正在开合样本；③在训练parent内复查反馈起点代理和实际命令时延、修订监督定义或软时间标签，重新冻结独立测试协议。均不新增头或架构。\n单seed，dev27parent且重叠窗口重复事件，CI仅反映parent采样噪声；4家族CI为逐规则区间、无多重检验校正，不以dev最优部署选型。实测静止是单爪局部差分；人工阶段与实发命令缺失，不能归因到任务阶段或识别实际标签精度上限。G0局部未来轨迹仅诊断，未进入G1。小样本分层不作因果结论。全量逐parent指标、全部偏移、每假事件轨迹/寿命及偏移中位数配对CI见JSON；尾部寿命截尾，不能由无回切断言保持正确。\n图：[偏移直方图和容差曲线](../results/e108g_diag_20261116_01/offset_tolerance.png)；[预定义前12个假事件轨迹](../results/e108g_diag_20261116_01/false_event_traces.png)。\n未完成：人工任务阶段分层和真实命令起点上限不可识别，已明示；未训练/未部署属于本轮边界。用户决定：无本轮阻塞项。')
 put('ANALYSIS_COMPLETE.json',dict(status='complete',elapsed_s=time.monotonic()-start,effective=effective,conclusion=conclusion,GPUh=0))
 print(conclusion,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:put('ANALYSIS_ERROR.json',dict(traceback=traceback.format_exc()));raise
