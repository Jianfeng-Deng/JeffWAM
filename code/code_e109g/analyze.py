import os,sys,json,time,collections,traceback,datetime
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e109g_generalization_20261116_01/mplconfig')
sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';R=T/'notes/e109g_gripper_generalization_20261116.md'
import numpy as np
from scipy.stats import spearmanr

def put(n,x):
 p=O/n;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)));os.replace(tmp,p)
def report(s):
 with R.open('a') as f:f.write('\n'+s+'\n')
def ci(x):return np.quantile(x,[.025,.5,.975]).tolist()
def fmt(x):return '['+', '.join(f'{v:.4f}' for v in x)+']'
def main():
 assert (O/'COLLECT_COMPLETE.json').exists() and (O/'METADATA_COMPLETE.json').exists();assert not (O/'ANALYSIS_COMPLETE.json').exists();metrics=json.loads((O/'METRICS.json').read_text());cov=json.loads((O/'COVERAGE.json').read_text());ex=json.loads((O/'EXPOSURE.json').read_text());meta=json.loads((O/'PARENT_METADATA_RESOLVED.json').read_text())['records'];full=json.loads((O/'FULL_CAPTURE_STATS.json').read_text());steps=[2000,5000,10000,15000,20000];suites=['train16','train_aligned','dev'];tols=['5','10','20'];final=metrics['20000'];trainpars=sorted(final['train_aligned']['10']['parent_raw']);devpars=sorted(final['dev']['10']['parent_raw']);assert len(trainpars)==145 and len(devpars)==27
 for suite in suites:
  assert final[suite]['10']['true']==sum(v['window_events'] for v in cov[suite].values()),'coverage truth mismatch'
  for st in metrics:assert list(metrics[st][suite]['10']['parent_raw'])==sorted(cov[suite])
 rng=np.random.default_rng(90);it=rng.integers(145,size=(2000,145));idd=rng.integers(27,size=(2000,27));sub=np.array([rng.choice(145,27,replace=False) for _ in range(2000)]);np.savez_compressed(O/'bootstrap_indices_seed90.npz',train=it,dev=idd,train27=sub)
 a=lambda st,s,t:np.array(metrics[str(st)][s][str(t)]['parent_metrics'])
 protocol={};noise={};samples={};ceil={}
 lines=['## 1 已完成：同口径20k评分','主指标完全沿用E108-G：50帧/30Hz、当前实测阈值状态初始化预测事件、onset标签状态初始化真事件、同爪同方向DP一一匹配、每parent先累积TP/预测/真值再计算，parent等权。无事件parent的F1和recall均记0；未修改标签或动作。','|集合|容差|F1|召回|假检/分钟|TP/真/预测|','|---|---:|---:|---:|---:|---|']
 for s in suites:
  for t in tols:
   v=final[s][t];lines.append(f"|{s}|±{t}|{v['f1']:.6f}|{v['recall']:.6f}|{v['false_per_min']:.3f}|{v['matched']}/{v['true']}/{v['pred']}|")
 lines+=['|容差|train16−dev原差距|train16−aligned（同145parent配对）[95%CI]|aligned−dev剩余差距[95%CI]|口径差/原差距（带符号）|','|---|---:|---|---|---:|']
 for t in tols:
  old=a(20000,'train16',t)[:,0];al=a(20000,'train_aligned',t)[:,0];dv=a(20000,'dev',t)[:,0];d=old-al;gap=al.mean()-dv.mean();bs=al[sub].mean(1)-dv[idd].mean(1);p=dict(original_gap=float(old.mean()-dv.mean()),protocol_delta=float(d.mean()),protocol_ci95=ci(d[it].mean(1))[::2],aligned_gap=float(gap),aligned_gap_sampling_ci95=ci(bs)[::2],protocol_fraction=float(d.mean()/(old.mean()-dv.mean())) if old.mean()!=dv.mean() else None);protocol[t]=p;lines.append(f"|±{t}|{p['original_gap']:.6f}|{p['protocol_delta']:+.6f} {fmt(p['protocol_ci95'])}|{gap:+.6f} {fmt(p['aligned_gap_sampling_ci95'])}|{100*p['protocol_fraction']:+.2f}%|")
 lines+=['口径配对bootstrap严格在同145个训练parent内共用2000次索引；训练/dev是不同parent，不能伪作配对。剩余差距区间使用随机27个不同训练parent与dev parent bootstrap组合，反映相同parent规模的采样不确定性，不是因果效应。口径差是窗口/覆盖/代表canonical选择的联合描述，不能全部命名为过拟合。','|集合|每parent窗数min/median/max|每parent事件min/median/max|同爪覆盖事件间隔帧Q25/median/Q75|anchor相对录制位置Q25/median/Q75|初始实测/标签不一致窗|','|---|---|---|---|---|---:|']
 for s in suites:
  vv=list(cov[s].values());q=lambda z,ps:np.quantile(z,ps).round(4).tolist() if z else None;intervals=sum([v['event_intervals_frames'] for v in vv],[]);anchors=sum([v['anchor_fraction'] for v in vv],[]);lines.append(f"|{s}|{q([v['windows'] for v in vv],[0,.5,1])}|{q([v['window_events'] for v in vv],[0,.5,1])}|{q(intervals,[.25,.5,.75])}|{q(anchors,[.25,.5,.75])}|{sum(v['initial_mismatch_windows'] or 0 for v in vv)}|")
 lines+=['覆盖事件间隔仅指窗口命中的同爪事件之间的录制时间间距；可能跨未覆盖片段，不代表完整事件间隔。完整录制间隔已在物理分布表单列。aligned初始化采用与dev完全相同的实测状态与标签初始状态；每parent细表含唯一覆盖帧、事件、重叠率，见附录。']
 report('\n'.join(lines));put('PROTOCOL_COMPARISON.json',protocol)
 lines=['## 2 已完成：parent抽样噪声（2000次，seed90）','dev27按parent有放回bootstrap；训练侧每次随机无放回抽27parent，重复2000次。相同抽样索引用于三种容差及全部checkpoint。保留事件稀疏与窗口重叠结构，不把窗口或事件当独立样本。','|集合|容差|全体F1|分布2.5%/50%/97.5%|训练27次均值≤固定dev均值比例|','|---|---:|---:|---|---|']
 for s in suites:
  idx=idd if s=='dev' else sub;noise[s]={}
  for t in tols:
   arr=a(20000,s,t)[:,0];bs=arr[idx].mean(1);samples[s+'_'+t]=bs;record=dict(point=float(arr.mean()),quantiles=ci(bs),prob_train27_le_fixed_dev=float(np.mean(bs<=final['dev'][t]['f1'])) if s!='dev' else None);noise[s][t]=record;lines.append(f"|{s}|±{t}|{arr.mean():.6f}|{fmt(ci(bs))}|{record['prob_train27_le_fixed_dev']}|")
 lines+=['这些比例是训练子集尾部频率，并非已校准的假设检验p值或因果概率。','|集合|27parent窗口事件总数2.5%/50%/97.5%|唯一覆盖事件总数2.5%/50%/97.5%|有事件parent数2.5%/50%/97.5%|','|---|---|---|---|']
 counts={}
 for s in suites:
  ps=sorted(cov[s]);idx=idd if s=='dev' else sub;mat=np.array([[cov[s][p]['window_events'],cov[s][p]['unique_covered_events'],int(cov[s][p]['window_events']>0)] for p in ps]);b=mat[idx].sum(1);counts[s]={k:ci(b[:,j]) for j,k in enumerate(['window_events','unique_events','event_parents'])};lines.append(f"|{s}|{fmt(ci(b[:,0]))}|{fmt(ci(b[:,1]))}|{fmt(ci(b[:,2]))}|")
 unique=sum(v['unique_covered_events'] for v in cov['dev'].values());tp=final['dev']['10']['matched'];nt=final['dev']['10']['true'];rec=tp/nt;z=1.96;den=1+z*z/nt;center=(rec+z*z/(2*nt))/den;half=z*np.sqrt(rec*(1-rec)/nt+z*z/(4*nt*nt))/den;power=dict(event_windows=nt,unique_events=unique,micro_recall=rec,wilson95_optimistic=[center-half,center+half],mde80_two_equal_n49_p05=2.8*np.sqrt(.5/nt),n_independent_for_10pp_recall_halfwidth_worst_case=int(np.ceil(z*z*.25/.1**2)),n_per_group_for_10pp_80power_worst_case=int(np.ceil(2*2.8**2*.25/.1**2)))
 lines+=['事件功效说明：dev为'+str(nt)+'条事件窗口实例、'+str(unique)+'个唯一事件；微平均召回TP/真='+f'{tp}/{nt}={rec:.4f}'+'，不同于主表parent宏平均召回。即使乐观地把事件当独立，召回Wilson95%CI='+fmt(power['wilson95_optimistic'])+'。最坏p=.5、两组各49独立事件时，80%功效/双侧5%检验可辨差异约'+f"{power['mde80_two_equal_n49_p05']:.3f}"+'；若要辨认10pp差异，近似需每组392独立事件；单组召回±10pp精度约需97独立事件。真实parent聚类、重复事件和异质性会更差；这些只为事件召回的正态近似设计说明，不是F1精确功效或新阈值。']
 for s in suites:
  ps=sorted(final[s]['10']['parent_raw']);arr=a(20000,s,10)[:,0];evs=np.array([final[s]['10']['parent_raw'][p]['true']>0 for p in ps]);ceil[s]=dict(event_parent_fraction=float(evs.mean()),conditional_f1=float(arr[evs].mean()) if evs.any() else 0,macro_f1=float(arr.mean()))
 al=ceil['train_aligned'];dv=ceil['dev'];prev=.5*(al['conditional_f1']+dv['conditional_f1'])*(al['event_parent_fraction']-dv['event_parent_fraction']);cond=.5*(al['event_parent_fraction']+dv['event_parent_fraction'])*(al['conditional_f1']-dv['conditional_f1']);assert abs(prev+cond-protocol['10']['aligned_gap'])<1e-12
 lines+=['## 指标上限和事件覆盖的代数分解','零事件parent即使预测完美也记F1=0。下表是解释性补充，主指标不变。','|集合|有事件parent比例/理想上限|有事件parent上的条件F1@10|原宏F1@10|','|---|---:|---:|---:|']
 for s,v in ceil.items():lines.append(f"|{s}|{v['event_parent_fraction']:.6f}|{v['conditional_f1']:.6f}|{v['macro_f1']:.6f}|")
 lines+=['恒等式macroF1=p(event parent)×E(F1|event)。对aligned−dev做对称Kitagawa分解：事件parent比例项='+f'{prev:+.6f}'+'，条件F1项='+f'{cond:+.6f}'+'，恰好相加为剩余差距。它不是(a)(b)(c)(d)的互斥因果分摊，不能与口径差比例再相加。']
 report('\n'.join(lines));put('NOISE.json',dict(f1=noise,counts=counts,power=power,ceilings=ceil,prevalence_decomposition=dict(prevalence=prev,conditional=cond)));np.savez_compressed(O/'bootstrap_f1_samples.npz',**samples)
 lines=['## 3 已完成：固定checkpoint曲线与暴露分层','全部指定固定点只读评分，不以dev选择checkpoint。计步沿用E106-C：2000是原A2，后续5000/10000/15000/20000是自A2开始的Piper累计步，实际动作头总优化步数为2000/7000/12000/17000/22000。','|点|train16 F1@5/10/20|aligned F1@5/10/20|dev F1@5/10/20|','|---:|---|---|---|']
 for st in steps:lines.append('|'+str(st)+'|'+'|'.join('/'.join(f"{metrics[str(st)][s][t]['f1']:.6f}" for t in tols) for s in suites)+'|')
 curves={};lines+=['|同parent配对差@10|train16 Δ[95%CI]|aligned Δ[95%CI]|dev Δ[95%CI]|','|---|---|---|---|']
 for before,after in list(zip(steps[:-1],steps[1:]))+[(2000,20000)]:
  key=f'{after}-{before}';curves[key]={};cells=[]
  for s in suites:
   d=a(after,s,10)[:,0]-a(before,s,10)[:,0];idx=idd if s=='dev' else it;v=dict(delta=float(d.mean()),ci95=ci(d[idx].mean(1))[::2]);curves[key][s]=v;cells.append(f"{v['delta']:+.6f} {fmt(v['ci95'])}")
  lines.append('|'+key+'|'+'|'.join(cells)+'|')
 exposures={};lines+=['parent暴露分层在看F1前由采样计数排序等数量四组，计数并列按parent ID；直接暴露包括A2 2k与Piper20k，未把P1加入。','|暴露口径|四分位组|parent|暴露min/max|train16 F1@10|aligned F1@10|有事件parent(aligned)|','|---|---:|---:|---|---:|---:|---:|']
 for field in ('direct_grip_draws','draws_per_P3_window'):
  x=np.array([ex['parents'][p][field] for p in trainpars]);order=np.argsort(x,kind='stable');groups=np.array_split(order,4);records=[]
  for j,ids in enumerate(groups,1):
   y0=a(20000,'train16',10)[:,0];y=a(20000,'train_aligned',10)[:,0];rec=dict(quartile=j,n=len(ids),min=float(x[ids].min()),max=float(x[ids].max()),train16_f1=float(y0[ids].mean()),aligned_f1=float(y[ids].mean()),event_parents=sum(final['train_aligned']['10']['parent_raw'][trainpars[i]]['true']>0 for i in ids));records.append(rec);lines.append(f"|{field}|{j}|{len(ids)}|{rec['min']:.2f}/{rec['max']:.2f}|{rec['train16_f1']:.6f}|{rec['aligned_f1']:.6f}|{rec['event_parents']}|")
  correlations={}
  for s in ('train16','train_aligned'):
   y=a(20000,s,10)[:,0];rr=spearmanr(x,y);bs=[float(spearmanr(x[ix],y[ix]).statistic) for ix in it];correlations[s]=dict(spearman=float(rr.statistic),ci95=ci(bs)[::2])
  exposures[field]=dict(quartiles=records,correlations=correlations);lines.append(field+' Spearman(parent bootstrap95%CI)：'+str(correlations))
 lines+=['parent总暴露本来接近均匀，因此它对“多见是否更好”的辨识力有限。每候选窗暴露受parent窗口数/录制长度/来源/事件覆盖混杂；相关性不能证明记忆，也不能排除事件级记忆。没有跨seed训练、未见事件反事实或独立人工事件标签，(a)与(d)仍不可完全拆分。']
 report('\n'.join(lines));put('CURVE_PAIRED.json',curves);put('EXPOSURE_ASSOCIATION.json',exposures)
 # Metadata standardization: fixed natural strata, no model labels, all support sizes disclosed.
 md={r['parent']:r for r in meta};strata={s:collections.defaultdict(list) for s in ('train_aligned','dev')}
 for s,ps in [('train_aligned',trainpars),('dev',devpars)]:
  for i,p in enumerate(ps):r=md[p];strata[s][(r['recording_date']['UTC'],r['source_collection'])].append(i)
 shared=sorted(set(strata['train_aligned'])&set(strata['dev']));unsupported=sorted(set(strata['dev'])-set(strata['train_aligned']));supported=sum(len(strata['dev'][k]) for k in shared);std={};lines=['## 4 已完成：同录制日×来源的描述性标准化','用UTC录制日×原始source_collection作固定自然分组，不使用模型语言、任务阶段、模型预测标签或场景猜测。每组训练/dev父级人数明确列出；不是配对同一parent。','|UTC日期 / 来源|训练parent|dev parent|训练F1@10|dev F1@10|','|---|---:|---:|---:|---:|']
 atr=a(20000,'train_aligned',10)[:,0];adv=a(20000,'dev',10)[:,0]
 for k in sorted(set(strata['train_aligned'])|set(strata['dev'])):
  ti=strata['train_aligned'].get(k,[]);di=strata['dev'].get(k,[]);tm=float(atr[ti].mean()) if ti else None;dm=float(adv[di].mean()) if di else None;lines.append('|'+k[0]+' / '+k[1]+f'|{len(ti)}|{len(di)}|{tm}|{dm}|')
 if supported:
  wt={k:len(strata['dev'][k])/supported for k in shared};tm=sum(wt[k]*atr[strata['train_aligned'][k]].mean() for k in shared);dm=sum(wt[k]*adv[strata['dev'][k]].mean() for k in shared);res=[];rng2=np.random.default_rng(90)
  for repeat in range(2000):
   diff=0
   for k in shared:
    x=atr[strata['train_aligned'][k]];y=adv[strata['dev'][k]];diff+=wt[k]*(rng2.choice(x,len(x),replace=True).mean()-rng2.choice(y,len(y),replace=True).mean())
   res.append(diff)
  std=dict(supported_dev_parents=supported,unsupported_dev_parents=27-supported,unsupported_strata=[list(k) for k in unsupported],weighted_train=float(tm),supported_dev=float(dm),gap=float(tm-dm),ci95=ci(res)[::2],weights={' / '.join(k):v for k,v in wt.items()});lines+=['按dev共同支持组组成重加权后：训练F1@10='+f'{tm:.6f}'+'，对应dev='+f'{dm:.6f}'+'，差='+f'{tm-dm:+.6f}'+'，分层独立parent bootstrap95%CI='+fmt(std['ci95'])+'。仅覆盖dev '+str(supported)+'/27parent；其余'+str(27-supported)+'个无训练共同组，不能外推。小格（例如1个dev parent）bootstrap无法反映该格总体不确定性，区间可能偏窄。','训练全体均值→重加权均值差仅是组成敏感性；同时改变为共同支持dev集合，不能将差值占比当作场景/日期的因果解释率。']
 report('\n'.join(lines));put('DISTRIBUTION_STANDARDIZATION.json',std)
 # Parent-level audit table is report-local; no output flooding.
 lines=['## 附录：逐parent事件、间隔与窗口覆盖','完整对应表另见COVERAGE.json / FULL_CAPTURE_STATS.json / PARENT_METADATA_RESOLVED.json。E108 train16与aligned按同parent ID配对；dev独立。间隔中位数来自完整同爪同连续段事件，单位秒；“覆盖事件”列为窗口重复计数/唯一命中/完整录制事件；frame coverage是预测目标帧的union比例。','|集合|parent|窗|覆盖事件(重复/唯一/全)|全段事件间隔median(s)|frame coverage|anchor[min,max]|F1@10|','|---|---|---:|---|---|---:|---|---:|']
 for s in suites:
  for par,v in sorted(cov[s].items()):
   intervals=[]
   if s!='train16':
    cid=md[par]['cid'];intervals=full[cid].get('interval_s',[])
   else:
    # Aggregate same-parent captures represented by old16, exact intervals kept in full stats where present.
    intervals=sum([f.get('interval_s',[]) for f in full.values() if f['parent']==par],[])
   f1=final[s]['10']['parent_raw'][par];f1v=2*f1['tp']/(f1['pred']+f1['true']) if f1['pred']+f1['true'] else 0;lines.append(f"|{s}|{par}|{v['windows']}|{v['window_events']}/{v['unique_covered_events']}/{v['full_capture_events']}|{float(np.median(intervals)) if intervals else None}|{v['target_frame_coverage']:.4f}|{min(v['anchor_fraction']):.3f},{max(v['anchor_fraction']):.3f}|{f1v:.4f}|")
 report('\n'.join(lines))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,3.6))
 for s in suites:
  yy=[metrics[str(st)][s]['10']['f1'] for st in steps];ax[0].plot(steps,yy,'o-',label=s)
 ax[0].set(xlabel='Fixed checkpoint label (E106-C convention)',ylabel='Parent macro F1 +/-10');ax[0].legend()
 for s in suites:ax[1].hist(samples[s+'_10'],bins=35,alpha=.5,density=True,label=s)
 ax[1].set(xlabel='F1 +/-10: train subsets27 / dev bootstrap27',ylabel='Density');ax[1].legend();fig.tight_layout();fig.savefig(O/'generalization.png',dpi=160);plt.close(fig)
 put('ANALYSIS_COMPLETE.json',dict(status='complete',GPUh=0,bootstrap=2000,seed=90,original_gap=protocol['10']['original_gap'],protocol_delta=protocol['10']['protocol_delta'],aligned_gap=protocol['10']['aligned_gap'],standardized=std))
 print(json.dumps(dict(protocol=protocol,noise10={s:noise[s]['10'] for s in suites},curves=curves['20000-2000'],exposure=exposures,standardization=std),ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:put('ANALYSIS_FAILED.json',dict(traceback=traceback.format_exc()));raise
