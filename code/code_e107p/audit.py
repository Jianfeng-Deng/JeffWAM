import analyze as a
from pathlib import Path
import numpy as np,json,hashlib,fcntl,datetime
R=a.R;T=a.T
meta=a.read('vector_meta.json');entries=a.read('entries.json');em={x['index']:x for x in entries};vectors=np.load(R/'vectors.npz');nr={x['index']:x for x in a.read('neighbors.json')['rows']};rawscale=json.loads((T/'code_e88/artifacts/cpu_audit_01/residual_stats.json').read_text())['scale'];old=json.loads((T/'results/e106c_control_20261115_02/eval_S20000/evaluation.json').read_text())['world'];oi={(x['index'],x['h'],x['method']):x for x in old};dims={};maxdiff=0.;extra=[];amb=[];hist={};condpaired=[]
with (R/'grip_probe.pkl').open('rb') as f:a.pickle.load(f)
with (R/'grip_probe.pkl').open('rb') as f:probe=a.pickle.load(f)
def gp(A):return np.stack([m.predict_proba(sc.transform(A))[:,1] for sc,m in probe],1)
for subset in ('all27','original48'):
 for h in (10,25,50,100):
  mm=[v for v in meta if v['h']==h and (subset=='all27' or v['original48'])];ids=[v['index'] for v in mm];par=np.array([v['parent'] for v in mm]);w0,w,ps=a.weights(par);Y=np.stack([vectors[f'{i}_{h}_Y'] for i in ids]).astype('f8');C=np.stack([vectors[f'{i}_{h}_C'] for i in ids]).astype('f8');P={cond:np.stack([vectors[f'{i}_{h}_{cond}'] for i in ids]).astype('f8') for cond in ('no_traj','gt_action')};sc=np.array(rawscale[str(h)]).reshape(1,-1,1);copy=np.mean(((Y-C)/sc)**2,axis=(1,2));pool=lambda z:z.reshape(len(z),3,-1,768).mean(2).reshape(len(z),2304)
  realgp=gp(pool(Y));hi=(10,25,50).index(h) if h!=100 else None
  amask=np.array([[(hi is not None and nr.get(i,{}).get('candidate',False) and nr.get(i,{}).get('eligible',False) and .2<=nr[i]['proportions'][hi][g]<=.8) for g in range(2)] for i in ids]);mixed=amask.any(1)
  err={}
  for cond,p in P.items():
   err[cond]=np.mean(((p-Y)/sc)**2,axis=(1,2));maxdiff=max(maxdiff,max(abs(err[cond][j]-oi[i,h,cond]['model']) for j,i in enumerate(ids)))
   for tag,aa,bb in [('absolute',p,Y),('displacement',p-C,Y-C)]:
    aa=aa.reshape(len(aa),-1);bb=bb.reshape(len(bb),-1);vp=w0@(aa*aa)-(w0@aa)**2;vy=w0@(bb*bb)-(w0@bb)**2;dims[f'{subset}_{h}_{cond}_{tag}_pred_variance']=vp;dims[f'{subset}_{h}_{cond}_{tag}_real_variance']=vy;dims[f'{subset}_{h}_{cond}_{tag}_ratio']=vp/np.maximum(vy,1e-12)
   mass=(w0@(err[cond]*mixed))/max(w0@err[cond],1e-12);bmass=(w@(err[cond]*mixed))/np.maximum(w@err[cond],1e-12)
   extra.append(dict(subset=subset,h=h,condition=cond,skill=dict(value=float(1-(w0@err[cond])/(w0@copy)),ci95=a.interval(1-(w@err[cond])/(w@copy))),mixed_error_mass_fraction=dict(value=float(mass),ci95=a.interval(bmass)),mixed_windows=int(mixed.sum()),mixed_parents=len(np.unique(par[mixed]))))
   prob=gp(pool(p))
   if mixed.any():
    wa,wb,pa=a.weights(par[mixed]);mid=((((prob>=.3)&(prob<=.7))*amask).sum(1)/np.maximum(amask.sum(1),1))[mixed];rmid=((((realgp>=.3)&(realgp<=.7))*amask).sum(1)/np.maximum(amask.sum(1),1))[mixed]
    amb.append(dict(subset=subset,h=h,condition=cond,windows=int(mixed.sum()),parents=len(pa),gripper_pairs=int(amask.sum()),pred_middle=a.estimate(mid,wa,wb),real_middle=a.estimate(rmid,wa,wb),middle_excess=a.estimate(mid-rmid,wa,wb)))
   if subset=='all27' and h!=100:hist[h,cond]=(prob,realgp,amask)
  difference=err['gt_action']-err['no_traj'];condpaired.append(dict(subset=subset,h=h,normalized_mse_gt_minus_no=a.estimate(difference,w0,w),skill_gt_minus_no=dict(value=float(-(w0@difference)/(w0@copy)),ci95=a.interval(-(w@difference)/(w@copy)))))
np.savez_compressed(R/'per_dimension_variances.npz',**dims);a.put('ambiguous_gripper_landings.json',amb);a.put('error_mass_and_replay.json',dict(max_normalized_mse_abs_difference_from_E106C=maxdiff,cells=extra,paired_conditions=condpaired));assert maxdiff<1e-6
fig,axs=a.plt.subplots(3,4,figsize=(15,8.5))
for row,h in enumerate((10,25,50)):
 for j,cond in enumerate(('no_traj','gt_action')):
  p,y,mask=hist[h,cond]
  for g in range(2):
   ax=axs[row,j*2+g];sel=mask[:,g];ax.hist(y[sel,g],bins=np.linspace(0,1,16),alpha=.55,label='real future');ax.hist(p[sel,g],bins=np.linspace(0,1,16),alpha=.55,label='mean prediction');ax.axvspan(.3,.7,color='k',alpha=.08);ax.set_title(f'+{h} {cond} grip{g} n={sel.sum()}');ax.set_xlim(0,1)
axs[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(R/'ambiguous_gripper_histograms.png',dpi=160);a.plt.close(fig)
lines=['## 夹爪逐爪复核与数值重放','![真正混合夹爪](../results/'+R.name+'/ambiguous_gripper_histograms.png)','主落点诊断改用同一起点中近邻未来确实混合的夹爪：这只是预注册条件的逐爪细化，不改变起点/阈值/数据；此前两爪平均保留作附表。混合格子+10为24窗/15parent、+25为30/19、+50为25/19；+10/+50低于预注册30窗门槛，不能给强结论；+100未定义混合事件格子，NA。附表仅6parent，CI较宽。','|h|条件|窗/parent/爪对数|均值中间率[CI]|真实中间率[CI]|差[CI]|','|---:|---|---|---|---|---|']
fmt=lambda z:f"{z['value']:.4f} [{z['ci95'][0]:.4f},{z['ci95'][1]:.4f}]"
for z in amb:
 if z['subset']=='all27':lines.append(f"|{z['h']}|{z['condition']}|{z['windows']}/{z['parents']}/{z['gripper_pairs']}|{fmt(z['pred_middle'])}|{fmt(z['real_middle'])}|{fmt(z['middle_excess'])}|")
lines+=['','逐窗口相对位移见per_window_displacement.json；逐token×768维的绝对/位移方差及比值见per_dimension_variances.npz（坐标为固定相机/空间token序，不跨相机混算；逐维仅描述、不做逐维显著性检验，推断以trace比的parent bootstrap为准）。所有窗口的normalized MSE与E106C原固定终点评估重放最大绝对差='+str(maxdiff)+'，<1e−6；未重新挑选窗口或改归一化。','## 可改善范围的乐观情景上界','下面仅是“若能完全消除机械混合事件子集的现有误差、其它窗口不变”的误差质量上界，不是多峰模型可达收益，不是不可约方差估计。均值是平方损失的条件Bayes估计，多峰采样本身不保证单样本MSE下降；不使用best-of-K。','|h|条件|混合子集误差质量/全体[CI]|当前归一化skill[CI]|','|---:|---|---|---|']
for z in extra:
 if z['subset']=='all27':lines.append(f"|{z['h']}|{z['condition']}|"+(fmt(z['mixed_error_mass_fraction']) if z['h']!=100 else 'NA')+f"|{fmt(z['skill'])}|")
a.report('\n'.join(lines));a.put('AUDIT_NUMERIC_COMPLETE.json',dict(max_mse_replay_difference=maxdiff,dimension_keys=len(dims),status='complete'))
print('audit numeric complete',maxdiff)
