import os,sys,json,time,pickle,traceback,importlib.util
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e107p_diagnosis_20261116_02';REPORT=T/'notes/e107p_multimodal_diagnosis_20261116.md'
os.environ['MPLCONFIGDIR']=str(R/'mplconfig')
import numpy as np
from scipy.spatial.distance import cdist,pdist
from scipy.special import expit
from scipy.stats import rankdata
import matplotlib.pyplot as plt
sp=importlib.util.spec_from_file_location('linear',T/'code_e99b3/linear.py');linear=importlib.util.module_from_spec(sp);sys.modules['linear']=linear;sp.loader.exec_module(linear)
LR=linear.LogisticRegression;Scaler=linear.StandardScaler
B=2000

def put(n,x):
 p=R/n;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,p)
def read(n):return json.loads((R/n).read_text())
def report(s):
 with REPORT.open('a') as f:f.write('\n'+s+'\n')
def auc(y,s,w=None):
 y=np.array(y);s=np.array(s);w=np.ones(len(y)) if w is None else np.asarray(w);good=w>0;y=y[good];s=s[good];w=w[good]
 if not len(y) or (w[y==0].sum()*w[y==1].sum())==0:return None
 ix=np.argsort(s,kind='stable');y=y[ix];s=s[ix];w=w[ix];starts=np.r_[0,np.flatnonzero(np.diff(s))+1];pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(1-y),starts);return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/(pos.sum()*neg.sum()))
def interval(v):
 a=np.array([x for x in v if x is not None and np.isfinite(x)]);return np.quantile(a,[.025,.975]).tolist() if len(a) else None

def weights(parents):
 ps=np.unique(parents);which=np.searchsorted(ps,parents);cnt=np.bincount(which);draw=np.random.default_rng(90).integers(0,len(ps),(B,len(ps)));mult=np.array([np.bincount(d,minlength=len(ps)) for d in draw]);w=mult[:,which]/cnt[which]/len(ps);w0=1/cnt[which]/len(ps);return w0,w,ps

def estimate(x,w0,w):return dict(value=float(w0@x),ci95=interval(w@x))
def summarize_curve(Y,C,P,parents):
 w0,w,ps=weights(parents);y=Y.reshape(len(Y),-1).astype('f8');p=P.reshape(len(P),-1).astype('f8');c=C.reshape(len(C),-1).astype('f8')
 norm=lambda a:np.linalg.norm(a,axis=1)
 co=lambda a,b:np.sum(a*b,1)/np.maximum(norm(a)*norm(b),1e-12)
 out={k:estimate(x,w0,w) for k,x in dict(norm_ratio=norm(p)/norm(y),cos=co(p,y),delta_norm_ratio=norm(p-c)/np.maximum(norm(y-c),1e-12),delta_cos=co(p-c,y-c),mse=np.mean((p-y)**2,1),copy_mse=np.mean((c-y)**2,1)).items()}
 # exact parent-equal weighted trace covariance; per-coordinate ratios also saved.
 def variance(a):
  mu=w0@a;v=w0@(a*a)-mu*mu
  # Process bootstraps in blocks to keep CPU RAM bounded.
  ps=np.unique(parents);pm=np.stack([a[parents==p].mean(0) for p in ps]);pn=np.array([np.sum(a[parents==p]**2,1).mean() for p in ps]);bw=np.stack([w[:,parents==p].sum(1) for p in ps],1);gram=pm@pm.T
  boot=bw@pn-np.einsum('bi,ij,bj->b',bw,gram,bw,optimize=True)
  return v,boot
 for name,aa,bb in [('variance_ratio',p,y),('delta_variance_ratio',p-c,y-c)]:
  vp,bp=variance(aa);vy,by=variance(bb);out[name]=dict(value=float(vp.sum()/vy.sum()),ci95=interval(bp/by),median_dimension_ratio=float(np.median(vp[vy>1e-10]/vy[vy>1e-10])),dimension_ratio_q10_q90=np.quantile(vp[vy>1e-10]/vy[vy>1e-10],[.1,.9]).tolist())
 out.update(windows=len(Y),parents=len(ps));return out

def main():
 assert read('COLLECT_COMPLETE.json')['status']=='complete'
 train=np.load(R/'train_probe.npz');X=train['X'].astype('f8');gy=train['y'];tp=train['parent'];parents=np.unique(tp);val=np.isin(tp,parents[:max(1,len(parents)//5)])
 scaler=Scaler().fit(X[~val]);probe=[];hold={}
 for g in range(2):
  m=LR(C=.01,max_iter=400).fit(scaler.transform(X[~val]),gy[~val,g]);p=m.predict_proba(scaler.transform(X[val]))[:,1];w0,w,ps=weights(tp[val]);hold[str(g)]=dict(windows=int(val.sum()),parents=len(ps),auroc=auc(gy[val,g],p,w0),auroc_ci95=interval([auc(gy[val,g],p,ww) for ww in w]),brier=estimate((p-gy[val,g])**2,w0,w))
  sc=Scaler().fit(X);m=LR(C=.01,max_iter=400).fit(sc.transform(X),gy[:,g]);probe.append((sc,m))
 put('grip_probe_holdout.json',hold)
 with (R/'grip_probe.pkl').open('xb') as f:pickle.dump(probe,f)
 print('probe holdout',hold,flush=True)
 # Train-only projection. PCA uses no evaluation windows, languages or labels.
 mu=X.mean(0);_,sv,V=np.linalg.svd(X-mu,full_matrices=False);basis=V[:32].T;Z=(X-mu)@basis;band=float(np.median(pdist(Z)));ridge=.001*np.var(Z,axis=0).mean()
 np.savez_compressed(R/'projection.npz',mean=mu,basis=basis,train_std=X.std(0),bandwidth=band,ridge=ridge)
 put('distribution_contract.json',dict(pca_dim=32,explained_variance=float((sv[:32]**2).sum()/(sv**2).sum()),bandwidth=band,k=3,ridge=float(ridge),noise_scale=.05,bootstrap='same parent draws across paired methods; kNN support and OOF classifier held fixed; other distribution moments/kernel reweighted each draw'))
 # Numeric neighbors: all references train parents, no dev labels enter neighborhood construction.
 refs=read('train_reference.json');rx=np.array([v['x'] for v in refs]);rp=np.array([v['parent'] for v in refs]);phase=np.array([v['phase'] for v in refs]);fy=np.array([v['future'] for v in refs]);std=rx.std(0);std[std<1e-6]=1;rx=(rx-rx.mean(0))/std;rawmu=np.mean([v['x'] for v in refs],0)
 def neighbor(x,ph,exclude=None):
  good=(phase==ph).all(1)
  if exclude is not None:good&=rp!=exclude
  ids=np.flatnonzero(good)
  if not len(ids):return np.array([],int),np.array([])
  d=np.sqrt(np.mean((rx[ids]-x)**2,1));order=np.argsort(d,kind='stable');seen=set();out=[];ds=[]
  for j in order:
   idx=ids[j]
   if rp[idx] in seen:continue
   seen.add(rp[idx]);out.append(idx);ds.append(d[j])
   if len(out)==20:break
  return np.array(out),np.array(ds)
 radii=[]
 for i in np.linspace(0,len(refs)-1,min(1024,len(refs)),dtype=int):
  ids,d=neighbor(rx[i],phase[i],rp[i])
  if len(ids)>=10:radii.append(d[min(19,len(d)-1)])
 radius=float(np.quantile(radii,.95));dn=read('dev_numeric.json');branch={};nr=[]
 for v in dn:
  if not v['available']:continue
  ids,d=neighbor((np.array(v['x'])-rawmu)/std,v['phase']);ids=ids[d<=radius];eligible=len(ids)>=10
  proportions=fy[ids].mean(0) if eligible else np.full((3,2),-1.)
  mixed=((proportions>=.2)&(proportions<=.8)).any(1)&eligible
  branch[v['index']]=dict(candidate=bool(v['branch']),eligible=eligible,mixed=mixed.tolist(),proportions=proportions.tolist(),neighbors=len(ids))
  nr.append(dict(index=v['index'],**branch[v['index']]))
 put('neighbors.json',dict(radius=radius,radius_queries=len(radii),rows=nr,reference_windows=len(refs),reference_parents=len(np.unique(rp))))
 entries=read('entries.json');em={e['index']:e for e in entries};meta=read('vector_meta.json');vectors=np.load(R/'vectors.npz');numeric={v['index']:v for v in dn};brsummary=[]
 for h in (10,25,50):
  hi=(10,25,50).index(h);ids=[e['index'] for e in entries];par=np.array([em[i]['parent'] for i in ids]);w0,w,ps=weights(par)
  can=np.array([branch.get(i,{}).get('candidate',False) for i in ids]);elig=np.array([branch.get(i,{}).get('eligible',False) for i in ids]);mixed=np.array([branch.get(i,{}).get('mixed',[False]*3)[hi] for i in ids]);true=can&elig&mixed
  summary=dict(h=h,all_windows=len(ids),all_parents=len(ps),candidate_windows=int(can.sum()),eligible_windows=int(elig.sum()),mixed_all_windows=int((elig&mixed).sum()),true_windows=int(true.sum()),true_parents=len(np.unique(par[true])),fraction_all=estimate(true.astype(float),w0,w),mixed_all_fraction=estimate((elig&mixed).astype(float),w0,w),fraction_candidate=dict(value=float((w0*true).sum()/max((w0*can).sum(),1e-12)),ci95=interval((w@true)/np.maximum(w@can,1e-12))))
  brsummary.append(summary)
 put('branch_summary.json',brsummary)
 report('## 分叉近邻与探针拟合完成\n机械阶段=两爪实测阈值状态+两爪过去6帧运动方向（阈值3×训练合同noise），不是模型语义阶段；42维q12+FK18+因果Δq12以训练池标准差归一。同阶段按欧氏RMS，逐训练parent仅最近一个，最多20，至少10个；半径由1024训练查询leave-parent-out第20邻距离95分位锁定，实际='+str(radius)+'。未来状态用原E102-A onset_labels，不移动标签，不使用未来状态建邻。分叉候选=当前后50帧内有原合同onset；混合=.2–.8。此为经验条件混合代理，无法证明同语义阶段的不可约多峰。\n夹爪探针CPU拟合固定C=.01；训练parent首20%留出诊断后全train重拟合，未选参。'+json.dumps(hold,ensure_ascii=False)+'\n'+json.dumps(brsummary,ensure_ascii=False)+'\n')
 # Frozen E99B3 head only; no dev annotations read.
 with (T/'results/e99b3_20261115_01/heads.pkl').open('rb') as f:head=pickle.load(f)['b']
 pids=[p['id'] for p in json.loads((T/'results/e99b3_20261115_01/spec.json').read_text())['predicates']]
 def predprob(A):
  x=head['scaler'].transform(A);return np.stack([m.predict_proba(x)[:,1] if hasattr(m,'predict_proba') else np.full(len(x),m) for m in head['head']],1)
 def gripprob(A):return np.stack([m.predict_proba(sc.transform(A))[:,1] for sc,m in probe],1)
 curves=[];distributions=[];landings=[];semantic=[];hist={};windowstats=[]
 rng=np.random.default_rng(90)
 for subset in ('all27','original48'):
  for h in (10,25,50,100):
   mm=[v for v in meta if v['h']==h and (subset=='all27' or v['original48'])];ids=[v['index'] for v in mm];par=np.array([v['parent'] for v in mm]);w0,w,ps=weights(par)
   Y=np.stack([vectors[f'{i}_{h}_Y'] for i in ids]);C=np.stack([vectors[f'{i}_{h}_C'] for i in ids]);pool=lambda a:a.reshape(len(a),3,-1,768).mean(2).reshape(len(a),2304)
   yy=pool(Y).astype('f8');zz=(yy-mu)@basis;noise=yy+rng.normal(size=yy.shape)*.05*X.std(0);nz=(noise-mu)@basis;reference=predprob(yy);gptrue=gripprob(yy)
   hi=(10,25,50).index(h) if h!=100 else None;mult=np.array([bool(hi is not None and branch.get(i,{}).get('candidate',False) and branch.get(i,{}).get('eligible',False) and branch.get(i,{}).get('mixed',[False]*3)[hi]) for i in ids])
   for cond in ('no_traj','gt_action'):
    P=np.stack([vectors[f'{i}_{h}_{cond}'] for i in ids]);pp=pool(P).astype('f8');pz=(pp-mu)@basis
    curve=summarize_curve(Y,C,P,par);curve.update(subset=subset,h=h,condition=cond);curves.append(curve)
    for j,i in enumerate(ids):windowstats.append(dict(subset=subset,index=i,parent=par[j],h=h,condition=cond,displacement_ratio=float(np.linalg.norm(P[j]-C[j])/max(np.linalg.norm(Y[j]-C[j]),1e-12))))
    # Frozen-probe answer agreement AUROC; thresholds are E99B3 train/validation thresholds.
    predicted=predprob(pp);ans=(reference>=head['thresholds']).astype(int)
    for tag,mask in [('all',np.ones(len(ids),bool)),('mixed',mult)]:
     if not mask.any():
      semantic.append(dict(subset=subset,h=h,condition=cond,group=tag,windows=0,parents=0,macro_auroc=None,ci95=None,per_predicate=[]));continue
     wa,wb,pa=weights(par[mask]);au=[];per=[];boots=[]
     for j,pid in enumerate(pids):
      v=auc(ans[mask,j],predicted[mask,j],wa)
      if v is None:continue
      bs=[auc(ans[mask,j],predicted[mask,j],ww) for ww in wb];au.append(v);boots.append(np.array([np.nan if a is None else a for a in bs]));per.append(dict(predicate=pid,auroc=v,ci95=interval(bs),yes=int(ans[mask,j].sum()),no=int((1-ans[mask,j]).sum())))
     macro=np.nanmean(np.stack(boots),0) if boots else np.array([])
     semantic.append(dict(subset=subset,h=h,condition=cond,group=tag,windows=int(mask.sum()),parents=len(pa),macro_auroc=float(np.mean(au)) if au else None,ci95=interval(macro),per_predicate=per))
    gpp=gripprob(pp)
    for tag,mask in [('all',np.ones(len(ids),bool)),('mixed',mult)]:
     if not mask.any():landings.append(dict(subset=subset,h=h,condition=cond,group=tag,windows=0,parents=0));continue
     wa,wb,pa=weights(par[mask]);mid=((gpp[mask]>=.3)&(gpp[mask]<=.7)).mean(1);tmid=((gptrue[mask]>=.3)&(gptrue[mask]<=.7)).mean(1)
     landings.append(dict(subset=subset,h=h,condition=cond,group=tag,windows=int(mask.sum()),parents=len(pa),pred_middle=estimate(mid,wa,wb),real_middle=estimate(tmid,wa,wb),middle_excess=estimate(mid-tmid,wa,wb)))
    if subset=='all27':hist[h,cond]=(gpp,gptrue,mult)
    # All two-sample distribution baselines and the paired small-noise reference.
    dist=distribution(zz,pz,nz,par,band,ridge,w0,w);dist.update(subset=subset,h=h,condition=cond,windows=len(ids),parents=len(ps));distributions.append(dist)
    print('cell',subset,h,cond,'variance',curve['variance_ratio']['value'],'distribution',dist['mean']['mmd']['value'],flush=True)
    put('curves_partial.json',curves);put('distribution_partial.json',distributions)
   report('## 已完成 '+subset+' +'+str(h)+'\n收缩、分布、探针与谓词读数均已完成；详见JSON逐格CI。\n')
 put('curves.json',curves);put('distribution.json',distributions);put('landings.json',landings);put('semantic.json',semantic);put('per_window_displacement.json',windowstats)
 makeplots(curves,brsummary,hist)
 # Report tables and substantive limits.
 lines=['## 收缩曲线与完整基线','![收缩](../results/'+R.name+'/shrinkage.png)','范数、cos使用原始绝对token；delta指标使用P−current与Y−current。variance_ratio是窗口分布逐token逐维方差的trace比，另存逐维比的中位与10/90分位。各parent等权，bootstrap同parent成对2000次seed90。','|集|h|条件|窗/parent|范数比|方差比[95%CI]|位移范数比|位移方差比|绝对cos/位移cos|','|---|---:|---|---|---:|---|---:|---:|---|']
 fmt=lambda z:f"{z['value']:.4f} [{z['ci95'][0]:.4f},{z['ci95'][1]:.4f}]" if z.get('ci95') is not None else 'NA'
 for z in curves:lines.append(f"|{z['subset']}|{z['h']}|{z['condition']}|{z['windows']}/{z['parents']}|{z['norm_ratio']['value']:.4f}|{fmt(z['variance_ratio'])}|{fmt(z['delta_norm_ratio'])}|{fmt(z['delta_variance_ratio'])}|{z['cos']['value']:.4f}/{z['delta_cos']['value']:.4f}|")
 lines+=['','## 分叉比例与落点','![分叉](../results/'+R.name+'/branch.png)','![落点](../results/'+R.name+'/probe_histograms.png)','|h|候选窗|近邻合格窗|候选且混合窗/parent|全体占比[CI]|所有近邻混合占比[CI]|','|---:|---:|---:|---|---|---|']
 for z in brsummary:lines.append(f"|{z['h']}|{z['candidate_windows']}|{z['eligible_windows']}|{z['true_windows']}/{z['true_parents']}|{fmt(z['fraction_all'])}|{fmt(z['mixed_all_fraction'])}|")
 lines+=['','落点按两爪取平均，逐爪直方图保留。真实未来探针读数的双峰性需要观察验证；图形或开合混合不能单独证明视觉特征有两峰。','|h|条件|混合窗/parent|预测中间率[CI]|真实中间率[CI]|差[CI]|','|---:|---|---|---|---|---|']
 for z in landings:
  if z['subset']=='all27' and z['group']=='mixed':lines.append(f"|{z['h']}|{z['condition']}|{z['windows']}/{z['parents']}|"+('|'.join(fmt(z[k]) for k in ('pred_middle','real_middle','middle_excess')) if z['windows'] else 'NA|NA|NA')+'|')
 lines+=['','## 特征空间分布指标基线表','训练PCA32（2304维三相机池化特征），解释方差'+str(float((sv[:32]**2).sum()/(sv**2).sum()))+'；非解码空间。RBF核exp(−||x−y||²/(2σ²))，σ=训练两两距离中位数='+str(band)+'；biased MMD²允许同一真实集合比较（噪声参照有配对乐观偏差）。kNN精确率/召回率k=3，原样本support半径固定后parent加权CI，未把bootstrap重复样本当新增邻居。C2ST为5折parent隔离、fold内标准化、固定C=.01线性LR；CI条件于既有OOF拟合。Fréchet=均值距²+协方差Bures项；entropy=.5logdet(2πe(Cov+ridge I))，仅PCA高斯体积代理，不是条件多峰熵。噪声=真实未来+每维0.05×训练std的独立高斯，seed90。','|集|h|条件/参照|MMD²[CI]|P/R[CI]|C2ST acc[CI]|Fréchet[CI]|entropy proxy[CI]|','|---|---:|---|---|---|---|---|---|']
 for z in distributions:
  for k in ('mean','noise'):
   a=z[k];lines.append(f"|{z['subset']}|{z['h']}|{z['condition']}/{k}|{fmt(a['mmd'])}|{fmt(a['precision'])} / {fmt(a['recall'])}|{fmt(a['c2st'])}|{fmt(a['frechet'])}|{fmt(a['entropy'])}|")
 lines+=['','所有mean−noise配对差及CI见distribution.json；同一parent抽样同时重权真实/预测/噪声；非独立窗bootstrap。','## 谓词语义（冻结探针回答一致性代理）','E99B3 SigLIP2+LR原样复用（原训练/内部验证阈值），二值参考=真实未来特征上的冻结探针答案，预测分数=均值未来特征上的同探针概率。未读取dev_labels、stages_dev或模型dev语言。AUROC只量化探针答案保真，不是人工语义准确率；不能用于checkpoint选型。单一类别谓词跳过，逐谓词yes/no数和CI见semantic.json。','|h|条件|组|窗/parent|宏AUROC[CI]|有效谓词数|','|---:|---|---|---|---|---:|']
 for z in semantic:
  if z['subset']=='all27':lines.append(f"|{z['h']}|{z['condition']}|{z['group']}|{z['windows']}/{z['parents']}|"+(f"{z['macro_auroc']:.4f} {z['ci95']}" if z['macro_auroc'] is not None else 'NA')+f"|{len(z['per_predicate'])}|")
 report('\n'.join(lines))
 put('ANALYSIS_COMPLETE.json',dict(status='complete',cells=len(curves),gpu_training=False))

def distribution(y,p,noise,parents,band,ridge,w0,w):
 n,d=y.shape;distyy=cdist(y,y,'sqeuclidean');kyy=np.exp(-distyy/(2*band**2));rady=np.partition(distyy,3,axis=1)[:,3]
 def c2st(a):
  ps=np.unique(parents);ps=np.random.default_rng(90).permutation(ps);fold={p:i%5 for i,p in enumerate(ps)};correct=np.zeros(n)
  for f in range(5):
   test=np.array([fold[p]==f for p in parents]);tr=~test
   xx=np.r_[y[tr],a[tr]];target=np.r_[np.zeros(tr.sum()),np.ones(tr.sum())];sc=Scaler().fit(xx);m=LR(C=.01,max_iter=250).fit(sc.transform(xx),target)
   correct[test]=.5*((m.predict_proba(sc.transform(y[test]))[:,1]<.5)+(m.predict_proba(sc.transform(a[test]))[:,1]>=.5).astype(float))
  return correct
 def moments(a):
  means=np.vstack([w0@a,w@a]);second=np.einsum('ni,nj->nij',a,a);secs=np.tensordot(np.vstack([w0,w]),second,axes=(1,0));cov=secs-np.einsum('bi,bj->bij',means,means);return means,cov
 ym,ycov=moments(y)
 def metrics(a):
  daa=cdist(a,a,'sqeuclidean');dya=cdist(y,a,'sqeuclidean');ra=np.partition(daa,3,axis=1)[:,3]
  precision=(dya<=rady[:,None]).any(0).astype(float);recall=(dya<=ra[None,:]).any(1).astype(float)
  k=np.exp(-daa/(2*band**2))+kyy-np.exp(-dya/(2*band**2))-np.exp(-dya.T/(2*band**2))
  ww=np.vstack([w0,w]);mmd=np.einsum('bi,ij,bj->b',ww,k,ww,optimize=True);am,acov=moments(a);fd=[];ent=[]
  for j,(cy,ca) in enumerate(zip(ycov,acov)):
   eig,U=np.linalg.eigh((cy+cy.T)/2);sy=(U*np.sqrt(np.maximum(eig,0)))@U.T;inner=sy@ca@sy;cross=np.sqrt(np.maximum(np.linalg.eigvalsh((inner+inner.T)/2),0)).sum();fd.append(float(np.sum((ym[j]-am[j])**2)+np.trace(cy)+np.trace(ca)-2*cross));ent.append(float(.5*(d*np.log(2*np.pi*np.e)+np.linalg.slogdet(ca+ridge*np.eye(d))[1])))
  score=c2st(a);out={}
  for k,v in dict(mmd=mmd,frechet=fd,entropy=ent).items():out[k]=dict(value=float(v[0]),ci95=interval(v[1:]))
  out.update(precision=estimate(precision,w0,w),recall=estimate(recall,w0,w),c2st=estimate(score,w0,w))
  boots=dict(mmd=mmd[1:],frechet=np.array(fd[1:]),entropy=np.array(ent[1:]),precision=w@precision,recall=w@recall,c2st=w@score)
  return out,boots
 a,ab=metrics(p);b,bb=metrics(noise);delta={k:dict(value=a[k]['value']-b[k]['value'],ci95=interval(ab[k]-bb[k])) for k in a}
 return dict(mean=a,noise=b,paired_mean_minus_noise=delta)

def makeplots(curves,branches,hist):
 plt.rcParams.update({'font.size':9})
 fig,axs=plt.subplots(1,3,figsize=(13,3.5))
 for cond in ('no_traj','gt_action'):
  z=[v for v in curves if v['subset']=='all27' and v['condition']==cond]
  for ax,key in zip(axs,['variance_ratio','delta_variance_ratio','delta_norm_ratio']):
   vals=[v[key]['value'] for v in z];lo=[v[key]['ci95'][0] for v in z];hi=[v[key]['ci95'][1] for v in z];ax.plot([v['h'] for v in z],vals,'o-',label=cond);ax.fill_between([v['h'] for v in z],lo,hi,alpha=.15);ax.axhline(1,color='gray',linestyle='--');ax.set_title(key);ax.set_xlabel('horizon frames')
 axs[0].legend();fig.tight_layout();fig.savefig(R/'shrinkage.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,3.5));v=[z['fraction_all']['value'] for z in branches];lo=[z['fraction_all']['ci95'][0] for z in branches];hi=[z['fraction_all']['ci95'][1] for z in branches];ax.bar([str(z['h']) for z in branches],v);ax.errorbar(range(3),v,yerr=[np.array(v)-lo,np.array(hi)-v],fmt='none',color='k');ax.axhline(.1,color='r',linestyle='--');ax.set_ylabel('Mixed pre-event anchor fraction');ax.set_xlabel('horizon frames (mechanical-phase proxy)');fig.tight_layout();fig.savefig(R/'branch.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(4,4,figsize=(15,11))
 for row,h in enumerate((10,25,50,100)):
  for j,cond in enumerate(('no_traj','gt_action')):
   a,b,mask=hist[h,cond]
   for k,sel in enumerate((np.ones(len(a),bool),mask)):
    ax=axs[row,j*2+k];ax.hist(b[sel].ravel(),bins=np.linspace(0,1,21),alpha=.55,label='real future');ax.hist(a[sel].ravel(),bins=np.linspace(0,1,21),alpha=.55,label='mean prediction');ax.axvspan(.3,.7,alpha=.08,color='black');ax.set_title(f'+{h} {cond} '+('all' if k==0 else 'mixed')+f' n={sel.sum()}');ax.set_xlim(0,1)
 axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(R/'probe_histograms.png',dpi=150);plt.close(fig)
if __name__=='__main__':
 try:main()
 except BaseException:put('ANALYSIS_ERROR.json',dict(traceback=traceback.format_exc()));raise
