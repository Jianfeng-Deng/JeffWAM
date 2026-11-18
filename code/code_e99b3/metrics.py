from common import *
import numpy as np,pickle,collections,warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
PIDS=[p['id'] for p in read('spec.json')['predicates']];PI={p:i for i,p in enumerate(PIDS)}
METRICS=['AUROC','F1_yes','F1_no','F1_macro','balanced_accuracy','coverage','known_coverage','unknown_recall','unknown_precision','selective_balanced_accuracy']
def div(a,b):return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b!=0)
def finite(v):return float(v) if np.isfinite(v) else None
def mean(v,axis=0):
 n=np.isfinite(v).sum(axis);return div(np.nansum(v,axis),n)
def interval(a):
 a=a[np.isfinite(a)];return np.quantile(a,[.025,.975]).tolist() if len(a) else [None,None]
def stats_rows(y,p,ab,threshold,parent_indices,draws):
 # Row-resampled parent bootstrap represented by integer multiplicities. First row is unresampled point estimate.
 numparents=draws.shape[1];weights=np.concatenate([np.ones((1,numparents)),draws],0);binary=(y==0)|(y==1);pred=p>=threshold;known=~ab
 indicators=np.stack([binary&(y==1)&pred,binary&(y==0)&~pred,binary&(y==0)&pred,binary&(y==1)&~pred,known,y!=2,binary,binary&known,(y==2)&ab,y==2,ab,binary&(y==1)&pred&known,binary&(y==1)&~pred&known,binary&(y==0)&~pred&known,binary&(y==0)&pred&known],1).astype('f8')
 by=np.zeros((numparents,indicators.shape[1]));np.add.at(by,parent_indices,indicators);v=weights@by
 tp,tn,fp,fn=v[:,0],v[:,1],v[:,2],v[:,3];fy=div(2*tp,2*tp+fp+fn);fnn=div(2*tn,2*tn+fp+fn);ba=(div(tp,tp+fn)+div(tn,tn+fp))/2
 cover=div(v[:,4],v[:,5]+v[:,9]);kc=div(v[:,7],v[:,6]);ur=div(v[:,8],v[:,9]);up=div(v[:,8],v[:,10]);sba=(div(v[:,11],v[:,11]+v[:,12])+div(v[:,13],v[:,13]+v[:,14]))/2
 auc=np.full(len(weights),np.nan)
 if binary.any():
  yy=y[binary];pp=p[binary];ii=parent_indices[binary];order=np.argsort(pp,kind='stable');yy=yy[order];pp=pp[order];ii=ii[order];starts=np.r_[0,np.flatnonzero(np.diff(pp)!=0)+1]
  for st in range(0,len(weights),100):
   w=weights[st:st+100,ii];wy=np.add.reduceat(w*yy[None],starts,axis=1);wn=np.add.reduceat(w*(1-yy[None]),starts,axis=1);num=(wy*(np.cumsum(wn,1)-wn*.5)).sum(1);auc[st:st+len(w)]=div(num,wy.sum(1)*wn.sum(1))
 return np.stack([auc,fy,fnn,(fy+fnn)/2,ba,cover,kc,ur,up,sba],1)
def main():
 assert (R/'VLM_COMPLETE.json').exists();assert not (R/'METRICS_COMPLETE.json').exists()
 with (R/'heads.pkl').open('rb') as f:heads=pickle.load(f)
 parents=sorted({r['parent'] for r in read('dev_labels.json')});pari={p:i for i,p in enumerate(parents)};rng=np.random.default_rng(9903);sample=rng.integers(len(parents),size=(2000,len(parents)));draws=np.stack([np.bincount(v,minlength=len(parents)) for v in sample]);np.save(R/'bootstrap_parent_draws.npy',sample)
 lookup={m:{} for m in ('a','b','c','d')}
 for s in read('sources.json'):
  if s['pool']!='dev':continue
  z=np.load(R/(s['parent']+'_scores.npz'));vlm=np.load(R/(s['parent']+'_vlm.npz'))
  for method in ('b','c','d'):
   for f,p,u in zip(z['frames'],z[method+'_prob'],z[method+'_unknown']):lookup[method][s['parent'],int(f)]=(p,u>=.5)
  for f,p in zip(vlm['frames'],vlm['prob']):
   ab=np.zeros(21,bool);ab[PI['P073']]=True;lookup['a'][s['parent'],int(f)]=(p,ab)
 labels=read('dev_labels.json');results={};predictions=[]
 for name,select,answerkey in [('causal_main',lambda r:r['causal_valid'],'causal_answer'),('retrospective_event',lambda r:r['semantics']=='retrospective_event','answer'),('all_answer_mixed_appendix',lambda r:True,'answer')]:
  rr=[r for r in labels if select(r)];meta=dict(records=len(rr),parents=len({r['parent'] for r in rr}),answer_counts=dict(collections.Counter(r[answerkey] for r in rr)),label_basis='基于模型标注的 dev 真值',ci='2000 parent bootstrap, row-weighted clusters, macro across estimable predicates')
  groups={'all':np.ones(len(rr),bool)};viskey='causal_visibility' if name=='causal_main' else 'visibility';stagekey='causal_stage' if name=='causal_main' else 'stage'
  groups.update({'visibility:'+v:np.array([r[viskey]==v for r in rr]) for v in ('visible','partial','hidden')});groups.update({'stage:'+v:np.array([r[stagekey]==v for r in rr]) for v in ['S%02d'%i for i in range(1,11)]})
  results[name]=dict(meta=meta,methods={},paired_d_minus_c={});comparison_arrays={}
  for method in ('a','b','c','d'):
   th=np.full(21,.5) if method=='a' else heads[method]['thresholds'];p=np.array([lookup[method][r['parent'],r['frame']][0][PI[r['predicate']]] for r in rr]);ab=np.array([lookup[method][r['parent'],r['frame']][1][PI[r['predicate']]] for r in rr]);y=np.array([{'no':0,'yes':1,'unknown':2,'not_applicable':3}[r[answerkey]] for r in rr]);pid=np.array([PI[r['predicate']] for r in rr]);pp=np.array([pari[r['parent']] for r in rr]);out={}
   for g,sel in groups.items():
    # NA excluded from unknown/coverage denominators, never relabeled no.
    sel=sel&(y!=3);by_pred={};values=[]
    for j,predicate in enumerate(PIDS):
     mask=sel&(pid==j)
     a=stats_rows(y[mask],p[mask],ab[mask],th[j],pp[mask],draws);values.append(a);
     if method=='c':comparison_arrays[g,predicate]=a.copy()
     if method=='d' and (g,predicate) in comparison_arrays:
      delta=a-comparison_arrays[g,predicate];results[name]['paired_d_minus_c'].setdefault(g,dict(per_predicate={}))["per_predicate"][predicate]={k:dict(value=finite(delta[0,i]),ci95=interval(delta[1:,i])) for i,k in enumerate(METRICS)}
     by_pred[predicate]=dict(NA=int((groups[g]&(pid==j)&(y==3)).sum()),n=int(mask.sum()),yes=int((y[mask]==1).sum()),no=int((y[mask]==0).sum()),unknown=int((y[mask]==2).sum()),parents=len(set(pp[mask])),metrics={k:dict(value=finite(a[0,i]),ci95=interval(a[1:,i])) for i,k in enumerate(METRICS)})
    if values:
     a=mean(np.stack(values),0);macro={k:dict(value=finite(a[0,i]),ci95=interval(a[1:,i]),estimable_predicates=int(sum(np.isfinite(v[0,i]) for v in values))) for i,k in enumerate(METRICS)}
    else:macro={}
    if values and method=='c':comparison_arrays[g,'macro']=a.copy()
    if values and method=='d' and (g,'macro') in comparison_arrays:
     delta=a-comparison_arrays[g,'macro'];results[name]['paired_d_minus_c'].setdefault(g,dict(per_predicate={}))['macro']={k:dict(value=finite(delta[0,i]),ci95=interval(delta[1:,i])) for i,k in enumerate(METRICS)}
    out[g]=dict(macro=macro,per_predicate=by_pred,n=int(sel.sum()))
   # pooled coverage complements macro coverage and gives exact reference denominator.
   usable=y!=3;knownref=(y<2);unref=y==2
   out['coverage_pooled']=dict(non_NA=int(usable.sum()),covered=int((~ab&usable).sum()),coverage=float((~ab&usable).sum()/max(usable.sum(),1)),known_reference=int(knownref.sum()),known_covered=int((knownref&~ab).sum()),unknown_reference=int(unref.sum()),unknown_abstained=int((unref&ab).sum()),NA=int((y==3).sum()))
   results[name]['methods'][method]=out
   status('metrics',pool=name,method=method,records=len(rr))
   if name=='causal_main':
    for r,score,abstain in zip(rr,p,ab):predictions.append(dict(method=method,parent=r['parent'],frame=r['frame'],predicate=r['predicate'],answer=r['causal_answer'],probability=float(score),abstain=bool(abstain)))
 put('metrics.json',results)
 with (R/'causal_predictions.jsonl').open('x') as f:
  for r in predictions:f.write(json.dumps(r)+'\n')
 put('METRICS_COMPLETE.json',dict(brt=now(),bootstrap=2000,seed=9903,no_dev_selection=True))
 lines=['## dev 一次冻结结果 '+now(),'所有下列结果均为 **基于模型标注的 dev 真值**。AUROC/BA 不可估计的谓词不填0；宏均值只计可估计项。二分类主指标不因模型弃权删去参考 yes/no，选择性 BA 与覆盖率另存。','|集合|方法|AUROC [95%CI]|yes F1|no F1|平衡准确率|非NA覆盖率|','|---|---|---|---|---|---|---|']
 for pool in results:
  for method,d in results[pool]['methods'].items():
   m=d['all']['macro'];a=m['AUROC'];ci=a['ci95'];fmt=lambda v:'NA' if v is None else f'{v:.4f}'
   lines.append('|'+ '|'.join([pool,method,fmt(a['value'])+' ['+', '.join(map(fmt,ci))+']',fmt(m['F1_yes']['value']),fmt(m['F1_no']['value']),fmt(m['balanced_accuracy']['value']),fmt(d['coverage_pooled']['coverage'])])+'|')
 report('\n'.join(lines))
if __name__=='__main__':main()
