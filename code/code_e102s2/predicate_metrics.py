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
