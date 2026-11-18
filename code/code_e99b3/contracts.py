from common import *
import numpy as np
from metrics import stats_rows
from controller import replay,PI
from linear import LogisticRegression,StandardScaler
# Independent brute-force positive-negative pairs validate tie-aware cluster weighted AUC.
y=np.array([0,1,0,1,1,0]);p=np.array([.2,.8,.5,.5,.9,.8]);parents=np.array([0,0,1,1,2,2]);draw=np.array([[1,2,0],[2,0,1]])
a=stats_rows(y,p,np.zeros(6,bool),.5,parents,draw)
for k,ww in enumerate([np.ones(3),*draw]):
 w=ww[parents];num=den=0.
 for i in np.flatnonzero(y==1):
  for j in np.flatnonzero(y==0):num+=w[i]*w[j]*((p[i]>p[j])+.5*(p[i]==p[j]));den+=w[i]*w[j]
 assert abs(a[k,0]-num/den)<1e-12
# Hand-calculated unweighted confusion: TP3 TN1 FP2 FN0 -> yes F1=.75, no F1=.5, BA=2/3.
assert np.max(abs(a[0,[1,2,4]]-np.array([.75,.5,2/3])))<1e-12
p=np.ones((3,21));u=np.ones((3,21));assert replay(p,u,np.full(21,.5),1)==['S01']*3
u[:]=0;assert replay(p,u,np.full(21,.5),1)[0]=='S10'
p[:]=0;p[:,PI['P011']]=1;assert replay(p,u,np.full(21,.5),1)==['S02']*3
# Convex logistic fit separates a linear synthetic problem, finite optimum and zero-mean train scaling.
x=np.arange(-20,20).reshape(-1,1).astype('f4');ys=(x[:,0]>=0).astype(float);sc=StandardScaler().fit(x);xx=sc.transform(x);h=LogisticRegression(C=1).fit(xx,ys);assert np.mean((h.predict_proba(xx)[:,1]>=.5)==ys)==1 and abs(xx.mean())<1e-12;assert h.optimization_['gradient_max_abs']<1e-4
put('CPU_CONTRACTS.json',dict(status='passed',checks=['weighted_tied_AUC_vs_independent_pairwise','analytic_yes_no_F1_BA','unknown_holds','failure_priority','one_transition_per_frame','convex_logistic_separability_and_stationarity','train_scaler'],brt=now()))
print('CPU contracts passed')
