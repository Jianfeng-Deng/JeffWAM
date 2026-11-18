from train import *
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from numba import njit
S=[f'S{i:02}' for i in range(1,11)];SI={s:i for i,s in enumerate(S)};SPEC=oldread('spec.json')
stages={pool:oldread('stages_'+pool+'.json') for pool in ['train','dev']}
def truth(s):
 y=np.full(s['n'],-1,int)
 for z in stages[s['pool']]:
  if z['parent_id']==s['parent']:y[z['stage_start']:z['stage_end_exclusive']]=SI.get(z['stage_id'],-1)
 return y
truths={s['parent']:truth(s) for s in sources}
_cid2p={s['cid']:s['parent'] for s in sources}
CAUSAL_STAGE={}
for _r in jl(T/'data/e98_predicate_labels_train_causal_v1/stage_causal.jsonl'):
 _p=_cid2p.get(_r['canonical_id'])
 if _p is not None and (_p,_r['source_frame']) not in EXCL:CAUSAL_STAGE[_p,_r['source_frame']]=SI.get(_r['stage'],-1)
A=np.eye(10,dtype=bool)
for t in SPEC['transitions']:
 if t['source'] in SI and t['target'] in SI:A[SI[t['source']],SI[t['target']]]=1

def calibration(p,y):
 # Unweighted Platt calibration restores empirical prevalence after class-balanced head fit.
 z=np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1));out=[]
 for j in range(21):
  mask=np.isfinite(y[:,j]);a=z[mask,j];b=y[mask,j]
  if len(np.unique(b))<2:out.append([0.,float(np.log((b.sum()+1)/(len(b)-b.sum()+1)))]);continue
  def fun(v):
   zz=a*v[0]+v[1];rr=expit(zz)-b
   return np.mean(np.logaddexp(0,zz)-b*zz)+.005*v[0]**2,np.array([np.mean(rr*a)+.01*v[0],rr.mean()])
  opt=minimize(fun,[1.,0.],jac=True,method='L-BFGS-B');out.append(opt.x.tolist())
 return np.array(out)

def calibrate(p,c):return expit(np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1))*c[:,0]+c[:,1])

def stagefit(x,y):
 # Softmax linear emission model on calibrated predicate scores; P073 removed entirely.
 good=y>=0;x=x[good];y=y[good];xx=np.c_[x,np.ones(len(x))];yy=np.eye(10)[y];prior=(yy.sum(0)+1)/(len(y)+10)
 def fun(v):
  w=v.reshape(22,10);z=xx@w;pr=np.exp(z-logsumexp(z,axis=1,keepdims=True));loss=np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y])+.001*(w[:-1]**2).sum();g=xx.T@(pr-yy)/len(y);g[:-1]+=.002*w[:-1];return loss,g.ravel()
 opt=minimize(fun,np.zeros(220),jac=True,method='L-BFGS-B',options={'maxiter':300});return opt.x.reshape(22,10),prior

def transition(excluded):
 counts=np.zeros((10,10));dur={s:[] for s in S};illegal=[]
 for s in train:
  if fold[s['parent']]==excluded:continue
  y=truths[s['parent']];valid=(y[:-1]>=0)&(y[1:]>=0)
  for a,b in zip(y[:-1][valid],y[1:][valid]):counts[a,b]+=1
  idx=np.r_[0,np.where(y[1:]!=y[:-1])[0]+1,len(y)]
  for a,b in zip(idx[:-1],idx[1:]):
   if y[a]>=0:dur[S[y[a]]].append(int(b-a))
 for a,b in zip(*np.where((counts>0)&~A)):illegal.append(dict(source=S[a],target=S[b],count=int(counts[a,b])))
 counts=np.where(A,counts+.1,0);counts/=counts.sum(1,keepdims=True)
 return counts,dur,illegal

def build_aux(method,excluded):
 z=np.load(R/(method+'_annotation_oof.npz'));mask=np.array([fold[p]!=excluded for p in z['parent']]);p=z['prob'][mask];y=z['y'][mask];c=calibration(p,y);x=calibrate(p,c);x[:,PI['P073']]=.5
 sy=np.array([(CAUSAL_STAGE.get((p,f),-1) if os.environ.get('STAGE_SRC')=='causal' else truths[p][f]) for p,f in zip(z['parent'][mask],z['frame'][mask])]);w,prior=stagefit(x,sy);tr,dur,illegal=transition(excluded)
 return dict(cal=c,w=w,prior=prior,transition=tr,durations=dur,illegal=illegal)

def emission(p,aux):
 x=calibrate(p,aux['cal']);x[:,PI['P073']]=.5;z=np.c_[x,np.ones(len(x))]@aux['w'];return z-logsumexp(z,axis=1,keepdims=True)-np.log(aux['prior'])

def boundaries(y):return [(i,int(y[i-1]),int(y[i])) for i in range(1,len(y)) if y[i]!=y[i-1] and min(y[i-1],y[i])>=0]

def match(pp,tt,tol):
 # Per transition type sorted DP, one-to-one max cardinality then min total absolute error.
 matched=[]
 for typ in sorted(set((a,b) for i,a,b in pp)|set((a,b) for i,a,b in tt)):
  p=[(k,i) for k,(i,a,b) in enumerate(pp) if (a,b)==typ];t=[(k,i) for k,(i,a,b) in enumerate(tt) if (a,b)==typ];n=len(p);m=len(t);dp=[[ (0,0,[]) for _ in range(m+1)] for _ in range(n+1)]
  for i in range(1,n+1):
   for j in range(1,m+1):
    opts=[dp[i-1][j],dp[i][j-1]]
    if abs(p[i-1][1]-t[j-1][1])<=tol:
     v=dp[i-1][j-1];opts.append((v[0]+1,v[1]-abs(p[i-1][1]-t[j-1][1]),v[2]+[(p[i-1][0],t[j-1][0])]))
    dp[i][j]=max(opts,key=lambda a:(a[0],a[1]))
  matched+=dp[n][m][2]
 return matched

def record(pred,true):
 mask=true>=0;cm=np.zeros((10,11),int);np.add.at(cm,(true[mask],pred[mask]),1);pp=boundaries(pred);tt=boundaries(true);out=dict(cm=cm.tolist(),frames=len(true),pred=len(pp),true=len(tt),transitions={})
 for tol in (15,30):
  pairs=match(pp,tt,tol);errs=[pp[i][0]-tt[j][0] for i,j in pairs];pi={i for i,j in pairs};ti={j for i,j in pairs};miss={};extra={}
  for idx,(f,a,b) in enumerate(tt):
   if idx not in ti:miss[str((a,b))]=miss.get(str((a,b)),0)+1
  for idx,(f,a,b) in enumerate(pp):
   if idx not in pi:extra[str((a,b))]=extra.get(str((a,b)),0)+1
  out['transitions'][str(tol)]=dict(matched=len(pairs),errors=errs,miss=miss,extra=extra)
 return out

def aggregate(rr):
 cm=np.sum([r['cm'] for r in rr],axis=0);tp=np.diag(cm[:,:10]);den=cm.sum(1)+cm[:,:10].sum(0);f1=np.divide(2*tp,den,out=np.zeros(10,float),where=den>0);out=dict(accuracy=float(tp.sum()/max(cm.sum(),1)),macro_f1=float(f1.mean()))
 for tol in ('15','30'):
  mt=sum(r['transitions'][tol]['matched'] for r in rr);nt=sum(r['true'] for r in rr);npred=sum(r['pred'] for r in rr);err=np.array([e for r in rr for e in r['transitions'][tol]['errors']]);late=err[err>=0]
  out.update({f'precision{tol}':mt/max(npred,1),f'recall{tol}':mt/max(nt,1),f'f1_{tol}':2*mt/max(nt+npred,1),f'error_median{tol}':float(np.median(abs(err))) if len(err) else None,f'error_p90{tol}':float(np.quantile(abs(err),.9)) if len(err) else None,f'signed_delay_median{tol}':float(np.median(err)) if len(err) else None,f'nonnegative_delay_median{tol}':float(np.median(late)) if len(late) else None,f'false_per_min{tol}':(npred-mt)/max(sum(r['frames'] for r in rr)/1800,1e-9),f'matched{tol}':mt})
 return out
