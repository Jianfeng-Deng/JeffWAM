from common import *
import traceback
os.environ['OPENBLAS_NUM_THREADS']='2'
from threadpoolctl import threadpool_limits
threadpool_limits(limits=2);torch.set_num_threads(2);started=time.monotonic()
try:
 while not (R/'cpu_status.json').exists() or json.loads((R/'cpu_status.json').read_text())['status']=='evaluating':time.sleep(5)
 assert json.loads((R/'cpu_status.json').read_text())['status']=='completed'
 train,tr,dev=get_data();by=collections.defaultdict(list)
 for i,r in enumerate(train.rows):by[r['parent_component']].append(i)
 selected=[]
 for p,ids in sorted(by.items()):selected.extend(ids[j] for j in np.linspace(0,len(ids)-1,min(12,len(ids))).astype(int))
 parents=sorted(by);rng=np.random.default_rng(9305);validation=set(rng.choice(parents,max(1,len(parents)//5),replace=False).tolist());pool={};rawcur=[]
 # Preload <=12 original windows per training parent; no disk feature duplication.
 for name,data,ids in [('fit',train,selected),('dev',dev,list(range(len(dev.rows))))]:
  vals=[]
  for idx,i in enumerate(ids):
   b=data.get(i,H,'short','P1');obs=b['obs'].numpy();hv=b['history_valid'].numpy();vals.append((i,obs,hv,b))
   if name=='fit' and data.rows[i]['parent_component'] not in validation:rawcur.append(obs[4].reshape(-1,768))
   if idx%128==0:atomic(R/'linear_status.json',dict(status='loading',pool=name,done=idx,total=len(ids)))
  pool[name]=(data,vals)
 # Train-only PCA64, kept fixed across alpha selection and final refit.
 c=np.concatenate(rawcur);mu=c.mean(0);centered=c-mu;cov=centered.T@centered/len(c);ev,U=np.linalg.eigh(cov);U=U[:,-64:].astype('f4');del c,centered,rawcur
 allrecords=[];meta=[]
 for h in H:
  xys={}
  for name,(data,vals) in pool.items():
   xs=[];ys=[];ident=[];nt=8 if h<=50 else 1;layout=np.eye(3*nt,dtype='f4')
   for i,obs,hv,b in vals:
    offset=H.index(h)*24 if h<=50 else 72;n=3*nt;sl=slice(offset,offset+n)
    if not b['world_valid'][sl].all():continue
    zz=obs if h<=50 else obs.mean(2,keepdims=True);cur=zz[4].reshape(n,768);hist=zz[:4].reshape(4,n,768);delta=hist-cur[None];delta[~hv]=0
    feat=np.concatenate([(cur-mu)@U,*[delta[j]@U for j in range(4)],np.broadcast_to(hv.astype('f4'),(n,4)),layout],1)
    xs.append(feat);ys.append(b['world'][sl].numpy());ident.append(dict(index=i,row=data.rows[i],b=b,sl=sl,validation=data.rows[i]['parent_component'] in validation))
   xys[name]=(np.stack(xs),np.stack(ys),ident)
  X,Y,ident=xys['fit'];mask=np.array([not r['validation'] for r in ident]);d=X.shape[-1]
  xm=X[mask].reshape(-1,d).mean(0);std=np.maximum(X[mask].reshape(-1,d).std(0),.05);X=(X-xm)/std;Xd=(xys['dev'][0]-xm)/std
  A=X[mask].reshape(-1,d).astype('f8');B=Y[mask].reshape(-1,768).astype('f8');ym=B.mean(0);B-=ym;xtx=A.T@A/len(A);xty=A.T@B/len(A);scores=[];valX=X[~mask];valY=Y[~mask];val_ident=[r for r in ident if r['validation']]
  for alpha in (.0001,.001,.01,.1,1.,10.):
   W=np.linalg.solve(xtx+alpha*np.eye(d),xty);pred=valX@W+ym;errs=((pred-valY)**2).mean((1,2));par=collections.defaultdict(list)
   for r,e in zip(val_ident,errs):par[r['row']['parent_component']].append(e)
   scores.append((float(np.mean([np.mean(v) for v in par.values()])),alpha))
  alpha=min(scores)[1];Af=X.reshape(-1,d).astype('f8');Bf=Y.reshape(-1,768).astype('f8');ym=Bf.mean(0);W=np.linalg.solve(Af.T@Af/len(Af)+alpha*np.eye(d),Af.T@(Bf-ym)/len(Af));pred=Xd@W+ym
  # Low capacity full-coordinate linear AR: five scalar coefficients plus intercept,
  # separately fit per camera; complements PCA feature compression.
  # PCA model is linear in original current/history tokens, predicts all768 residual dimensions.
  for j,r in enumerate(xys['dev'][2]):
   b=r['b'];sl=r['sl'];cb={k:(v[sl][None] if k.startswith('world') and isinstance(v,torch.Tensor) and v.ndim>=1 else v) for k,v in b.items()};pr=cb['world_current']+torch.tensor(pred[j],dtype=torch.float32)[None]*cb['world_scale'];allrecords.append(rec(r['row'],h,pr,cb,index=r['index']))
  np.savez_compressed(R/f'ridge_h{h}_coefficients.npz',pca=U,pca_mean=mu,x_mean=xm,x_scale=std,coef=W.astype('f4'),intercept=ym.astype('f4'))
  meta.append(dict(h=h,fit_windows=len(ident),train_parents=len(parents),inner_validation_parents=len(validation),alpha=alpha,alpha_scores=scores,input='current token PCA64 + four same-position history-minus-current PCA64 + history-valid flags + camera/spatial onehot; linear in raw frozen features',output='full768 training-normalized residual',pca_explained_variance=float(ev[-64:].sum()/ev.sum()),dimension=d,no_dev_selection=True))
  atomic(R/'linear_status.json',dict(status='fitting',h=h,seconds=time.monotonic()-started));atomic(R/'linear_records.json',allrecords)
 atomic(R/'linear_summary.json',summary(allrecords));atomic(R/'linear_method.json',dict(horizons=meta,validation_parents=sorted(validation),selection_seed=9305,limitations='PCA64 is a restricted linear reference; failure cannot establish intrinsic unpredictability. No model or threshold selection on dev.'));atomic(R/'linear_status.json',dict(status='completed',seconds=time.monotonic()-started,GPUh=0))
except BaseException as e:atomic(R/'linear_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
