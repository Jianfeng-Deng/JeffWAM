from .common import *
import numpy as np,torch,traceback,gc
import torch.nn.functional as F
from .data import load,Data
from .metrics import rec,summary,decomposition

def resize_obs(obs,grid):
 if grid==8:return obs
 x=torch.from_numpy(obs).reshape(15,2,4,768).permute(0,3,1,2)
 return F.interpolate(x,size=(7,7),mode='bilinear',align_corners=False).permute(0,2,3,1).numpy().reshape(5,3,49,768)
def linear_delta(data,row,b,h):
 nt=b['world'].shape[0]//3;jj=np.flatnonzero(b['history_valid'].numpy());j=int(jj[-1]) if len(jj) else 4
 previous=b['target_obs'][j].numpy()
 if h>50:
  # Global uses canonical 2x4, also for 7x7 diagnostics.
  from .data import Features
  f=Features(8);previous=np.stack([f.get(row['canonical_id'],k,[row['obs_rows'][j][k]])[0] for k in range(3)]) if j!=4 else b['world_current'].numpy().reshape(3,1,768)
  previous=previous.mean(1,keepdims=True)
 z,_=data.index.get(row['canonical_id']);ct=z['camera_exposure_timestamp'];rr,_=data.index.resolve(row,h);ratio=[]
 for k in range(3):
  now=row['obs_rows'][4][k];past=row['obs_rows'][j][k];dt=ct[now,k]-ct[past,k]
  ratio.append((ct[rr[k],k]-ct[now,k])/dt if dt>1e-7 else 0.)
 cur=b['world_current'].numpy();scale=b['world_scale'].numpy()
 return (cur-previous.reshape(-1,768))*np.repeat(ratio,nt)[:,None]/scale

def run(grid):
 started=time.monotonic();torch.set_num_threads(2);train,_,dev=load(grid);rows=json.loads((R/'ridge_train_rows.json').read_text());bykey={(r['canonical_id'],r['anchor']):i for i,r in enumerate(train.rows)};ids=[bykey[(r['canonical_id'],r['anchor'])] for r in rows]
 parents=sorted({r['parent_component'] for r in train.rows});rng=np.random.default_rng(9305);val=set(rng.choice(parents,len(parents)//5,replace=False).tolist())
 # As E93 B5: PCA fitted only inner-training SigLIP2 current O tokens. O remains unchanged.
 raw=[]
 for i in ids:
  if train.rows[i]['parent_component'] not in val:raw.append(train.base[i]['obs'][4].numpy().reshape(-1,768))
 c=np.concatenate(raw);mu=c.mean(0);centered=c-mu;cov=centered.T@centered/len(c);ev,U=np.linalg.eigh(cov);U=U[:,-64:].astype('f4');del c,centered,raw
 allrec=[];meta=[]
 for h in H:
  pools={}
  for name,data,indices in [('fit',train,ids),('dev',dev,range(len(dev.rows)))]:
   xs=[];ys=[];ident=[];lins=[];nt=grid if h<=50 else 1;n=3*nt;layout=np.eye(n,dtype='f4')
   for j,i in enumerate(indices):
    b=data.get(i,(h,),'short','P1')
    if not b['world_valid'].all():continue
    row=data.rows[i];obs=b['obs'].numpy();hv=b['history_valid'].numpy();obs=resize_obs(obs,grid) if h<=50 else obs.mean(2,keepdims=True);cur=obs[4].reshape(n,768);hist=obs[:4].reshape(4,n,768);delta=hist-cur[None];delta[~hv]=0
    xs.append(np.concatenate([(cur-mu)@U,*[delta[j]@U for j in range(4)],np.broadcast_to(hv.astype('f4'),(n,4)),layout],1));ys.append(b['world'].numpy());ident.append(row)
    if name=='dev':lins.append(linear_delta(data,row,b,h))
    if j%256==0:atomic(R/f'ridge_{grid}_status.json',dict(status='loading',pool=name,h=h,done=j,seconds=time.monotonic()-started))
   pools[name]=(np.stack(xs),np.stack(ys),ident,np.stack(lins) if lins else None)
  X,Y,ident,_=pools['fit'];mask=np.array([r['parent_component'] not in val for r in ident]);d=X.shape[-1];xm=X[mask].reshape(-1,d).mean(0);std=np.maximum(X[mask].reshape(-1,d).std(0),.05);X=(X-xm)/std;Xd=(pools['dev'][0]-xm)/std
  A=X[mask].reshape(-1,d).astype('f8');B=Y[mask].reshape(-1,768).astype('f8');ym=B.mean(0);B-=ym;xtx=A.T@A/len(A);xty=A.T@B/len(A);scores=[];v_ident=[r for r in ident if r['parent_component'] in val]
  for alpha in (.0001,.001,.01,.1,1.,10.):
   W=np.linalg.solve(xtx+alpha*np.eye(d),xty);pred=X[~mask]@W+ym;errs=((pred-Y[~mask])**2).mean((1,2));p=collections.defaultdict(list)
   for r,e in zip(v_ident,errs):p[r['parent_component']].append(e)
   scores.append((float(np.mean([np.mean(z) for z in p.values()])),alpha))
  alpha=min(scores)[1];Af=X.reshape(-1,d).astype('f8');Bf=Y.reshape(-1,768).astype('f8');ym=Bf.mean(0);W=np.linalg.solve(Af.T@Af/len(Af)+alpha*np.eye(d),Af.T@(Bf-ym)/len(Af));pred=Xd@W+ym
  for j,row in enumerate(pools['dev'][2]):allrec.append(rec(row,h,pred[j],pools['dev'][1][j],pools['dev'][3][j]))
  np.savez_compressed(R/f'ridge_{grid}_h{h}_coefficients.npz',pca=U,pca_mean=mu,x_mean=xm,x_scale=std,coef=W.astype('f4'),intercept=ym.astype('f4'))
  meta.append(dict(h=h,fit_windows=len(ident),train_parents=len(parents),inner_validation_parents=len(val),alpha=alpha,scores=scores,pca_explained_variance=float(ev[-64:].sum()/ev.sum()),dimension=d,source_input='SigLIP2 O, same E93 B5 PCA64/4 histories/valid/camera-position onehot',fine_mapping='fixed bilinear SigLIP2 2x4 ->7x7; linear; h100 original canonical global' if grid==49 else 'original spatial correspondence',no_dev_selection=True))
  atomic(R/f'ridge_{grid}_records.json',allrec);atomic(R/f'ridge_{grid}_status.json',dict(status='fitting',h=h,seconds=time.monotonic()-started));del pools,X,Y,Xd,A,B,Af,Bf,pred;gc.collect()
 atomic(R/f'ridge_{grid}_summary.json',summary(allrec));atomic(R/f'ridge_{grid}_decomposition.json',decomposition(allrec));atomic(R/f'ridge_{grid}_method.json',dict(horizons=meta,validation_parents=sorted(val),seed=9305,grid=grid,dev_evaluations_per_final_fit=1));atomic(R/f'ridge_{grid}_status.json',dict(status='completed',seconds=time.monotonic()-started,GPUh=0))
 report('## 3 '+str(grid)+'token 岭回归完成\n'+json.dumps(summary(allrec),ensure_ascii=False));print('ridge complete',grid,round(time.monotonic()-started))
if __name__=='__main__':
 try:run(int(sys.argv[1]))
 except BaseException as e:atomic(R/f'ridge_{sys.argv[1]}_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
