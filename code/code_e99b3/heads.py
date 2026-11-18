from common import *
import numpy as np,pickle,collections
import torch
from linear import LogisticRegression,StandardScaler,balanced_accuracy_score
PIDS=[p['id'] for p in read('spec.json')['predicates']];PI={p:i for i,p in enumerate(PIDS)}
torch.set_num_threads(2)
class MLP(torch.nn.Module):
 def __init__(self,d):super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(d,128),torch.nn.SiLU(),torch.nn.Linear(128,21))
 def forward(self,x):return self.net(x)
def load_frame(pool,all_frames=False):
 sources=[s for s in read('sources.json') if s['pool']==pool];keys=[];xs={k:[] for k in ('b','c','d')}
 for s in sources:
  o=R/'features'/s['parent'];fs=np.load(o/'target_rows.npy');ix=np.arange(len(fs)) if all_frames else np.searchsorted(fs,s['annotation_frames']);keys.extend((s['parent'],int(f)) for f in fs[ix]);b=np.load(o/'b.npy',mmap_mode='r')[ix].astype('f4');c=np.load(o/'c.npy',mmap_mode='r')[ix].astype('f4');pr=np.load(o/'proprio.npy',mmap_mode='r')[ix];xs['b'].append(b);xs['c'].append(c);xs['d'].append(np.concatenate([c,pr],1))
 return keys,{k:np.concatenate(v) for k,v in xs.items()}
def targets(keys):
 ix={k:i for i,k in enumerate(keys)};y=np.full((len(keys),21),np.nan,'f4');u=y.copy()
 for r in read('train_labels.json'):
  k=(r['parent'],r['frame']);j=PI[r['predicate']];i=ix[k]
  if r['answer'] in ('yes','no'):y[i,j]=r['answer']=='yes';u[i,j]=0
  elif r['answer']=='unknown':u[i,j]=1
 return y,u

def logistic_fit(x,y,C):
 models=[]
 for j in range(21):
  good=np.isfinite(y[:,j]);v=y[good,j]
  if len(np.unique(v))<2:models.append(float(v[0]) if len(v) else .5)
  else:models.append(LogisticRegression(C=C,class_weight='balanced',max_iter=250,solver='liblinear',random_state=9903).fit(x[good],v))
 return models
def logistic_predict(models,x):
 return np.stack([m.predict_proba(x)[:,1] if hasattr(m,'predict_proba') else np.full(len(x),m) for m in models],1).astype('f4')
def mlp_fit(x,y,epochs):
 torch.manual_seed(9903);m=MLP(x.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);xt=torch.from_numpy(x.astype('f4'));good=torch.from_numpy(np.isfinite(y));yt=torch.from_numpy(np.nan_to_num(y));pos=(yt*good).sum(0);neg=((1-yt)*good).sum(0);pw=(neg/pos.clamp_min(1)).clamp(.1,20);lossfn=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=pw)
 g=torch.Generator().manual_seed(9903);snaps={}
 for epoch in range(1,epochs+1):
  m.train()
  for batch in torch.randperm(len(x),generator=g).split(512):
   loss=(lossfn(m(xt[batch]),yt[batch])*good[batch]).sum()/good[batch].sum().clamp_min(1);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  if epoch in (10,30):snaps[epoch]={k:v.detach().clone() for k,v in m.state_dict().items()}
 return m,snaps
def mlp_predict(m,x):
 m.eval()
 with torch.inference_mode():return torch.cat([m(torch.from_numpy(z.astype('f4'))).sigmoid() for z in np.array_split(x,max(1,int(np.ceil(len(x)/512))))]).numpy()
def ba(y,p,thresholds=.5):
 vals=[]
 for j in range(21):
  good=np.isfinite(y[:,j]);v=y[good,j]
  if len(np.unique(v))==2:vals.append(balanced_accuracy_score(v,p[good,j]>=(thresholds if np.isscalar(thresholds) else thresholds[j])))
 return float(np.mean(vals))
def thresholds(y,p):
 out=[]
 for j in range(21):
  good=np.isfinite(y[:,j]);v=y[good,j]
  if len(np.unique(v))<2:out.append(.5);continue
  candidates=[.5,.35,.65,.2,.8];scores=[balanced_accuracy_score(v,p[good,j]>=t) for t in candidates];out.append(candidates[int(np.argmax(scores))])
 return out

def fit():
 keys,xs=load_frame('train');y,u=targets(keys);val=set(read('split.json')['validation']);iv=np.array([p in val for p,f in keys]);audit={};art={};oof={}
 for method,x in xs.items():
  scaler=StandardScaler().fit(x[~iv]);xf=scaler.transform(x).astype('f4');candidates=[]
  if method=='b':
   for C in (.01,.1,1.):
    h=logistic_fit(xf[~iv],y[~iv],C);p=logistic_predict(h,xf[iv]);candidates.append((ba(y[iv],p),C,h,p))
  else:
   m,snaps=mlp_fit(xf[~iv],y[~iv],30)
   for epoch,state in snaps.items():
    m.load_state_dict(state);p=mlp_predict(m,xf[iv]);candidates.append((ba(y[iv],p),epoch,state,p))
  best=max(candidates,key=lambda z:z[0]);th=thresholds(y[iv],best[3]);unknown=logistic_fit(xf[~iv],u[~iv],.1);up=logistic_predict(unknown,xf[iv]);up[:,PI['P073']]=1
  oof[method]=dict(keys=[k for k,yes in zip(keys,iv) if yes],prob=best[3],unknown=up)
  audit[method]=dict(candidates=[dict(value=c[1],macro_balanced_accuracy=c[0]) for c in candidates],selected=best[1],validation_BA_at_selected_thresholds=ba(y[iv],best[3],th),thresholds=th)
  # Produce validation controller scores with the exact 48-parent fit, before all60 refit.
  if method=='d':
   m=MLP(x.shape[1]);m.load_state_dict(best[2]);vscores={}
   for s in read('sources.json'):
    if not s['validation']:continue
    z=R/'features'/s['parent'];xx=np.concatenate([np.load(z/'c.npy').astype('f4'),np.load(z/'proprio.npy')],1);xx=scaler.transform(xx).astype('f4');pp=mlp_predict(m,xx);uu=logistic_predict(unknown,xx);uu[:,PI['P073']]=1;np.savez_compressed(R/(s['parent']+'_val_scores.npz'),prob=pp,unknown=uu)
  fullsc=StandardScaler().fit(x);xf=fullsc.transform(x).astype('f4')
  if method=='b':head=logistic_fit(xf,y,best[1])
  else:head,_=mlp_fit(xf,y,best[1]);head=head.state_dict()
  art[method]=dict(scaler=fullsc,head=head,thresholds=np.array(th),unknown=logistic_fit(xf,u,.1),kind='logistic' if method=='b' else 'mlp',dim=x.shape[1],selected=best[1],untrained=[j for j in range(21) if not np.isfinite(y[:,j]).any()])
  status('fit_heads',method=method,selected=best[1],validation_ba=best[0])
 with (R/'heads.pkl').open('xb') as f:pickle.dump(art,f)
 put('internal_validation.json',audit);put('HEADS_LOCK.json',dict(brt=now(),heads_sha256=sha(R/'heads.pkl'),validation_sha256=sha(R/'internal_validation.json'),protocol_sha256=sha(R/'PROTOCOL_LOCK.json')))
 report('## 内部验证与头锁定 '+now()+'\n'+json.dumps(audit,ensure_ascii=False)+'\n训练选择完成；之后不按 dev 修改。二分类与弃权分开训练；二分类主指标在全部参考 yes/no 上计算，另报选择性覆盖。P073 固定弃权。')

def predict_sources(pool):
 with (R/'heads.pkl').open('rb') as f:art=pickle.load(f)
 for s in read('sources.json'):
  if s['pool']!=pool:continue
  z=R/'features'/s['parent'];b=np.load(z/'b.npy',mmap_mode='r');c=np.load(z/'c.npy',mmap_mode='r');pr=np.load(z/'proprio.npy',mmap_mode='r');outs={}
  for method in ('b','c','d'):
   a=art[method];x=(b if method=='b' else c if method=='c' else np.concatenate([c,pr],1)).astype('f4');x=a['scaler'].transform(x).astype('f4')
   if method=='b':p=logistic_predict(a['head'],x)
   else:m=MLP(a['dim']);m.load_state_dict(a['head']);p=mlp_predict(m,x)
   u=logistic_predict(a['unknown'],x);u[:,PI['P073']]=1
   outs[method+'_prob']=p;outs[method+'_unknown']=u
  np.savez_compressed(R/(s['parent']+'_scores.npz'),frames=np.load(z/'target_rows.npy'),**outs);status('head_predictions',pool=pool,parent=s['parent'])
if __name__=='__main__':fit() if sys.argv[1]=='fit' else predict_sources(sys.argv[1])
