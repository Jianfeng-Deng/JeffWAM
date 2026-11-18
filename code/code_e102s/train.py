from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,pickle,torch,traceback
from linear import StandardScaler,LogisticRegression,balanced_accuracy_score
from scipy.stats import rankdata
OLD=T/'results/e99b3_20261115_01'
def oldread(n):return json.loads((OLD/n).read_text())
sources=oldread('sources.json');train=[s for s in sources if s['pool']=='train'];dev=[s for s in sources if s['pool']=='dev'];PIDS=[p['id'] for p in oldread('spec.json')['predicates']];PI={p:i for i,p in enumerate(PIDS)}
parents=sorted(s['parent'] for s in train);perm=np.random.default_rng(10101).permutation(parents);fold={p:i%5 for i,p in enumerate(perm)}
torch.set_num_threads(2)
keys=[(s['parent'],int(f)) for s in train for f in s['annotation_frames']];pidx=np.array([fold[p] for p,f in keys]);keyix={k:i for i,k in enumerate(keys)}
y=np.full((len(keys),21),np.nan,'f4');u=y.copy()
for r in oldread('train_labels.json'):
 i=keyix[r['parent'],r['frame']];j=PI[r['predicate']]
 if r['answer'] in ['yes','no']:y[i,j]=r['answer']=='yes';u[i,j]=0
 elif r['answer']=='unknown':u[i,j]=1

def load(s):
 d=R/'features'/s['parent'];return np.load(d/'b.npy',mmap_mode='r'),np.load(d/'state.npy',mmap_mode='r'),np.load(d/'times.npy')
def indices(frames,times,offs):
 fr=np.asarray(frames);ix=fr[:,None]+np.array(offs)[None];valid=ix>=0;ix=ix.clip(0);gaps=np.r_[0,np.cumsum((np.diff(times)<=0)|(np.diff(times)>.050001))];valid&=gaps[ix]==gaps[fr,None];return ix,valid

def make(s,frames,method,length=60,gripper_fold=None):
 b,pr,t=load(s);fr=np.asarray(frames)
 if method=='H0':return np.asarray(b[fr],dtype='f4')
 if method=='H1':
  ix,v=indices(fr,t,[0,-6,-12,-24,-48]);z=np.asarray(b[ix],dtype='f4');z[~v]=0;return np.concatenate([z.reshape(len(fr),-1),(z[:,0,None]-z[:,1:]).reshape(len(fr),-1),v],1).astype('f4')
 ix,v=indices(fr,t,list(range(-length,1,3)));z=np.asarray(b[ix],dtype='f4');z[~v]=0
 if method=='H3':
  q=np.array(pr[ix]);
  if gripper_fold is not None:q[:,:,[6,13]]=np.load(R/'features'/s['parent']/'gripper_fold.npy',mmap_mode='r')[gripper_fold][ix]
  q[~v]=0;z=np.concatenate([z,q],2)
 return np.concatenate([z,v[:,:,None]],2).astype('f4')

def logistic_fit(x,y,C):
 out=[]
 for j in range(21):
  mask=np.isfinite(y[:,j]);v=y[mask,j]
  if len(np.unique(v))<2:out.append(float(v[0]) if len(v) else .5)
  else:out.append(LogisticRegression(C=C,max_iter=250).fit(x[mask],v))
 return out

def logistic_predict(h,x):return np.stack([m.predict_proba(x)[:,1] if hasattr(m,'predict_proba') else np.full(len(x),m) for m in h],1).astype('f4')

class Temporal(torch.nn.Module):
 def __init__(self,d):
  super().__init__();self.proj=torch.nn.Linear(d,64);self.c1=torch.nn.Conv1d(64,64,5,dilation=1);self.c2=torch.nn.Conv1d(64,64,5,dilation=4);self.out=torch.nn.Linear(64,42)
 def forward(self,x):
  z=torch.nn.functional.silu(self.proj(x)).transpose(1,2);z=torch.nn.functional.silu(self.c1(torch.nn.functional.pad(z,(4,0))));z=torch.nn.functional.silu(self.c2(torch.nn.functional.pad(z,(16,0))));return self.out(z[:,:,-1])

def fit(x,mask,method):
 if method in ['H0','H1']:
  sc=StandardScaler().fit(x[mask]);xx=sc.transform(x[mask]).astype('f4');return dict(sc=sc,head=logistic_fit(xx,y[mask],.01),unknown=logistic_fit(xx,u[mask],.1),method=method)
 # Fit statistics only on observed training history tokens, never heldout parent.
 xx=x[mask];valid=xx[:,:,-1]>0;sc=StandardScaler().fit(xx[valid]);sc.mean_[-1]=0;sc.scale_[-1]=1;xt=sc.transform(xx).astype('f4');xt[~valid]=0
 torch.manual_seed(102);m=Temporal(x.shape[-1]);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);tar=np.concatenate([y[mask],u[mask]],1);good=torch.from_numpy(np.isfinite(tar));yt=torch.from_numpy(np.nan_to_num(tar));pos=(yt*good).sum(0);neg=((1-yt)*good).sum(0);pw=(neg/pos.clamp_min(1)).clamp(.1,20);lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=pw,reduction='none');xt=torch.from_numpy(xt);gen=torch.Generator().manual_seed(102)
 for ep in range(10):
  for ix in torch.randperm(len(xt),generator=gen).split(64):
   z=m(xt[ix]);loss=(lossfn(z,yt[ix])*good[ix]).sum()/good[ix].sum().clamp_min(1);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 return dict(sc=sc,head=m.state_dict(),dim=x.shape[-1],method=method)

def predict(a,x):
 if a['method'] in ['H0','H1']:
  xx=a['sc'].transform(x).astype('f4');p=logistic_predict(a['head'],xx);u=logistic_predict(a['unknown'],xx)
 else:
  v=x[:,:,-1]>0;xx=a['sc'].transform(x).astype('f4');xx[~v]=0;m=Temporal(a['dim']);m.load_state_dict(a['head']);m.eval()
  with torch.inference_mode():z=torch.cat([m(torch.from_numpy(q)).sigmoid() for q in np.array_split(xx,max(1,int(np.ceil(len(xx)/128))))]).numpy()
  p,u=z[:,:21],z[:,21:]
 u[:,PI['P073']]=1;return p,u

def ba(y,p):
 return float(np.mean([balanced_accuracy_score(y[np.isfinite(y[:,j]),j],p[np.isfinite(y[:,j]),j]>=.5) for j in range(21) if len(np.unique(y[np.isfinite(y[:,j]),j]))==2]))
def auc(y,p):
 out=[]
 for j in range(21):
  v=np.isfinite(y[:,j]);yy=y[v,j];pp=p[v,j];n=yy.sum();nn=len(yy)-n
  if n*nn:out.append((rankdata(pp)[yy==1].sum()-n*(n+1)/2)/(n*nn))
 return float(np.mean(out))

def thresholds(y,p):
 out=[]
 for j in range(21):
  mask=np.isfinite(y[:,j]);yy=y[mask,j]
  if len(np.unique(yy))<2:out.append(.5);continue
  cc=[.5,.35,.65,.2,.8];out.append(max(cc,key=lambda t:balanced_accuracy_score(yy,p[mask,j]>=t)))
 return out

def main():
 assert (R/'FEATURES_COMPLETE.json').exists() and not (R/'HEADS_LOCK.json').exists();put('folds.json',dict(seed=10101,parent_to_fold=fold,full_trajectory_parents=parents));locked={};candidates=[]
 for method in ['H0','H1','H2','H3']:
  options=[60] if method in ['H0','H1'] else [30,60];best=None
  for length in options:
   name=method+'_'+str(length);saved=R/(name+'_candidate.pkl')
   if saved.exists():
    with saved.open('rb') as f:cand=pickle.load(f)
   else:
    x=np.concatenate([make(s,s['annotation_frames'],method,length) for s in train]);pp=np.empty_like(y);uu=np.empty_like(y);arts=[]
    for k in range(5):
     if method=='H3':
      offset=0
      for ss in train:
       fr=ss['annotation_frames'];tt=np.load(R/'features'/ss['parent']/'times.npy');ii,vv=indices(fr,tt,list(range(-length,1,3)));gg=np.array(np.load(R/'features'/ss['parent']/'gripper_fold.npy',mmap_mode='r')[k][ii]);gg[~vv]=0;x[offset:offset+len(fr),:,2304+np.array([6,13])]=gg;offset+=len(fr)
     mask=pidx!=k;a=fit(x,mask,method);p,up=predict(a,x[~mask]);pp[~mask]=p;uu[~mask]=up;arts.append(a);status('crossfit',method=method,history=length,fold=k)
    cand=dict(method=method,length=length,prob=pp,unknown=uu,models=arts,auc=auc(y,pp),ba=ba(y,pp))
    with saved.open('xb') as f:pickle.dump(cand,f)
    del x
   candidates.append({k:cand[k] for k in ['method','length','auc','ba']})
   if best is None or (cand['auc'],cand['ba'])>(best['auc'],best['ba']):best=cand
  length=best['length'];locked[method]=dict(history_frames=length,stride=3 if method in ['H2','H3'] else None,thresholds=(oldread('internal_validation.json')['b']['thresholds'] if method=='H0' else [.5]*21),oof_auc=best['auc'],oof_ba=best['ba'])
  np.savez_compressed(R/(method+'_annotation_oof.npz'),prob=best['prob'],unknown=best['unknown'],y=y,u=u,parent=np.array([p for p,f in keys]),frame=np.array([f for p,f in keys]))
  for s in train:
   path=R/(s['parent']+'_'+method+'_oof.npz')
   if path.exists():continue
   ps=[];us=[];a=best['models'][fold[s['parent']]]
   for fr in np.array_split(np.arange(s['n']),max(1,int(np.ceil(s['n']/256)))):
    p,up=predict(a,make(s,fr,method,length,fold[s['parent']] if method=='H3' else None));ps.append(p);us.append(up)
   np.savez_compressed(path,prob=np.concatenate(ps),unknown=np.concatenate(us));status('full_oof',method=method,parent=s['parent'])
  full=R/(method+'_full.pkl')
  if not full.exists():
   x=np.concatenate([make(s,s['annotation_frames'],method,length) for s in train]);a=fit(x,np.ones(len(keys),bool),method)
   with full.open('xb') as f:pickle.dump(a,f)
   del x
  if method=='H0':
   audits=[]
   for ss in train:
    oldpath=T/'results/e101s_20261115_01'/(ss['parent']+'_b_oof.npz')
    if not oldpath.exists():continue
    aa=np.load(R/(ss['parent']+'_H0_oof.npz'));bb=np.load(oldpath);err=float(abs(aa['prob']-bb['prob']).max());ue=float(abs(aa['unknown']-bb['unknown']).max());assert err<1e-6 and ue<1e-6,(ss['parent'],err,ue);audits.append(dict(parent=ss['parent'],prob_max_abs=err,unknown_max_abs=ue))
   put('H0_E101_OOF_REPRODUCTION.json',audits)
  put('head_training_progress.json',locked)
 put('HEADS_LOCK.json',dict(brt=now(),methods=locked,candidates=candidates,objective='train 60-parent five-fold macro AUROC then BA; H0 thresholds inherited unchanged from E99; H1/H2/H3 binary and unknown thresholds fixed .5 a priori',unknown='separate binary unknown head; primary yes/no never removed on abstention; P073 fixed abstain',architecture='H2/H3 shared Linear(input,64), two causal Conv1d kernel5 dilation1/4, 42 outputs, 10 epochs AdamW lr.001; no stage input',causal_history='zero invalid pre-start or timestamp-gap tokens; validity bit; H1 current,-6,-12,-24,-48 and current-minus-history differences',training_label_basis='existing retrospective model labels',dev_evaluations=0,hashes={str(p):sha(p) for p in (T/'code_e102s').glob('*.py')}))
 report('## 谓词头训练锁定 '+now()+'\n'+json.dumps(locked,ensure_ascii=False)+'\n训练60 parent五折；H2/H3历史30/60帧、stride3（11/21时点）仅按训练袋外AUROC、次级BA选择；固定10 epoch/64隐藏维两层因果卷积。unknown独立输出，P073固定弃权；新头二分类与unknown阈值预设0.5；H0原阈值原样继承，不新用模型标签选阈值。主二分类表保留所有参考yes/no。')
 status('HEADS_LOCKED')
if __name__=='__main__':
 try:main()
 except Exception:put('TRAIN_FAILURE_'+str(int(time.time()))+'.json',dict(brt=now(),traceback=traceback.format_exc()));raise
