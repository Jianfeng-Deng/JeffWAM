import os,sys,json,time,collections,traceback,hashlib,datetime,importlib.util
from pathlib import Path
import numpy as np
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e97a_robot_env_20261114_01';E93=T/'results/e93_dev_diag_20261114_01'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from e90.data import load_data,Data
H=(10,25,50,100);ALPHAS=(.0001,.001,.01,.1,1.,10.)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def save(name,x):
 p=R/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def status(stage,**kw):save('status.json',dict(stage=stage,brt=now(),**kw));print(stage,kw,flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def datasets():
 p,ready,tr,old,_=load_data();caps=json.loads((E93/'dev/captures_full.json').read_text());rows=json.loads((E93/'dev/windows_full.json').read_text())
 dev=Data(rows,caps,E93/'features',tr.base.stats,{'scale':tr.norm},{'__human_default__':old.base.text_cache['__human_default__']})
 assert len(dev.rows)==211
 assert not {r['parent_component'] for r in tr.rows}&{r['parent_component'] for r in dev.rows}
 assert all(r['canonical_id']!='piper_4dc107335389f5711a0f65e0' for d in (tr,dev) for r in d.rows)
 return p,tr,dev
P=Path('/mnt/workspace/dengjianfeng/demo4_zip_tie_active_gaze_20261030')
spec=importlib.util.spec_from_file_location('piper_fk_readonly',P/'bench/piper_bench/fk.py');fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk)
CHAIN=fk.load_chain() # Explicit chain avoids loading fitted camera params; FK only.
class States:
 def __init__(self):self.memo={}
 def get(self,data,cid):
  if cid not in self.memo:
   cap=data.index.captures[cid];z=dict(np.load(cap['numeric_cache'],allow_pickle=False));supp=R/'dev_states'/(cid+'.npz')
   if cap.get('split')=='dev' and supp.exists():z.update(dict(np.load(supp,allow_pickle=False)))
   needed=('q','qpos_raw','timestamp','state_source_brackets','state_available_time','continuous_segment','p1_source_mask')
   if not all(k in z for k in needed):self.memo[cid]=(z,None,None,'missing:'+','.join(k for k in needed if k not in z));return self.memo[cid]
   q=z['q'].astype('f8').copy();q[:,[6,13]]=z['qpos_raw'][:,[6,13]]
   good=np.isfinite(q).all(1)&np.isfinite(z['state_available_time']);br=z['state_source_brackets']
   good &= np.isfinite(br).all((1,2))&((br[:,:,1]-br[:,:,0]).max(1)<=.0750001)&(br[:,:,1].max(1)<=z['state_available_time']+1e-6)
   poses=[]
   for sl in (slice(0,6),slice(7,13)):
    m=fk.fk_link6(q[:,sl],CHAIN);xyz=m[:,:3,3]+m[:,:3,:3]@np.array([0,0,.1358]);rot=np.concatenate([m[:,:3,0],m[:,:3,1]],1);poses.append(np.concatenate([xyz,rot],1))
   pose=np.concatenate(poses,1);good &= np.isfinite(pose).all(1)
   self.memo[cid]=(z,q,pose,good)
  return self.memo[cid]
 def features(self,data,row,h):
  z,q,pose,good=self.get(data,row['canonical_id']);a=row['anchor'];f=a+h
  if q is None:return None,good
  if f>=len(q):return None,'end'
  if not good[a] or not good[f]:return None,'invalid_endpoint_state'
  if not np.all(z['continuous_segment'][a:f+1]==z['continuous_segment'][a]):return None,'state_seam'
  if not z['p1_source_mask'][a:f+1].all():return None,'state_source_mask'
  dt=np.diff(z['timestamp'][a:f+1])
  if not ((dt>0)&(dt<=.050001)).all():return None,'timestamp_gap'
  past=[];flags=[]
  for off,valid in zip((-24,-18,-12,-6),row['history_valid']):
   j=a+off;ok=bool(valid and j>=0 and good[j] and z['continuous_segment'][j]==z['continuous_segment'][a] and z['state_available_time'][j]<=z['state_available_time'][a]+1e-7)
   if valid and not ok:return None,'invalid_required_history_state'
   past.append(q[j]-q[a] if ok else np.zeros(14));flags.append(float(ok))
  return np.concatenate([q[a],*past,q[f]-q[a],pose[a],pose[f]-pose[a],flags]).astype('f4'),None

def x0(obs,hv,h,z):
 nt=8 if h<=50 else 1;n=3*nt
 zz=obs if h<=50 else obs.mean(2,keepdims=True);cur=zz[4].reshape(n,768);hist=zz[:4].reshape(4,n,768);delta=hist-cur[None];delta[~hv]=0
 return np.concatenate([(cur-z['pca_mean'])@z['pca'],*[delta[j]@z['pca'] for j in range(4)],np.broadcast_to(hv.astype('f4'),(n,4)),np.eye(n,dtype='f4')],1)
def stats(X,floor):return X.mean(0),np.maximum(X.std(0),floor)
def sufficient(X,Z,Y):
 n,k,d=X.shape;r=Z.shape[-1];A=X.reshape(-1,d).astype('f8');B=Y.reshape(-1,768).astype('f8');Z=Z.astype('f8');ym=B.mean(0);B=B-ym;B=B.reshape(n,k,768)
 return dict(G=A.T@A/(n*k),a=A.T@B.reshape(-1,768)/(n*k),D=Z.T@Z/n,C=np.stack([X[:,j].T@Z/(n*np.sqrt(k)) for j in range(k)]),b=np.stack([Z.T@B[:,j]/(n*np.sqrt(k)) for j in range(k)]),ym=ym,k=k)
def fit_suff(s,alpha,method):
 if method=='R0':return dict(w0=np.linalg.solve(s['G']+alpha*np.eye(len(s['G'])),s['a']),ym=s['ym'])
 D=s['D']+alpha*np.eye(len(s['D']));DC=np.linalg.solve(D,s['C'].transpose(2,0,1).reshape(len(D),-1)).reshape(len(D),s['k'],-1).transpose(1,0,2);Db=np.linalg.solve(D,s['b'].transpose(1,0,2).reshape(len(D),-1)).reshape(len(D),s['k'],768).transpose(1,0,2)
 A=s['G']+alpha*np.eye(len(s['G']));rhs=s['a'].copy()
 for j in range(s['k']):A-=s['C'][j]@DC[j];rhs-=s['C'][j]@Db[j]
 W=np.linalg.solve(A,rhs);V=np.stack([Db[j]-DC[j]@W for j in range(s['k'])])
 return dict(w0=W,w1=V,ym=s['ym'])
def fit_r1(Z,Y,alpha):
 n,k,_=Y.shape;ym=Y.mean(0);V=np.linalg.solve(Z.T@Z/n+alpha*np.eye(Z.shape[-1]),Z.T@(Y-ym).reshape(n,-1)/n).reshape(Z.shape[-1],k,768).transpose(1,0,2)
 return dict(w1=V,ym=ym)
def predict(m,X,Z,method):
 if method=='R1':return np.einsum('nr,krd->nkd',Z,m['w1'],optimize=True)+m['ym']
 pred=X@m['w0']+m['ym']
 if method=='R2':pred+=np.sqrt(X.shape[1])*np.einsum('nr,krd->nkd',Z,m['w1'],optimize=True)
 return pred

def parent_mean(v,parents):return float(np.mean([np.mean(v[np.array(parents)==p]) for p in sorted(set(parents))]))
def fit_main():
 start=time.monotonic();assert not (R/'FIT_LOCK.json').exists();p,tr,dev=datasets();st=States();by=collections.defaultdict(list)
 for i,r in enumerate(tr.rows):by[r['parent_component']].append(i)
 parents=sorted(by);val=set(np.random.default_rng(9305).choice(parents,len(parents)//5,replace=False).tolist());ids=[]
 for parent,ii in sorted(by.items()):ids.extend(ii[j] for j in np.linspace(0,len(ii)-1,min(12,len(ii))).astype(int))
 selected=set(ids);coverage=collections.defaultdict(collections.Counter);samplemeta=[];pool={h:dict(X=[],Z=[],Y=[],parents=[],indices=[]) for h in H};frozen={h:dict(np.load(E93/f'ridge_h{h}_coefficients.npz')) for h in H}
 # State audit for every original P1 window, using authoritative cached horizon endpoint table.
 table=np.load(T/'code_e88/artifacts/cpu_audit_01/P1_horizon_rows.npz')['rows'];assert len(table)==len(tr.rows)
 for cid,ii in __import__('itertools').groupby(sorted(range(len(tr.rows)),key=lambda i:tr.rows[i]['canonical_id']),key=lambda i:tr.rows[i]['canonical_id']):
  ii=list(ii);cap=tr.index.captures[cid];source=cap['source_collection'];z,q,pose,g=st.get(tr,cid)
  for i in ii:
   row=tr.rows[i];coverage[source]['P1_windows']+=1
   if q is not None and g[row['anchor']]:coverage[source]['current_state_windows']+=1
   for j,h in enumerate(H):
    if table[i,j,0]<0:continue
    coverage[source][f'image_h{h}']+=1;feat,why=st.features(tr,row,h)
    coverage[source][f'paired_h{h}' if feat is not None else f'excluded_h{h}:{why}']+=1
   if i not in selected:continue
   b=tr.get(i,H,'short','P1');obs=b['obs'].numpy();hv=b['history_valid'].numpy()
   for h in H:
    off=H.index(h)*24 if h<=50 else 72;n=24 if h<=50 else 3;sl=slice(off,off+n)
    if not b['world_valid'][sl].all():continue
    feat,why=st.features(tr,row,h)
    if feat is None:continue
    d=pool[h];d['X'].append(x0(obs,hv,h,frozen[h]));d['Z'].append(feat);d['Y'].append(b['world'][sl].numpy());d['parents'].append(row['parent_component']);d['indices'].append(i)
  if len(samplemeta)%40==0:status('loading_training',captures=len(samplemeta),selected_loaded=sum(len(x['indices']) for x in pool.values()))
  samplemeta.append(dict(canonical_id=cid,parent=cap['parent_component'],source=source,windows=len(ii),state_present=q is not None,numeric=cap['numeric_cache']))
 save('state_coverage_train.json',dict(sources={k:dict(v) for k,v in coverage.items()},captures=samplemeta,all_P1_windows=len(tr.rows),parents=len(parents)))
 save('selection.json',dict(seed=9305,validation_parents=sorted(val),selected_train_indices=ids,selected_windows=len(ids),dev='E93 fixed 27 parents / 211 windows; h100 26/208',robot_features='q_current14 (signed raw grips), four masked past q deltas14, future q delta14, current own-base FK xyz+rot6d18, future FK xyz+rot6d delta18, history flags4; total124; no action array'))
 report=T/'notes/e97a_robot_env_decomp_20261114.md'
 with report.open('a') as f:f.write('\n### 状态全池核查已完成\n'+json.dumps({k:dict(v) for k,v in coverage.items()},ensure_ascii=False)+'\nFK 仅调用已有 fk_link6，显式加载 URDF chain，不调用相机投影；末端几何 offset=0.1358m。\n')
 methods=[]
 for h in H:
  d=pool.pop(h);X=np.stack(d.pop('X'));Z=np.stack(d.pop('Z'));Y=np.stack(d.pop('Y'));ps=d['parents'];mask=np.array([p not in val for p in ps]);xm,xs=stats(X[mask].reshape(-1,X.shape[-1]),.05);zm,zs=stats(Z[mask],1e-6);X=(X-xm)/xs;Z=(Z-zm)/zs
  s=sufficient(X[mask],Z[mask],Y[mask]);scores={};models={}
  for method in ('R0','R1','R2'):
   ss=[]
   for alpha in ALPHAS:
    m=fit_r1(Z[mask],Y[mask],alpha) if method=='R1' else fit_suff(s,alpha,method)
    pr=predict(m,X[~mask],Z[~mask],method);err=((pr-Y[~mask])**2).mean((1,2));score=parent_mean(err,[p for p,mk in zip(ps,mask) if not mk]);ss.append([score,alpha])
   alpha=min(ss)[1];scores[method]=dict(alpha=alpha,scores=ss);status('alpha_selected',h=h,method=method,alpha=alpha)
  # Same standardization remains frozen from inner fit, as in E93; full242 parent refit.
  s=sufficient(X,Z,Y)
  for method in ('R0','R1','R2'):
   a=scores[method]['alpha'];m=fit_r1(Z,Y,a) if method=='R1' else fit_suff(s,a,method)
   m={k:v.astype('f4') for k,v in m.items()};np.savez_compressed(R/f'{method}_h{h}.npz',**m,xm=xm,xs=xs,zm=zm,zs=zs)
   models[method]=m
  old=frozen[h];pred=predict(models['R0'],X,Z,'R0');ref=((np.stack([x0(tr.get(i,(h,),'short','P1')['obs'].numpy(),np.array(tr.rows[i]['history_valid']),h,old) for i in d['indices']])-old['x_mean'])/old['x_scale'])@old['coef']+old['intercept']
  audit=float(np.max(abs(pred-ref)));assert audit<3e-4,('R0 differs E93',h,audit)
  methods.append(dict(h=h,windows=len(Y),parents=len(set(ps)),inner_fit_parents=len({p for p,m in zip(ps,mask) if m}),inner_validation_parents=len({p for p,m in zip(ps,mask) if not m}),alphas=scores,R0_frozen_max_abs_prediction_diff=audit))
  save('fit_method.json',methods);status('horizon_fitted',h=h,seconds=time.monotonic()-start)
 save('FIT_LOCK.json',dict(status='coefficients_locked_before_dev',brt=now(),methods=methods,hashes={p.name:sha(p) for p in R.glob('R[012]_h*.npz')},dev_evaluations=0,CPU_seconds=time.monotonic()-start,GPUh=0))
 with report.open('a') as f:f.write('\n### 拟合协议已锁定（dev 评分前）\n'+json.dumps(methods,ensure_ascii=False)+'\nR0 为 E93 原共享线性映射；R1 允许每相机/格点各自的状态映射；R2 用共享 R0 加逐格状态项（sqrt(token数)缩放，使逐格状态项的岭惩罚不因网格数改变）。所有组使用同一内层 parent，alpha 网格沿 E93。当前状态含显式等待至 state_available_time；未来实测 q 是诊断性 oracle 条件，非当前可部署输入。\n')
 status('fit_completed',seconds=time.monotonic()-start)
if __name__=='__main__':
 try:fit_main()
 except BaseException:save('failure.json',dict(brt=now(),traceback=traceback.format_exc(),GPUh=0));raise
