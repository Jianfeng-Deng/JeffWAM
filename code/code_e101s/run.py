import os,sys
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');OLD=T/'results/e99b3_20261115_01';R=T/'results/e101s_20261115_01';REPORT=T/'notes/e101s_stage_controller_20261115.md'
sys.path.insert(0,str(T/'code_e99b3'))
import heads,controller,numpy as np,json,pickle,time,hashlib,traceback,datetime,itertools
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from linear import StandardScaler

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def write(name,x):
 p=R/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,p)
def read(name):return json.loads((OLD/name).read_text())
def status(s,**kw):write('status.json',dict(stage=s,brt=now(),**kw));print(s,kw,flush=True)
def report(s):
 with REPORT.open('a') as f:f.write('\n'+s+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
S=[f'S{i:02}' for i in range(1,11)];SI={s:i for i,s in enumerate(S)};SPEC=read('spec.json');PIDS=heads.PIDS;PI=heads.PI
sources=read('sources.json');train=[s for s in sources if s['pool']=='train'];full=[s for s in train if s['validation']];dev=[s for s in sources if s['pool']=='dev'];stages={pool:read('stages_'+pool+'.json') for pool in ['train','dev']}
assert len(train)==60 and len(dev)==20 and len(full)==12
assert not {s['parent'] for s in train}&{s['parent'] for s in dev}
assert all(s['cid']!='piper_4dc107335389f5711a0f65e0' for s in sources)
def truth(s):
 y=np.full(s['n'],-1,int)
 for z in stages[s['pool']]:
  if z['parent_id']==s['parent']:y[z['stage_start']:z['stage_end_exclusive']]=SI.get(z['stage_id'],-1)
 return y
truths={s['parent']:truth(s) for s in sources}
parents=sorted(s['parent'] for s in train);perm=np.random.default_rng(10101).permutation(parents);fold={p:i%5 for i,p in enumerate(perm)}
A=np.eye(10,dtype=bool)
for t in SPEC['transitions']:
 if t['source'] in SI and t['target'] in SI:A[SI[t['source']],SI[t['target']]]=1

def raw(s,method,annotations=False):
 z=OLD/'features'/s['parent'];frames=np.load(z/'target_rows.npy');ix=np.searchsorted(frames,s['annotation_frames']) if annotations else np.arange(len(frames));assert np.array_equal(frames[ix],s['annotation_frames']) if annotations else True
 x=np.load(z/('b.npy' if method=='b' else 'c.npy'),mmap_mode='r')[ix].astype('f4')
 if method=='d':x=np.concatenate([x,np.load(z/'proprio.npy',mmap_mode='r')[ix]],1)
 return frames[ix],x

def crossfit():
 assert not (R/'OOF_COMPLETE.json').exists()
 keys,xs=heads.load_frame('train');y,u=heads.targets(keys);pf=np.array([fold[p] for p,f in keys]);audit=[]
 write('folds.json',dict(seed=10101,parent_to_fold=fold,complete_trajectory_parents=[s['parent'] for s in full]))
 for method in ['b','d']:
  x=xs[method];oof=np.empty_like(y);ou=np.empty_like(y)
  for k in range(5):
   tr=pf!=k;te=~tr;sc=StandardScaler().fit(x[tr]);xt=sc.transform(x[tr]).astype('f4');xe=sc.transform(x[te]).astype('f4')
   h=heads.logistic_fit(xt,y[tr],.01) if method=='b' else heads.mlp_fit(xt,y[tr],10)[0]
   uh=heads.logistic_fit(xt,u[tr],.1)
   pred=lambda xx:heads.logistic_predict(h,xx) if method=='b' else heads.mlp_predict(h,xx)
   oof[te]=pred(xe);ou[te]=heads.logistic_predict(uh,xe);ou[te,PI['P073']]=1
   for s in full:
    if fold[s['parent']]!=k:continue
    frames,xx=raw(s,method);xx=sc.transform(xx).astype('f4');pp=pred(xx);uu=heads.logistic_predict(uh,xx);uu[:,PI['P073']]=1
    assert len(frames)==s['n'] and np.array_equal(frames,np.arange(s['n']))
    np.savez_compressed(R/(s['parent']+'_'+method+'_oof.npz'),prob=pp,unknown=uu)
   audit.append(dict(method=method,fold=k,fit_parents=sorted(set(p for (p,f),v in zip(keys,tr) if v)),heldout_parents=sorted(p for p in parents if fold[p]==k),fit_rows=int(tr.sum()),heldout_rows=int(te.sum())))
   status('crossfit',method=method,fold=k)
  np.savez_compressed(R/(method+'_annotation_oof.npz'),prob=oof,unknown=ou,y=y,u=u,parent=np.array([p for p,f in keys]),frame=np.array([f for p,f in keys]))
 write('OOF_COMPLETE.json',dict(brt=now(),audit=audit,score_sources='5-fold held-out parent; C=.01 / d epochs=10 inherited E99 train-only choice; train60 annotations; full12 trajectories',gpu_hours=0))
 report('## 分数与交叉拟合完成 '+now()+'\n60 train parent 按 parent 5折，归一化、b逻辑回归(C=.01)、d原128-SiLU MLP(10 epoch)、unknown头均只拟合另外48 parent。固定继承E99训练选型，不重选谓词头。原位读取特征，GPUh=0。仅原内部验证12 train parent缓存具备全源帧轨迹（其余48仅标注帧），控制器候选只在这12条完整袋外轨迹评分；60条稀疏标注分数用于校准和发射训练。未插值分数，未平移标签。该覆盖限制需要后续独立验证。')

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
 sy=np.array([truths[p][f] for p,f in zip(z['parent'][mask],z['frame'][mask])]);w,prior=stagefit(x,sy);tr,dur,illegal=transition(excluded)
 return dict(cal=c,w=w,prior=prior,transition=tr,durations=dur,illegal=illegal)
def emission(p,aux):
 x=calibrate(p,aux['cal']);x[:,PI['P073']]=.5;z=np.c_[x,np.ones(len(x))]@aux['w'];return z-logsumexp(z,axis=1,keepdims=True)-np.log(aux['prior'])

def smooth_rule(prob,unknown,alpha,hi,lo,m):
 state='S01';out=[];events=set();ps=prob[0].copy();us=unknown[0].copy();yes=np.zeros(21,bool);age=0
 def guard(g,answers):
  if 'all' in g:return all(guard(v,answers) for v in g['all'])
  if 'any' in g:return any(guard(v,answers) for v in g['any'])
  if 'predicate' in g:return answers[PI[g['predicate']]]==g['equals']
  if 'event' in g:return g['event'] in events
  if 'context' in g:return g['context']=='batch_continue_enabled'
  raise ValueError(g)
 for p,u in zip(prob,unknown):
  ps=(1-alpha)*ps+alpha*p;us=(1-alpha)*us+alpha*u;yes=np.where(yes,ps>=lo,ps>=hi);ans=['unknown' if v>=.5 else 'yes' if q else 'no' for q,v in zip(yes,us)];age+=1;cleared=False
  candidates=[t for t in SPEC['transitions'] if t['source']==state and t['target'] in SI and guard(t['guard'],ans)] if age>=m else []
  if candidates:
   priority=min(t['priority'] for t in candidates);top=[t for t in candidates if t['priority']==priority]
   if len({t['target'] for t in top})==1:
    t=top[0];state=t['target'];age=0
    if t['kind'] in ('failure','next_cycle','retry','abandon') or 'increment_attempt_id' in t['effects']:events.clear();cleared=True
  if not cleared and ans[PI['P101']]!='yes' and ans[PI['P071']]=='yes':events.add('P071_confirmed_earlier_this_attempt')
  out.append(SI[state])
 return np.array(out)

def hmm(e,aux,temp,m,viterbi=False):
 # CPU causal forward recursion. Emission receives only <=t frame scores.
 loga=np.full((10,10),-np.inf);mask=aux['transition']>0;loga[mask]=np.log(aux['transition'][mask]);e=e*temp
 if viterbi:
  v=np.full(10,-np.inf);v[0]=0;back=np.zeros((len(e),10),np.int16)
  for i,ee in enumerate(e):
   z=v[:,None]+loga;back[i]=z.argmax(0);v=z.max(0)+ee;v-=v.max()
  path=np.zeros(len(e),int);path[-1]=v.argmax()
  for i in range(len(e)-1,0,-1):path[i-1]=back[i,path[i]]
  return path
 f=np.full(10,-np.inf);f[0]=0;state=0;age=0;out=[]
 for ee in e:
  f=logsumexp(f[:,None]+loga,axis=0)+ee;f-=logsumexp(f);age+=1
  if age>=m:
   nxt=int(np.argmax(np.where(A[state],f,-np.inf)))
   if nxt!=state:state=nxt;age=0
  out.append(state)
 return np.array(out)

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

def tune():
 assert not (R/'CONTROLLER_LOCK.json').exists();aux={};cache={};candidates=[]
 for method in ['b','d']:
  for k in range(5):aux[method,k]=build_aux(method,k)
  for s in full:
   z=np.load(R/(s['parent']+'_'+method+'_oof.npz'));a=aux[method,fold[s['parent']]];cache[method,s['parent']]=(z['prob'],z['unknown'],emission(z['prob'],a),a)
 for kind in ['S1','S2']:
  for method in ['b','d']:
   grid=[dict(alpha=a,hi=h,lo=h-gap,m=m) for a,h,gap,m in itertools.product([.05,.2],[.5,.65],[.15,.3],[15,45])] if kind=='S1' else [dict(temp=t,m=m) for t,m in itertools.product([.1,.3,1.],[1,15,45])]
   for cfg in grid:
    rr=[]
    for s in full:
     p,u,e,a=cache[method,s['parent']];pred=smooth_rule(p,u,**cfg) if kind=='S1' else hmm(e,a,**cfg);rr.append(record(pred,truths[s['parent']]))
    val=aggregate(rr);candidates.append(dict(kind=kind,method=method,config=cfg,metrics=val));status('tune',kind=kind,method=method,config=cfg,f1_30=val['f1_30'])
 best={kind:max([c for c in candidates if c['kind']==kind],key=lambda c:(c['metrics']['f1_30'],c['metrics']['macro_f1'], -c['metrics']['false_per_min30'])) for kind in ['S1','S2']}
 preferred=max(best.values(),key=lambda c:(c['metrics']['f1_30'],c['metrics']['macro_f1']))['kind'];final_aux={m:build_aux(m,-1) for m in ['b','d']}
 with (R/'final_aux.pkl').open('xb') as f:pickle.dump(final_aux,f)
 write('training_candidates.json',candidates)
 lock=dict(brt=now(),selected=best,recommended_before_dev=preferred,objective='pooled transition-type exact F1 @ ±30 source frames, then macro stage F1; train OOF full12 only',causal=True,dev_evaluations=0,S3='same selected S2 model + temperature, ordinary noncausal Viterbi; no output dwell heuristic',initial_state='S01',states=S,graph=A.tolist(),scope='S1/S2/S3 graph excludes terminal BATCH_WAIT; S0 original including terminal',hashes={'final_aux':sha(R/'final_aux.pkl'),'script':sha(Path(__file__))},score_source='E99 frozen dev b,d; no new world model checkpoint; default E100C not used because exact score reuse requested')
 write('CONTROLLER_LOCK.json',lock)
 report('## 控制器与训练选型锁定 '+now()+'\n```json\n'+json.dumps(best,ensure_ascii=False,indent=2)+'\n```\n训练预选部署候选：'+preferred+'。S1共32候选，S2共18候选；目标为训练袋外完整轨迹typed boundary F1@±30，次级宏F1。S1因果EMA、Schmitt迟滞、最短驻留、原守卫与历史事件；S2对谓词概率作无类别权重Platt校准，用校准谓词向量训练线性softmax阶段发射，除阶段先验得到似然；在线log前向滤波，输出限制合法边及驻留。发射、校准、转移矩阵在评分fold外训练；转移含训练自环，每阶段经验驻留分布另存。P073不进入发射（固定0.5），S1继续unknown。S3同一HMM离线Viterbi，仅附录，不作部署；普通HMM自环对应几何驻留，未实现显式HSMM。S1/S2均跑b/d，融合未做。S0原样保留BATCH_WAIT；S1/S2仅10阶段，S09终止边排除。未改E97图内边、未改标签。')

if __name__=='__main__':
 try:
  if sys.argv[1]=='fit':crossfit();tune();status('LOCKED_READY_FOR_DEV')
 except Exception:
  write('FAILURE_'+str(int(time.time()))+'.json',dict(brt=now(),traceback=traceback.format_exc()));status('FAILED');raise
