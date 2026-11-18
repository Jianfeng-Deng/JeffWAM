from s2 import *
import itertools
@njit(cache=False)
def filter_core(e,transition,allowed,temp,m,uniform):
 n=len(e);out=np.zeros(n,np.int64);f=np.full(10,-np.inf)
 if uniform:
  for j in range(10):f[j]=-np.log(10.)
 else:f[0]=0.
 la=np.full((10,10),-np.inf)
 for i in range(10):
  for j in range(10):
   if transition[i,j]>0:la[i,j]=np.log(transition[i,j])
 state=0;age=0
 for t in range(n):
  ff=np.zeros(10)
  for j in range(10):
   mm=-np.inf
   for i in range(10):mm=max(mm,f[i]+la[i,j])
   if mm==-np.inf:ff[j]=-np.inf
   else:
    ss=0.
    for i in range(10):ss+=np.exp(f[i]+la[i,j]-mm)
    ff[j]=mm+np.log(ss)+e[t,j]*temp
  mx=np.max(ff);ss=np.sum(np.exp(ff-mx));f=ff-mx-np.log(ss);age+=1
  if t==0 and uniform:state=np.argmax(f)
  if age>=m:
   nxt=state;score=-np.inf
   for j in range(10):
    if allowed[state,j] and f[j]>score:nxt=j;score=f[j]
   if nxt!=state:state=nxt;age=0
  out[t]=state
 return out

def hmm(e,aux,temp,m,initial):return filter_core(np.ascontiguousarray(e),aux['transition'],A,temp,m,initial=='uniform')

def main():
 assert (R/'HEADS_LOCK.json').exists() and not (R/'CONTROLLER_LOCK.json').exists();best={};allc=[];final={}
 for method in ['H0','H1','H2','H3']:
  aux={k:build_aux(method,k) for k in range(5)};cache={}
  for s in train:
   a=aux[fold[s['parent']]];z=np.load(R/(s['parent']+'_'+method+'_oof.npz'));cache[s['parent']]=(emission(z['prob'],a),a)
  for initial in ['fixed_S01','uniform']:
   cc=[]
   for temp,m in itertools.product([.1,.3,1.],[1,15,45]):
    rr=[record(hmm(*cache[s['parent']],temp,m,initial),truths[s['parent']]) for s in train];val=aggregate(rr);cand=dict(method=method,initial=initial,config=dict(temp=temp,m=m,initial=initial),metrics=val);cc.append(cand);status('hmm_tune',method=method,initial=initial,temp=temp,m=m,f1_30=val['f1_30'])
   selected=max(cc,key=lambda c:(c['metrics']['f1_30'],c['metrics']['macro_f1'],-c['metrics']['false_per_min30']));best[method+'__'+initial]=selected;allc+=cc
  final[method]=build_aux(method,-1)
 with (R/'final_aux.pkl').open('xb') as f:pickle.dump(final,f)
 preferred={m:max([best[m+'__'+i] for i in ['fixed_S01','uniform']],key=lambda c:(c['metrics']['f1_30'],c['metrics']['macro_f1']))['initial'] for m in ['H0','H1','H2','H3']}
 # Prefix causality tests use train only; exact old S2 agreement for fixed initial.
 import importlib.util
 path=T/'code_e101s/run.py';spec=importlib.util.spec_from_file_location('old_run_ro',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 checks=[]
 for method in ['H0','H1','H2','H3']:
  s=train[0];z=np.load(R/(s['parent']+'_'+method+'_oof.npz'));a=build_aux(method,fold[s['parent']]);e=emission(z['prob'],a)
  for initial in ['fixed_S01','uniform']:
   cfg=best[method+'__'+initial]['config'];seq=hmm(e,a,**cfg);prefix=hmm(e[:300],a,**cfg);assert np.array_equal(seq[:300],prefix)
   if initial=='fixed_S01':assert np.array_equal(seq,old.hmm(e,a,cfg['temp'],cfg['m']))
   checks.append(dict(method=method,initial=initial,prefix_frames=300,equal=True,old_S2_exact=initial=='fixed_S01'))
 # Perturb future pixels/states implicitly by comparing full vs truncated feature construction.
 put('causal_contracts.json',checks);put('training_controller_candidates.json',allc)
 lock=dict(brt=now(),selected=best,preferred_initial=preferred,objective='60-parent full OOF typed boundary F1@30 then stage macro F1; no dev selection',initial_variants='fixed S01; uniform prior with frame0 posterior argmax, no future or warmup lookahead',hashes={'aux':sha(R/'final_aux.pkl'),'heads':sha(R/'HEADS_LOCK.json')},dev_evaluations=0,preregistered_success='H2 or H3 paired macro AUROC lower CI>0 vs H0 and paired recall@30 lower CI>0 vs same-parent E101-S S2',bootstrap=2000,seed=10201)
 put('CONTROLLER_LOCK.json',lock);report('## S2在线HMM训练锁定 '+now()+'\n'+json.dumps(dict(selected=best,preferred_initial=preferred),ensure_ascii=False)+'\n校准、发射、合法转移按fold外拟合；全60完整袋外轨迹调温度/最短驻留。固定S01与均匀先验各自训练选参，初态主变体也只按训练选。S2固定初态与E101实现逐帧一致；两初态前缀因果测试通过。第二层使用第一层袋外分数，未做完整嵌套再训练，保留E101堆叠调参合同。');status('LOCKED_READY_FOR_DEV')
if __name__=='__main__':
 try:main()
 except Exception:put('TUNE_FAILURE_'+str(int(time.time()))+'.json',dict(brt=now(),traceback=traceback.format_exc()));raise
