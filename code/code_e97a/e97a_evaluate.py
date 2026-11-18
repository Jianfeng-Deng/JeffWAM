from e97a_core import *
import subprocess,threading
from e96a.model import WorldModel
from e96a.train import configure
from e86.data import move
CK=T/'results/e96a_m1_p1_20261114_01/ckpt_05000.pt'
CAMS=('top','left_wrist','right_wrist')
def gpurows():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 out=[]
 for l in s.splitlines():
  a=[v.strip() for v in l.split(',')];out.append(dict(index=int(a[0]),uuid=a[1],total=int(a[2]),used=int(a[3]),util=int(a[4])))
 return out
class Guard:
 def __init__(self):
  self.start=time.monotonic();self.peak=4096;self.error=None;self.done=threading.Event();rows=gpurows();good=[g for g in rows if g['util']==0 and g['total']-g['used']>=self.peak*1.5]
  if not good:raise RuntimeError('No idle GPU meets 1.5x admission; CPU inference required')
  self.g=min(good,key=lambda g:g['used']);os.environ['CUDA_VISIBLE_DEVICES']=self.g['uuid'];os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';self.ledger=dict(status='active',started=now(),gpu=self.g,estimate_peak_mib=self.peak,budget_s=1080,segments=[]);save('gpu_ledger.json',self.ledger)
  self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
 def monitor(self):
  while not self.done.wait(2):
   try:
    g=next(g for g in gpurows() if g['uuid']==self.g['uuid'])
    if g['total']-g['used']<1.5*self.peak:self.error='shared GPU memory below 1.5x reserve'
    if time.monotonic()-self.start>1000:self.error='GPU time guard'
   except Exception as e:self.error=str(e)
 def check(self):
  if self.error:raise RuntimeError(self.error)
  if time.monotonic()-self.start>1000:raise RuntimeError('GPU walltime guard')
  if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()/2**20>self.peak:raise RuntimeError('peak exceeded estimate')
 def finish(self,error=None):
  self.done.set();self.thread.join(5);elapsed=time.monotonic()-self.start;self.ledger.update(status='closed',elapsed_s=elapsed,GPUh=elapsed/3600,error=error or self.error,finished=now(),actual_peak_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0);save('gpu_ledger.json',self.ledger)
def weights(parents):
 c=collections.Counter(parents);return np.array([1/(len(c)*c[p]) for p in parents])
def aggregate_entries(entries):
 out={};rng=np.random.default_rng(90)
 for key in sorted({(r['h'],r['camera'],r['section'],r['method']) for r in entries}):
  rr=[r for r in entries if (r['h'],r['camera'],r['section'],r['method'])==key];ps=sorted({r['parent'] for r in rr});cols=['model','copy','cosine']
  a=np.array([[np.mean([r[c] for r in rr if r['parent']==p]) for c in cols] for p in ps]);v=a.mean(0);draw=a[rng.integers(len(a),size=(2000,len(a)))].mean(1);skill=1-v[0]/v[1]
  out['|'.join(map(str,key))]=dict(h=key[0],camera=key[1],section=key[2],method=key[3],windows=len(rr),parents=len(ps),mse=v[0],copy_mse=v[1],skill=float(skill),skill_ci95=np.quantile(1-draw[:,0]/draw[:,1],[.025,.975]).tolist(),cosine=v[2])
 return out

def evaluate():
 start=time.monotonic();lock=json.loads((R/'FIT_LOCK.json').read_text());assert lock['status']=='coefficients_locked_before_dev'
 for p,h in lock['hashes'].items():assert sha(R/p)==h
 with (R/'DEV_ONCE_LOCK.json').open('x') as f:json.dump(dict(started=now(),protocol='single immutable pass; no alpha or model selection'),f)
 _,train,dev=datasets();st=States();models={(h,m):dict(np.load(R/f'{m}_h{h}.npz')) for h in H for m in ('R0','R1','R2')};old={h:dict(np.load(E93/f'ridge_h{h}_coefficients.npz')) for h in H};buf={h:dict(Y=[],X=[],Z=[],idx=[],parents=[]) for h in H};cov=collections.defaultdict(collections.Counter)
 for i,row in enumerate(dev.rows):
  b=dev.get(i,H,'short','P1');source=dev.index.captures[row['canonical_id']]['source_collection'];cov[source]['windows']+=1
  for h in H:
   off=H.index(h)*24 if h<=50 else 72;n=24 if h<=50 else 3;sl=slice(off,off+n)
   if not b['world_valid'][sl].all():continue
   cov[source][f'image_h{h}']+=1;feat,why=st.features(dev,row,h)
   if feat is None:cov[source][f'excluded_h{h}:{why}']+=1;continue
   cov[source][f'paired_h{h}']+=1;d=buf[h];d['Y'].append(b['world'][sl].numpy());d['X'].append(x0(b['obs'].numpy(),b['history_valid'].numpy(),h,old[h]));d['Z'].append(feat);d['idx'].append(i);d['parents'].append(row['parent_component'])
 save('state_coverage_dev.json',{k:dict(v) for k,v in cov.items()})
 for h,d in buf.items():
  d['Y']=np.stack(d['Y']);d['X']=np.stack(d['X']);d['Z']=np.stack(d['Z']);d['pred']={}
  for method in ('R0','R1','R2'):
   m=models[h,method];d['pred'][method]=predict(m,(d['X']-m['xm'])/m['xs'],(d['Z']-m['zm'])/m['zs'],method).astype('f4')
 # Required frozen E96a forward only. Data preparation and all fitting precede GPU admission.
 guard=None
 try:
  configure();cp=torch.load(CK,map_location='cpu',weights_only=False);assert cp['step']==5000;assert sha(CK)=='e55c33857e59aaceef559a510e874c602d3827e232cf5817e5edeb7db6fc9a98'
  model=WorldModel();model.load_state_dict(cp['model']);del cp;model.eval()
  if os.environ.get('E97A_CPU_ONLY')=='1':
   device='cpu';save('cpu_forward_device.json',dict(device='cpu',reason='GPU0 admitted by free-memory/utilization but CUDA returned busy/unavailable before allocation; no further GPU attempts'))
  else:
   try:guard=Guard();device='cuda'
   except RuntimeError as e:
    if 'No idle' not in str(e):raise
    device='cpu';save('gpu_ledger.json',dict(status='CPU_fallback_no_idle_device',GPUh=0,reason=str(e)))
  model=model.to(device)
  with torch.inference_mode():
   for h,d in buf.items():
    preds=[]
    for j in range(0,len(d['idx']),8):
     if guard:guard.check()
     batch=dev.batch(d['idx'][j:j+8],(h,),'short','P1');pr=model.mean_rollout(move(batch,device)).cpu();delta=(pr-batch['world_current'])/batch['world_scale'];preds.extend(delta.numpy())
    d['pred']['E96a']=np.stack(preds);np.savez_compressed(R/f'dev_forward_snapshot_h{h}.npz',target=d['Y'],R0=d['pred']['R0'],R1=d['pred']['R1'],R2=d['pred']['R2'],E96a=d['pred']['E96a'],indices=d['idx'],parents=d['parents']);status('E96a_forward',h=h,windows=len(preds),device=device)
  del model
  if guard:torch.cuda.synchronize();torch.cuda.empty_cache();guard.finish();guard=None
 except BaseException as e:
  if guard:guard.finish(str(e))
  raise
 entries=[];decomp=[];evs=[];checks=[]
 refs=json.loads((T/'results/e96a_control_20261114_01/E96a_dev_records.json').read_text());ref={(r['index'],r['h']):r for r in refs}
 for h,d in buf.items():
  Y=d['Y'].astype('f8');n,k,_=Y.shape;nt=k//3;parents=d['parents'];w=weights(parents);S=d['pred']['R1'].astype('f8');E=Y-S
  # Fixed train-only output subspaces: each token's R1 coefficient rows and intercept.
  Q=[];ranks=[]
  for j in range(k):
   A=np.concatenate([models[h,'R1']['w1'][j],models[h,'R1']['ym'][j][None]],0).astype('f8');_,sv,V=np.linalg.svd(A,full_matrices=False);rank=int((sv>sv[0]*max(A.shape)*np.finfo('f4').eps).sum());Q.append(V[:rank].T);ranks.append(rank)
  def project(A):return np.stack([(A[:,j]@Q[j])@Q[j].T for j in range(k)],1)
  Yrob=project(Y);Yenv=Y-Yrob
  for i,index in enumerate(d['idx']):
   mse=float(((d['pred']['E96a'][i]-Y[i])**2).mean());assert abs(mse-ref[index,h]['model'])<2e-5
  checks.append(dict(h=h,E96a_max_abs_old_mse_diff=max(abs(float(((d['pred']['E96a'][i]-Y[i])**2).mean())-ref[index,h]['model']) for i,index in enumerate(d['idx'])),R0_max_abs_old_mse_diff=None,projection_ranks=ranks))
  pred=dict(d['pred'],copy=np.zeros_like(Y));predproj={m:project(p.astype('f8')) for m,p in pred.items()}
  for cj,cam in list(enumerate(CAMS))+[(None,'all')]:
   sl=slice(None) if cj is None else slice(cj*nt,(cj+1)*nt)
   y=Y[:,sl];s=S[:,sl];e=E[:,sl];energy=np.mean(y*y,(1,2));rem=np.mean(e*e,(1,2));se=np.mean(s*s,(1,2));cross=2*np.mean(s*e,(1,2));yr=Yrob[:,sl];ye=Yenv[:,sl]
   pm=lambda a:float(w@a);den=pm(energy)
   decomp.append(dict(h=h,camera=cam,windows=n,parents=len(set(parents)),copy_energy=den,robot_point_explained_share=1-pm(rem)/den,environment_unexplained_share=pm(rem)/den,robot_prediction_energy_share=pm(se)/den,cross_share=pm(cross)/den,additivity_error=abs(pm(se+rem+cross-energy)),robot_subspace_energy_share=pm(np.mean(yr*yr,(1,2)))/den,environment_subspace_energy_share=pm(np.mean(ye*ye,(1,2)))/den,subspace_additivity_error=abs(pm(np.mean(yr*yr+ye*ye-y*y,(1,2))))))
   for method,p in pred.items():
    pp=p[:,sl].astype('f8');pe=pp-y;ym=np.einsum('n,nkd->kd',w,y);em=np.einsum('n,nkd->kd',w,pe);mse=pm(np.mean(pe*pe,(1,2)));varY=den-float(np.mean(ym*ym));ev=1-(mse-float(np.mean(em*em)))/varY
    evs.append(dict(h=h,camera=cam,method=method,explained_variance=ev))
    for section,target,prediction in [('full',y,pp),('environment_subtraction',e,pp-s),('environment_subspace',ye,pp-predproj[method][:,sl]),('robot_subspace',yr,predproj[method][:,sl])]:
     # Subtraction copy means zero environmental change conditional on oracle R1, hence skill(R1)=0.
     if method=='copy' and section=='environment_subtraction':prediction=np.zeros_like(e)
     mse_i=np.mean((prediction-target)**2,(1,2));cp=np.mean(target*target,(1,2));dot=np.sum(prediction*target,(1,2));cos=dot/np.maximum(np.sqrt(np.sum(prediction**2,(1,2))*np.sum(target**2,(1,2))),1e-12)
     for ii in range(n):entries.append(dict(h=h,camera=cam,section=section,method=method,parent=parents[ii],index=d['idx'][ii],model=float(mse_i[ii]),copy=float(cp[ii]),cosine=float(cos[ii])))
  np.savez_compressed(R/f'dev_predictions_h{h}.npz',target=Y.astype('f4'),R0=d['pred']['R0'],R1=d['pred']['R1'],R2=d['pred']['R2'],E96a=d['pred']['E96a'],indices=d['idx'],parents=parents)
  save('evaluation_audit.json',checks)
 save('dev_records.json',entries);summ=aggregate_entries(entries)
 evlookup={(r['h'],r['camera'],r['method']):r['explained_variance'] for r in evs}
 for r in summ.values():
  if r['section']=='full':r['explained_variance']=evlookup[r['h'],r['camera'],r['method']]
 save('dev_summary.json',summ);save('energy_decomposition.json',decomp)
 gain=[]
 for h in H:
  for cam in (*CAMS,'all'):
   total=summ[f'{h}|{cam}|full|E96a'];rr=summ[f'{h}|{cam}|robot_subspace|E96a'];ee=summ[f'{h}|{cam}|environment_subspace|E96a'];g=total['copy_mse']-total['mse'];gr=rr['copy_mse']-rr['mse'];ge=ee['copy_mse']-ee['mse'];gain.append(dict(h=h,camera=cam,total_gain=g,robot_subspace_gain=gr,environment_subspace_gain=ge,robot_subspace_gain_share=gr/g,environment_subspace_gain_share=ge/g,additivity_error=abs(g-gr-ge)))
 save('gain_attribution.json',gain);save('EVALUATION_COMPLETE.json',dict(status='completed_single_dev_pass',brt=now(),seconds=time.monotonic()-start,windows={h:len(d['idx']) for h,d in buf.items()},GPU=json.loads((R/'gpu_ledger.json').read_text()),no_model_training=True))
 status('evaluation_completed',seconds=time.monotonic()-start)
if __name__=='__main__':
 try:evaluate()
 except BaseException:save('evaluation_failure.json',dict(brt=now(),traceback=traceback.format_exc()));raise
