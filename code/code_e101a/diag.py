import os,sys,json,time,traceback,collections,subprocess,threading,datetime,hashlib
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu'); OUT=T/'results/e101a_diag_20261115_01'; REPORT=T/'notes/e101a_action_diag_20261115.md'
sys.path.insert(0,str(T/'code_e100c'))
import numpy as np,torch
from e99w2_data import load,pose_np
from e99w2_model import WorldModel
from e86.data import move,JOINTS,GRIPS
from e96a.train import configure
configure()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def save(n,x):
 p=OUT/n; tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));os.replace(tmp,p)
def report(s):
 with REPORT.open('a') as f:f.write('\n'+s+'\n')
def status(s,**kw):save('status_'+sys.argv[1]+'.json',dict(status=s,brt=now(),**kw));print(s,kw,flush=True)
def metrics(pred,y,q,parents):
 out={};parents=np.array(parents)
 for h in (8,50):
  e=pred[:,:h]-y[:,:h];c=q[:,None]-y[:,:h]
  agg=lambda a:np.mean([a[parents==p].mean(0) for p in sorted(set(parents))],0)
  mae=agg(np.abs(e).mean(1)); mse=agg((e**2).mean(1));copy=agg((c*c).mean(1));cm=agg(np.abs(c).mean(1))
  out[str(h)]=dict(joint_L1_rad=float(mae[JOINTS].mean()),gripper_L1_mm=float(mae[GRIPS].mean()*1000),dim_L1=mae.tolist(),dim_RMSE=np.sqrt(mse).tolist(),dim_skill=(1-mse/np.maximum(copy,1e-20)).tolist(),joint_skill=float(1-mse[JOINTS].mean()/copy[JOINTS].mean()),gripper_skill=float(1-mse[GRIPS].mean()/copy[GRIPS].mean()),copy_joint_L1_rad=float(cm[JOINTS].mean()),copy_gripper_L1_mm=float(cm[GRIPS].mean()*1000))
 return out

def prepare():
 status('loading');_,_,tr,dev=load();groups=collections.defaultdict(list)
 for i,r in enumerate(tr.rows):groups[r['parent_component']].append(i)
 ps=sorted(groups);ids=[]
 for p in ps:
  ii=sorted(groups[p],key=lambda i:(tr.rows[i]['canonical_id'],tr.rows[i]['anchor']))
  ids.extend(ii[j] for j in np.unique(np.linspace(0,len(ii)-1,min(16,len(ii))).astype(int)))
 fixed=[sorted(groups[p],key=lambda i:(tr.rows[i]['canonical_id'],tr.rows[i]['anchor']))[len(groups[p])//2] for p in ps[:64]]
 assert not set(ps)&{r['parent_component'] for r in dev.rows}
 metadata={};audit={}
 for name,d,indices in [('train',tr,ids),('train64',tr,fixed),('dev',dev,list(range(48)))]:
  xs=[];im=[];ys=[];qs=[];pars=[];raw=[];maxeq=0.;roundtrip=0.;anchors=[]
  for k,i in enumerate(indices):
   b=d.get(i,(10,),'short');r=d.rows[i];q=b['q_current'].numpy();ctx=b['context'].numpy();qhist=ctx[:14][None]+ctx[14:70].reshape(4,14);poses=pose_np(qhist);valid=ctx[120:124];poses=(poses-ctx[84:102])*valid[:,None]
   x=np.concatenate([ctx[:70],ctx[84:102],poses.ravel(),valid]);y=b['action'].numpy();z,zz,_,_=d.states.get(d,r['canonical_id']);a=r['anchor'];maxeq=max(maxeq,float(np.abs(y-zz[a+1:a+51]).max()))
   mean=b['action_mean'].numpy();std=b['action_std'].numpy();anc=b['q_anchor'].numpy();roundtrip=max(roundtrip,float(np.abs(((y-anc-mean)/std)*std+mean+anc-y).max()))
   xs.append(x);im.append(b['obs'][4].numpy().reshape(-1));ys.append(y);qs.append(q);pars.append(r['parent_component']);raw.append(dict(index=i,parent=r['parent_component'],cid=r['canonical_id'],anchor=a));anchors.append(anc)
   if k%256==0:status('extract_'+name,done=k,total=len(indices))
  np.savez_compressed(OUT/(name+'.npz'),x=np.array(xs),image=np.array(im),y=np.array(ys),q=np.array(qs),parents=np.array(pars),indices=indices)
  metadata[name]=raw;audit[name]=dict(windows=len(indices),parents=len(set(pars)),target_next_q_max=maxeq,roundtrip_max=roundtrip,anchor_grips_zero=bool(np.all(np.array(anchors)[:,GRIPS]==0)),action_std_min=np.min(std,axis=0).tolist(),action_std_max=np.max(std,axis=0).tolist(),action_mean=mean.tolist(),copy=metrics(np.repeat(np.array(qs)[:,None],50,1),np.array(ys),np.array(qs),pars))
 save('fixed_windows.json',metadata);save('contract_audit.json',audit)
 assert all(v['target_next_q_max']<1e-6 and v['roundtrip_max']<1e-6 and v['anchor_grips_zero'] for v in audit.values())
 report('## 1 口径实际核查完成\n固定抽样见 `results/e101a_diag_20261115_01/fixed_windows.json`。关节顺序 L6,Lgrip,R6,Rgrip；关节 rad，夹爪 signed m。q_anchor 的关节为当前 q，夹爪为0；反变换 action_mean + std × FM输出 + q_anchor。各offset训练统计，无单位换算。当前复制使用 raw signed gripper，与模型反变换同坐标。CPU逐窗往返及next-frame反馈等值核查通过（详细误差见contract_audit.json）；未发现新口径bug。E88截零复制bug已修复，旧版本不能直接比较。')
 status('complete',counts={k:v['windows'] for k,v in audit.items()})

def gpu():
 status('loading');_,_,tr,dev=load();cp=torch.load(T/'results/e100c_c2_s90_20261115_03/ckpt_05000.pt',map_location='cpu',weights_only=False);assert cp['step']==5000;m=WorldModel(tr.norm);m.load_state_dict(cp['model']);del cp
 torch.backends.mha.set_fastpath_enabled(False)
 def rows():
  s=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
  return [(a.strip(),int(b.strip())) for a,b in (l.split(',') for l in s.splitlines())]
 peak=4096;chosen=sorted(rows(),key=lambda r:r[1],reverse=True)[0];assert chosen[1]>=peak*1.5
 os.environ['CUDA_VISIBLE_DEVICES']=chosen[0];start=time.monotonic();save('gpu_ledger.json',dict(status='active',start=now(),pid=os.getpid(),gpu=chosen[0],free_admission_mib=chosen[1],estimate_mib=peak))
 error=[];stop=threading.Event()
 def monitor():
  while not stop.wait(2):
   try:
    free=dict(rows())[chosen[0]]
    if free<peak*1.5:error.append('shared memory pressure');return
   except BaseException as e:error.append(str(e));return
 def check():
  assert not error,str(error)
  assert time.monotonic()-start<3300,'budget exhausted'
  assert torch.cuda.max_memory_allocated()/2**20<=peak,'peak estimate exceeded'
 thread=threading.Thread(target=monitor,daemon=True);thread.start();results={};predstore={}
 try:
  m=m.to('cuda').eval();meta=json.loads((OUT/'fixed_windows.json').read_text())
  with torch.inference_mode():
   for name,d in [('dev',dev),('train64',tr)]:
    dat=np.load(OUT/(name+'.npz'));preds={f'S{s}_N{n}':[] for s in (4,8,16) for n in (1,8,16)};preds.update({f'S{s}_zero':[] for s in (4,8,16)})
    for k,r in enumerate(meta[name]):
     check();b=move(d.batch([r['index']],(10,),'short'),'cuda')
     for s in (4,8,16):
      samples=[]
      for n in range(16):
       check();a=m.infer(b,'action_only',s,90+n)[0][0].cpu().numpy();samples.append(a)
       if n+1 in (1,8,16):preds[f'S{s}_N{n+1}'].append(np.mean(samples,0))
      a=torch.zeros(1,50,14,device='cuda')
      for j in range(s):
       t=torch.full((1,),j/s,device='cuda');a=a+m(b,a,None,t,t)['action']/s
      preds[f'S{s}_zero'].append((b['action_mean']+a*b['action_std']+b['q_anchor'][:,None])[0].cpu().numpy())
     if k%8==0:status('inference_'+name,done=k,total=len(meta[name]),gpu_s=time.monotonic()-start)
    results[name]={method:metrics(np.array(v),dat['y'],dat['q'],dat['parents']) for method,v in preds.items()};predstore.update({name+'_'+k:np.array(v) for k,v in preds.items()});save('sampling_metrics.json',results);np.savez_compressed(OUT/'predictions.npz',**predstore)
  status('complete',gpu_s=time.monotonic()-start)
 except BaseException as e:
  error.append(repr(e));raise
 finally:
  stop.set();thread.join(3);elapsed=time.monotonic()-start;save('gpu_ledger.json',dict(status='closed',start_gpu=chosen,elapsed_s=elapsed,GPUh=elapsed/3600,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,error=error,finish=now()));del m;torch.cuda.empty_cache()

def probe():
 status('loading');tr=np.load(OUT/'train.npz');ds={k:np.load(OUT/(k+'.npz')) for k in ('dev','train64')};X=tr['x'].astype('f8');images=tr['image'];pars=tr['parents'];y=tr['y']-tr['q'][:,None];yt=y.reshape(len(y),-1);results={};details={}
 # Fixed training-only random projection preserves all 24 current spatial tokens, no dev fitting.
 rng=np.random.default_rng(101);proj=rng.normal(size=(images.shape[1],128)).astype('f4')/np.sqrt(128);ip=images@proj;hold=np.array([sorted(set(pars)).index(p)%5==0 for p in pars])
 for version in ('state','state_image'):
  xx=X if version=='state' else np.concatenate([X,ip],1);other={k:(d['x'] if version=='state' else np.concatenate([d['x'],d['image']@proj],1)) for k,d in ds.items()}
  # Select ridge alpha on held-out TRAIN parents only, then refit all train windows.
  def scaled(a,fit):
   mu=fit.mean(0);sd=fit.std(0);sd[sd<1e-5]=1;return np.concatenate([(a-mu)/sd,np.ones((len(a),1))],1),mu,sd
  a,mu,sd=scaled(xx[~hold],xx[~hold]);av=np.concatenate([(xx[hold]-mu)/sd,np.ones((hold.sum(),1))],1);yy=yt[~hold];targetscale=yy.std(0).clip(.0001);scores={}
  for alpha in (.1,1.,10.,100.,1000.):
   reg=np.eye(a.shape[1])*alpha;reg[-1,-1]=0;coef=np.linalg.solve(a.T@a+reg,a.T@yy);scores[alpha]=float(np.mean(((av@coef-yt[hold])/targetscale)**2))
  alpha=min(scores,key=scores.get);a,mu,sd=scaled(xx,xx);reg=np.eye(a.shape[1])*alpha;reg[-1,-1]=0;coef=np.linalg.solve(a.T@a+reg,a.T@yt)
  for k,d in ds.items():
   aa=np.concatenate([(other[k]-mu)/sd,np.ones((len(d['q']),1))],1);pred=(aa@coef).reshape(-1,50,14)+d['q'][:,None];results[version+'_ridge_'+k]=metrics(pred,d['y'],d['q'],d['parents'])
  status('ridge_'+version,alpha=alpha)
  # Small deterministic delta MLP, CPU only, fixed 1000 updates; residual zero initialization.
  torch.manual_seed(101);net=torch.nn.Sequential(torch.nn.Linear(a.shape[1]-1,128),torch.nn.ReLU(),torch.nn.Linear(128,700));torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);at=torch.tensor(a[:,:-1],dtype=torch.float32);ys=yt.std(0).clip(.0005);target=torch.tensor(yt/ys,dtype=torch.float32);hist=[];gen=torch.Generator().manual_seed(101)
  for step in range(1000):
   ix=torch.randint(len(a),(128,),generator=gen);loss=((net(at[ix])-target[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
   if step%100==0:hist.append([step,float(loss)])
  net.eval()
  with torch.no_grad():
   for k,d in ds.items():
    aa=torch.tensor((other[k]-mu)/sd,dtype=torch.float32);pred=(net(aa).numpy()*ys).reshape(-1,50,14)+d['q'][:,None];results[version+'_mlp_'+k]=metrics(pred,d['y'],d['q'],d['parents'])
  details[version]=dict(ridge_alpha=alpha,train_parent_validation_scores=scores,mlp_updates=1000,mlp_loss=hist,input_dim=xx.shape[1],image_projection='fixed Gaussian128 seed101 all24x768 current tokens; not full-token capacity claim',selection='ridge alpha only train-parent holdout; fixed MLP1000, no dev selection')
  save('probe_metrics.json',results);save('probe_config.json',details);status('probe_'+version)
 status('complete',cuda_initialized=torch.cuda.is_initialized())

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  save('failure_'+sys.argv[1]+'.json',dict(brt=now(),traceback=traceback.format_exc()));status('failed');raise
