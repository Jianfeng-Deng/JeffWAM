import os,sys,json,time,traceback,copy,collections,hashlib,datetime
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';R=T/'notes/e109g_gripper_generalization_20261116.md'
sys.path.insert(0,str(T/'code_e108g'));import collect as old;old.common.CONTROL=T/'results/e106c_control_20261115_02'
import analyze as ev
import numpy as np,torch,h5py
from e86.data import build_window
from e99w2_data import PlannedData

def put(n,x):
 p=O/n;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),allow_nan=False));os.replace(tmp,p)
def report(s):
 with R.open('a') as f:f.write('\n'+s+'\n')
def brt():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def candidates(c,data):
 cid=c['canonical_id'];z,q,_,good=data.states.get(data,cid);n=len(q);t=z['timestamp'];ct=z['camera_exposure_timestamp'];rc=z['camera_received_timestamp'];br=z['state_source_brackets'];avail=np.maximum(t,np.max(br,axis=(1,2)));seg=np.cumsum(np.r_[0,(np.diff(t)<=0)|(np.diff(t)>.05)]);valid=np.isfinite(br).all((1,2))&((br[:,:,1]-br[:,:,0]).max(1)<=.075)
 # Same numeric view as E93 prepare; no train P3 interval/pool filter, pure recording timing.
 zz=dict(timestamp=t,camera_exposure_timestamp=ct,camera_received_timestamp=rc,state_available_time=avail,continuous_segment=seg,original_action_index=np.arange(1,n),p1_source_mask=np.ones(n,bool),p3_source_mask=np.ones(n,bool));cc=dict(c,split='dev');out=[];reasons=collections.Counter()
 for a0 in np.unique(np.linspace(24,n-53,16).astype(int)//4*4):
  a=int(a0)
  if a<24 or a+50>=n or seg[a-24]!=seg[a+50] or not valid[a-24:a+51].all():reasons['segment_or_state']+=1;continue
  oi=np.empty((2,3),np.int64);wi=np.empty((3,3),np.int64)
  for k in range(3):
   ac=np.maximum.accumulate(np.maximum(ct[:,k],rc[:,k]));oi[:,k]=np.searchsorted(ac,avail[[a-6,a]],side='right')-1
   for j,off in enumerate((10,25,50)):
    rr=np.arange(max(0,a+off-2),min(n,a+off+3));wi[j,k]=rr[np.argmin(abs(ct[rr,k]-t[a+off]))]
  if any(abs(ct[wi[j,k],k]-t[a+off])>1/60+1e-6 for j,off in enumerate((10,25,50)) for k in range(3)):reasons['world_clock']+=1;continue
  try:w=build_window(cc,zz,a,oi,wi,'P1')
  except ValueError as e:reasons[str(e)]+=1;continue
  w.update(split=c['split'],_stage='P3',source='piper',stratum=next(x for x in ('demo3','demo4','demo6') if x in c['source_collection']))
  out.append(w)
  if len(out)==8:break
 return out,dict(reasons)
def main():
 start=time.monotonic();assert (O/'BASELINE_REPRODUCED.json').exists() and not (O/'COLLECT_COMPLETE.json').exists();old.configure();torch.backends.mha.set_fastpath_enabled(False);p,ready,tr,dev=old.datasets();contract=old.contract();oldrows=json.loads((T/'results/e108g_diag_20261116_01/predictions.json').read_text());oldtrain=[r for r in oldrows if r['split']=='train'];olddev=[r for r in oldrows if r['split']=='dev']
 # Audit candidate reconstruction for every frozen dev parent before new scores.
 da=[]
 for cid in dict.fromkeys(r['canonical_id'] for r in dev.rows):
  cc=dev.index.captures[cid];ws,why=candidates(cc,dev);actual=[r for r in dev.rows if r['canonical_id']==cid];expected=[r['anchor'] for r in actual];got=[r['anchor'] for r in ws];same=got==expected
  da.append(dict(cid=cid,parent=cc['parent_component'],anchors_expected=expected,anchors_reconstructed=got,exact_anchors=same,exact_obs=all(np.array_equal(a['obs_rows'],b['obs_rows']) for a,b in zip(actual,ws)),rejects=why))
 assert all(x['exact_anchors'] and x['exact_obs'] for x in da),'DEV_RULE_RECONSTRUCTION_MISMATCH'
 put('DEV_RULE_AUDIT.json',da)
 parents={r['parent_component'] for r in tr.rows};eligible={r['canonical_id'] for r in tr.rows};selected=[];capture_audit=[];seen=set()
 for c in p.captures:
  par=c['parent_component'];cid=c['canonical_id']
  if par not in parents or par in seen or cid not in eligible:continue
  assert cid not in p.excluded and c['split']=='train';ws,why=candidates(c,tr);capture_audit.append(dict(parent=par,cid=cid,windows=len(ws),rejects=why))
  if ws:selected+=ws;seen.add(par)
 assert seen==parents,(len(seen),len(parents))
 # All selection before output; no replacements due to feature/model results.
 put('ALIGNED_WINDOW_LOCK.json',dict(rule='exact E93 first8 of16 fixed quantiles; one first eligible P3 canonical in original manifest order per P3 training parent; no P3 interval filtering; same human short input',rows=selected,capture_audit=capture_audit,dev_rule_exact_all27=True,brt=brt()))
 aligned=PlannedData(selected,p.captures,ready['cache'],tr.base.stats,{'scale':tr.norm},{'__human_default__':dev.base.text_cache['__human_default__']},stage='P3');aligned.states=tr.states
 # Existing caches only; eligibility is frozen regardless of cache availability.
 for i,r in enumerate(selected):
  try:aligned.batch([i],(10,),'short')
  except Exception as e:put('ALIGNED_CACHE_FAILURE.json',dict(index=i,row=r,error=repr(e)));raise
 report('## 窗口规则锁定与复核\nE93规则在冻结dev全部27parent/211窗的anchor及obs_rows原样重建一致。训练145parent按原manifest顺序取首个属于P3训练池的canonical，按整段录制16分位候选前8合法窗；不施加P3阶段区间过滤。窗口在模型输出前锁定，全部只读既有特征可取。short human default、50帧、当前实测状态初始化/原onset标签初值、一一匹配、parent等权均保持一致。\n')
 labels={};suites={'train_aligned':[],'train16':oldtrain,'dev':olddev};
 for i,r in enumerate(selected):
  z,q,_,good=aligned.states.get(aligned,r['canonical_id']);cid=r['canonical_id'];a=r['anchor']
  if cid not in labels:labels[cid]=old.onset_labels(q,z['continuous_segment'],good,contract)[0]
  b=aligned.batch([i],(10,),'short');lab=labels[cid];suites['train_aligned'].append(dict(split='train_aligned',index=i,cid=cid,parent=r['parent_component'],anchor=a,current=b['grip_current'][0].numpy().astype(int).tolist(),label_init=lab[a].tolist(),labels=lab[a+1:a+51].tolist(),width=q[a:a+51,[6,13]].tolist(),original48=False))
 put('WINDOW_TRUTH.json',suites)
 lock=json.loads((old.common.CONTROL/'PRETRAIN_LOCK.json').read_text());steps=[2000,5000,10000,15000,20000];groups=['A2','F0','F2','S15000','S20000'];allmetrics={};hashes={}
 for step,group in zip(steps,groups):
  ck=Path(lock['checkpoints'][group]);hashes[str(step)]=old.sha(ck);cp=torch.load(ck,map_location='cpu',weights_only=False);m=old.WorldModel(tr.norm);m.load_state_dict(cp['model']);del cp;m.eval().requires_grad_(False);allmetrics[str(step)]={}
  for name,data,rows in [('train_aligned',aligned,suites['train_aligned']),('train16',tr,oldtrain),('dev',dev,olddev)]:
   ss=[];logits=[]
   with torch.inference_mode():
    for j,r in enumerate(rows):
     # 20k old16/dev can reuse only after independent211 baseline reproduction.
     if step==20000 and name in ('train16','dev'):log=np.array(r['logits'],dtype='f4')
     else:
      b=data.batch([r['index']],(10,),'short');tt=torch.zeros(1);log=m(b,torch.zeros(1,50,14),None,tt,tt)['grip_logits'][0].numpy()
     assert np.isfinite(log).all();ss.append((log>0).astype('i1'));logits.append(log)
     if j%200==0:put('status.json',dict(status='inference',step=step,suite=name,done=j,total=len(rows),elapsed_s=time.monotonic()-start,GPUh=0))
   np.savez_compressed(O/f'logits_{step}_{name}.npz',logits=np.array(logits));allmetrics[str(step)][name]=ev.score(rows,ss)
   put('METRICS_PARTIAL.json',allmetrics);print('SCORE',step,name,{k:round(v['f1'],6) for k,v in allmetrics[str(step)][name].items()},flush=True)
  del m
  assert old.sha(ck)==hashes[str(step)]
 put('METRICS.json',allmetrics);put('CHECKPOINT_HASHES.json',hashes);assert not torch.cuda.is_initialized();put('COLLECT_COMPLETE.json',dict(status='complete',elapsed_s=time.monotonic()-start,GPUh=0,windows={k:len(v) for k,v in suites.items()},brt=brt()))
if __name__=='__main__':
 try:main()
 except BaseException:put('COLLECT_FAILED.json',dict(traceback=traceback.format_exc(),GPUh=0,brt=brt()));raise
