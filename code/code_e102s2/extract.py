from common import *
import numpy as np
OLD=T/'results/e99b3_20261115_01'
def oldread(n):return json.loads((OLD/n).read_text())
def run():
 import h5py,importlib.util
 sources=oldread('sources.json');assert len(sources)==80 and all(s['cid']!=EXCLUDE for s in sources)
 dest=R/'features';dest.mkdir(exist_ok=True)
 # Two fixed training-only mixture midpoint thresholds; no analog gripper value is retained.
 qs=[]
 for s in sources:
  if s['pool']=='train':
   with h5py.File(s['source'],'r') as h:qs.append(h['observations/qpos_raw'][::30,[6,13]] if False else h['observations/qpos_raw'][:][::30][:,[6,13]])
 vals=np.concatenate(qs);thresholds=[]
 for j in range(2):
  v=vals[:,j];a,b=np.quantile(v,[.1,.9])
  for _ in range(30):
   cut=(a+b)/2;lo=v[v<=cut];hi=v[v>cut]
   if not len(lo) or not len(hi):break
   a,b=lo.mean(),hi.mean()
  thresholds.append(float((a+b)/2))
 put('GRIPPER_LOCK.json',dict(training_only=True,thresholds=thresholds,method='two-cluster midpoint on train qpos_raw stride30; state=raw>threshold, 1=open; no opening regression',brt=now()))
 fkpath=Path('/mnt/workspace/dengjianfeng/demo4_zip_tie_active_gaze_20261030/bench/piper_bench/fk.py');sp=importlib.util.spec_from_file_location('fk_ro',fkpath);fk=importlib.util.module_from_spec(sp);sp.loader.exec_module(fk);chain=fk.load_chain()
 for s in sources:
  out=dest/s['parent'];out.mkdir(exist_ok=True)
  if (out/'state.npy').exists():continue
  with h5py.File(s['source'],'r') as h:
   times=h['timestamp'][:s['n']];q=h['observations/qpos'][:s['n']].astype('f8');gr=h['observations/qpos_raw'][:s['n']][:,[6,13]];br=h['state_source_brackets'][:s['n']] if 'state_source_brackets' in h else None
  q[:,[6,13]]=gr>np.array(thresholds);pose=[]
  for sl in (slice(0,6),slice(7,13)):
   mat=fk.fk_link6(q[:,sl],chain);xyz=mat[:,:3,3]+mat[:,:3,:3]@np.array([0,0,.1358]);rot=np.concatenate([mat[:,:3,0],mat[:,:3,1]],1);pose.append(np.concatenate([xyz,rot],1))
  qp=np.concatenate([q,*pose],1).astype('f4');available=np.maximum.accumulate(np.maximum(times,br[:,:,1].max(1))) if br is not None else times
  rows=np.arange(s['n']);ix=np.minimum(rows,np.searchsorted(available,times,side='right')-1);good=(ix>=0)&(times-times[ix.clip(0)]<=.100001)&np.isfinite(qp[ix.clip(0)]).all(1)
  st=np.zeros((s['n'],33),'f4');st[good,:32]=qp[ix[good]];st[:,-1]=good;assert (available[ix[good]]<=times[good]+1e-7).all();np.save(out/'state.npy',st);np.save(out/'times.npy',times)
 guard=Guard('full_frame_siglip2',8)
 try:
  import torch,cv2
  from e86.vision import FrozenEncoder,preprocess,PREPROCESS
  torch.set_num_threads(2);cv2.setNumThreads(1);enc=FrozenEncoder('cuda:0');put('vision_contract.json',PREPROCESS)
  # Re-extract cached train rows before filling new frames. Exact same 96-image batch contract.
  s=next(s for s in sources if s['pool']=='train');z=OLD/'features'/s['parent'];ids=np.load(z/'rows.npy')[:96];old=np.load(z/'raw.npy',mmap_mode='r');audit=[]
  for ci,cam in enumerate(CAMS):
   cap=cv2.VideoCapture(s['videos'][cam]);cur=-1;xx=[];vv=[]
   for f in ids:
    while cur<f:assert cap.grab();cur+=1
    ok,bgr=cap.retrieve();assert ok;x,v,_=preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));xx.append(x);vv.append(v)
   guard.check()
   with torch.inference_mode():pred=enc.images(torch.stack(xx),torch.stack(vv)).cpu().numpy().astype('f2')
   err=float(np.max(abs(pred.astype('f4')-old[:len(ids),ci].astype('f4'))));audit.append(dict(camera=cam,frames=len(ids),raw_max_abs=err,pooled_max_abs=float(abs(pred.astype('f4').mean(1).astype('f2').astype('f4')-old[:len(ids),ci].astype('f4').mean(1).astype('f2').astype('f4')).max())))
   cap.release()
  put('CACHE_REPRODUCTION.json',dict(brt=now(),parent=s['parent'],audit=audit));assert max(a['raw_max_abs'] for a in audit)<.1,audit
  report('## 已有缓存数值复现 '+now()+'\n'+json.dumps(audit,ensure_ascii=False)+'\n仅训练帧，三路各96帧，先通过再补全。')
  for si,s in enumerate(sources):
   out=dest/s['parent']
   if (out/'COMPLETE.json').exists():continue
   olddir=OLD/'features'/s['parent'];ids=np.load(olddir/'rows.npy');raw=np.load(olddir/'raw.npy',mmap_mode='r');n=s['n'];arr=np.lib.format.open_memmap(out/'b.npy',mode='w+',dtype='f2',shape=(n,2304));known=np.zeros(n,bool);known[ids]=True
   for ci,cam in enumerate(CAMS):
    for start in range(0,len(ids),1024):arr[ids[start:start+1024],ci*768:(ci+1)*768]=np.asarray(raw[start:start+1024,ci],dtype='f4').mean(1).astype('f2')
    missing=np.where(~known)[0]
    if not len(missing):continue
    cap=cv2.VideoCapture(s['videos'][cam]);assert cap.isOpened();cur=-1;xx=[];vv=[];ii=[]
    def flush():
     if not ii:return
     guard.check()
     with torch.inference_mode():pp=enc.images(torch.stack(xx),torch.stack(vv)).cpu().numpy().astype('f2').astype('f4').mean(1).astype('f2')
     assert np.isfinite(pp).all();arr[ii,ci*768:(ci+1)*768]=pp;xx.clear();vv.clear();ii.clear()
    for f in missing:
     while cur<f:assert cap.grab(),(s['parent'],cam,int(f));cur+=1
     ok,bgr=cap.retrieve();assert ok;x,v,_=preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));xx.append(x);vv.append(v);ii.append(int(f))
     if len(ii)==96:flush()
    flush();cap.release()
   arr.flush();target=np.load(olddir/'target_rows.npy');baseline=np.load(olddir/'b.npy',mmap_mode='r');diff=float(abs(np.asarray(arr[target],dtype='f4')-baseline.astype('f4')).max());assert diff==0,diff
   (out/'COMPLETE.json').write_text(json.dumps(dict(n=n,baseline_target_max_abs=diff,reused=len(ids),new=n-len(ids))))
   status('features',parent=si+1,total=80,last=s['parent'],elapsed_gpu_seconds=time.monotonic()-guard.started)
  put('FEATURES_COMPLETE.json',dict(brt=now(),parents=80,frames=sum(s['n'] for s in sources),pooled_baseline_max_abs=0));report('## 全量特征完成 '+now()+'\n80 parent全187641帧，三路2304D fp16；所有旧target池化特征最大差0。因果本体32D+有效位，夹爪仅二态；来源右端点不得超过当前时间，未用未来插值。')
 finally:guard.finish()
if __name__=='__main__':run()
