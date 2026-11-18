from .common import *
from .resource import Guard
import traceback,queue,concurrent.futures,threading
G=None;executor=None;stop=threading.Event()
try:
 plan=json.loads((R/'cache_plan.json').read_text());assert (R/'two_parent_selfcheck.json').exists();assert json.loads((R/'two_parent_selfcheck.json').read_text())['passed']
 G=Guard('DINOv3_parallel_CPU_decode_single_GPU_encode');import torch,numpy as np,cv2
 from .vision import Encoder,preprocess,pool
 torch.set_num_threads(2);torch.manual_seed(90);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;cv2.setNumThreads(1);model=Encoder();q=queue.Queue(maxsize=12)
 contractsha=sha(CACHE/'contract.json');tasks=[];complete={};receipts={};count=0;frames_done=0;timing=dict(gpu=0.,write=0.,wait=0.)
 for c in plan['captures']:
  dest=CACHE/c['canonical_id']
  if (dest/'receipt.json').exists():
   assert json.loads((dest/'receipt.json').read_text())['contract_sha256']==contractsha;count+=1;continue
  dest.mkdir(exist_ok=False);complete[c['canonical_id']]=0;receipts[c['canonical_id']]=[]
  for cam in c['cameras']:tasks.append((c,cam,dest))
 def put(x):
  while not stop.is_set():
   try:q.put(x,timeout=1);return
   except queue.Full:pass
 def decode(task):
  c,cam,dest=task;v=None
  try:
   name=cam['camera'];rr=np.array(cam['rows'],dtype='i8');fr=np.array(cam['fine_rows'],dtype='i8');np.save(dest/(name+'_rows.npy'),rr);np.save(dest/(name+'_7x7_rows.npy'),fr)
   arr=np.lib.format.open_memmap(dest/(name+'.npy'),mode='w+',dtype='f2',shape=(len(rr),8,768));fine=np.lib.format.open_memmap(dest/(name+'_7x7.npy'),mode='w+',dtype='f2',shape=(len(fr),49,768));fmap={int(r):i for i,r in enumerate(fr)}
   before=Path(cam['path']).stat();v=cv2.VideoCapture(cam['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);assert v.isOpened();cursor=-1;px=[];ms=[];ii=[];hw=None;aff=None
   def flush():
    if not px:return
    put(('batch',arr,fine,fmap,rr,list(ii),torch.stack(px),torch.stack(ms)));px.clear();ms.clear();ii.clear()
   for i,row in enumerate(rr):
    if stop.is_set():return
    while cursor<row:assert v.grab(),(cam['path'],int(row));cursor+=1
    ok,bgr=v.retrieve();assert ok
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);x,m,a=preprocess(rgb);hw=list(rgb.shape[:2]);aff=a.tolist();px.append(x);ms.append(m);ii.append(i)
    if len(px)>=128:flush()
   flush();v.release();v=None;assert before.st_mtime_ns==Path(cam['path']).stat().st_mtime_ns
   put(('end',c,arr,fine,dict(camera=name,frames=len(rr),fine_frames=len(fr),original_hw=hw,affine=aff,source_path=cam['path'],source_unchanged=True)))
  except BaseException as e:put(('error',repr(e),traceback.format_exc()))
  finally:
   if v is not None:v.release()
 executor=concurrent.futures.ThreadPoolExecutor(max_workers=6);futures=[executor.submit(decode,t) for t in tasks];ends=0
 while ends<len(tasks):
  G.check();t=time.monotonic()
  try:item=q.get(timeout=2)
  except queue.Empty:continue
  timing['wait']+=time.monotonic()-t
  if item[0]=='error':raise RuntimeError(item[1]+'\n'+item[2])
  if item[0]=='batch':
   _,arr,fine,fmap,rr,ii,x,m=item;t=time.monotonic();m=m.cuda();p=model.patches(x);z=pool(p,m,(2,4)).cpu().numpy().astype('f2');assert np.isfinite(z).all();need=[j for j,idx in enumerate(ii) if int(rr[idx]) in fmap];z7=pool(p[need],m[need],(7,7)).cpu().numpy().astype('f2') if need else None;timing['gpu']+=time.monotonic()-t
   t=time.monotonic();arr[ii]=z
   if need:assert np.isfinite(z7).all();fine[[fmap[int(rr[ii[j]])] for j in need]]=z7
   timing['write']+=time.monotonic()-t;frames_done+=len(ii)
  else:
   _,c,arr,fine,record=item;arr.flush();fine.flush();ends+=1;cid=c['canonical_id'];receipts[cid].append(record);complete[cid]+=1
   if complete[cid]==3:
    records=sorted(receipts[cid],key=lambda r:('cam_high','cam_left_wrist','cam_right_wrist').index(r['camera']));atomic(CACHE/cid/'receipt.json',dict(canonical_id=cid,split=c['split'],parent_component=c['parent_component'],contract_sha256=contractsha,cameras=records));count+=1
    status('extracting_parallel',done=count,total=len(plan['captures']),gpu_seconds=G.check(),segment_frames=frames_done,timing=timing);print(json.dumps(dict(done=count,total=len(plan['captures']),GPU_s=round(G.check()),segment_frames=frames_done,timing={k:round(v,2) for k,v in timing.items()})),flush=True)
 executor.shutdown(wait=True);executor=None
 atomic(R/'extract_complete.json',dict(brt=brt(),captures=len(plan['captures']),train_parents=242,dev_parents=27,frames=plan['train_frames'],fine_frames=plan['detail_frames'],cache_bytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file()),parallel_timing=timing));status('extraction_complete');report('## 1 特征提取完成\n'+(R/'extract_complete.json').read_text())
except BaseException as e:status('stopped_extract_parallel',error=str(e),traceback=traceback.format_exc());raise
finally:
 stop.set()
 if executor:executor.shutdown(wait=True,cancel_futures=True)
 if G:G.finish()
