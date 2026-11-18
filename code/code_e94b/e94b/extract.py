from .common import *
import traceback
from .resource import Guard
G=None
try:
 plan=json.loads((R/'cache_plan.json').read_text());assert plan['cumulative_write_estimate']<=30e9
 G=Guard('DINOv3_extract_including_selfcheck');import torch,numpy as np,cv2
 from .vision import Encoder,preprocess,pool,DINO,WEIGHT,MEAN,STD
 from e86.vision import PREPROCESS
 torch.set_num_threads(2);torch.manual_seed(90);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;cv2.setNumThreads(1)
 model=Encoder();checks=[]
 # Exactly two parents before bulk writes. Independent official intermediate path.
 for c in plan['captures'][:2]:
  cam=c['cameras'][0];v=cv2.VideoCapture(cam['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);assert v.isOpened()
  for j in range(cam['rows'][0]+1):assert v.grab()
  ok,bgr=v.retrieve();v.release();assert ok
  x,m,a=preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));x=x[None];m=m[None].cuda();G.check()
  p=model.patches(x);q=model.patches(x)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):official=model.model.get_intermediate_layers(x.cuda(),n=1,norm=True)[0]
  diff=float((p-q).abs().max());odiff=float((p-official).abs().max());assert p.shape==(1,196,768) and torch.isfinite(p).all() and diff==0 and odiff==0
  z=pool(p,m,(2,4));f=pool(p,m,(7,7));assert torch.isfinite(z).all() and torch.isfinite(f).all()
  checks.append(dict(parent=c['parent_component'],canonical_id=c['canonical_id'],frame=cam['rows'][0],patch_shape=list(p.shape),repeat_max_abs=diff,official_intermediate_max_abs=odiff,std=float(p.float().std())))
 atomic(R/'two_parent_selfcheck.json',dict(passed=True,checks=checks));report('## 1 两parent加载/数值自检通过\n'+json.dumps(checks,ensure_ascii=False))
 contract=dict(schema='e94b-dinov3-v1',encoder='DINOv3 ViT-B/16 LVD1689M',weights_path=str(WEIGHT),weights_sha256=sha(WEIGHT),source_path=str(DINO),source_sha256={str(p.relative_to(DINO)):sha(p) for p in sorted((DINO/'dinov3').rglob('*.py'))},license='Meta DINOv3 custom license; internal research only',official_source='https://github.com/facebookresearch/dinov3',preprocess={**PREPROCESS,'version':'e94b_dinov3_e84_fov_v1','mean':MEAN,'std':STD},token='official forward_features x_norm_patchtokens, no CLS/registers',pooling='same valid-area weighted adaptive average as E86 2x4;7x7 same formula;h100 mean of each camera 2x4',precision='official float16 autocast;pool float32;cache float16',frame_plan_sha256=sha(R/'cache_plan.json'),target_only=True,observation='immutable SigLIP2 image+text',refs=plan['refs'])
 assert not (CACHE/'contract.json').exists();atomic(CACHE/'contract.json',contract)
 for ci,c in enumerate(plan['captures']):
  dest=CACHE/c['canonical_id'];dest.mkdir(exist_ok=False);records=[]
  for cam in c['cameras']:
   G.check();name=cam['camera'];rows=np.asarray(cam['rows'],dtype='i8');fr=np.asarray(cam['fine_rows'],dtype='i8');np.save(dest/(name+'_rows.npy'),rows);np.save(dest/(name+'_7x7_rows.npy'),fr)
   arr=np.lib.format.open_memmap(dest/(name+'.npy'),mode='w+',dtype='f2',shape=(len(rows),8,768));fine=np.lib.format.open_memmap(dest/(name+'_7x7.npy'),mode='w+',dtype='f2',shape=(len(fr),49,768));fmap={int(r):i for i,r in enumerate(fr)}
   before=Path(cam['path']).stat();v=cv2.VideoCapture(cam['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);assert v.isOpened();cursor=-1;px=[];ms=[];ii=[];hw=None;aff=None
   def flush():
    if not px:return
    G.check();x=torch.stack(px);m=torch.stack(ms).cuda();p=model.patches(x);z=pool(p,m,(2,4)).cpu().numpy().astype('f2');assert np.isfinite(z).all();arr[ii]=z
    need=[j for j,idx in enumerate(ii) if int(rows[idx]) in fmap]
    if need:
     z7=pool(p[need],m[need],(7,7)).cpu().numpy().astype('f2');assert np.isfinite(z7).all();fine[[fmap[int(rows[ii[j]])] for j in need]]=z7
    px.clear();ms.clear();ii.clear()
   for i,row in enumerate(rows):
    while cursor<row:assert v.grab(),(cam['path'],int(row));cursor+=1
    ok,bgr=v.retrieve();assert ok
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);x,m,a=preprocess(rgb);hw=list(rgb.shape[:2]);aff=a.tolist();px.append(x);ms.append(m);ii.append(i)
    if len(px)>=64:flush()
   flush();v.release();arr.flush();fine.flush();del arr,fine
   assert before.st_mtime_ns==Path(cam['path']).stat().st_mtime_ns
   records.append(dict(camera=name,frames=len(rows),fine_frames=len(fr),original_hw=hw,affine=aff,source_path=cam['path'],source_unchanged=True))
  atomic(dest/'receipt.json',dict(canonical_id=c['canonical_id'],split=c['split'],parent_component=c['parent_component'],contract_sha256=sha(CACHE/'contract.json'),cameras=records))
  status('extracting',done=ci+1,total=len(plan['captures']),gpu_seconds=G.check());print(json.dumps(dict(done=ci+1,total=len(plan['captures']),GPU_s=round(G.check()))),flush=True)
 atomic(R/'extract_complete.json',dict(brt=brt(),parents=len(plan['captures']),frames=plan['train_frames'],fine_frames=plan['detail_frames'],cache_bytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())))
 status('extraction_complete');report('## 1 特征提取完成\n'+(R/'extract_complete.json').read_text())
except BaseException as e:status('stopped_extract',error=str(e),traceback=traceback.format_exc());raise
finally:
 if G:G.finish()
