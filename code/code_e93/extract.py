import os,sys,time,json,subprocess,traceback
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e93_dev_diag_20261114_01';sys.path.insert(0,str(T/'code_e90'));os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from e86.common import atomic,sha,brt
started=time.monotonic();ledger=R/'gpu_ledger_extract.json';device=None
try:
 cards=[list(map(int,l.split(','))) for l in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()];device,free=max(cards,key=lambda z:z[1]);assert free>=12288
 os.environ['CUDA_VISIBLE_DEVICES']=str(device)
 import torch,numpy as np,cv2
 from e86.vision import FrozenEncoder,PREPROCESS,MODEL,preprocess
 from e86.data import CAMS
 torch.set_num_threads(2);cv2.setNumThreads(1)
 def guard():
  elapsed=time.monotonic()-started;peak=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0
  f=int(subprocess.check_output(['nvidia-smi','-i',str(device),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  atomic(ledger,dict(status='active',seconds=elapsed,GPUh=elapsed/3600,gpu=device,peak_mib=peak,free_mib=f,brt=brt()))
  assert elapsed<6600 and peak<=8192 and f>=max(4096,peak*.5),'GPU budget or shared headroom guard'
 encoder=FrozenEncoder('cuda:0');old=T/'code_e86/artifacts/dev_features_v1';contract=json.loads((old/'contract.json').read_text());assert contract['preprocess']==PREPROCESS
 assert all(sha(p)==h for p,h in contract['encoder_files'].items())
 def encode_cam(cam,callback):
  path=cam['path'];s=Path(path).stat();v=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);assert v.isOpened(),path
  ids=cam['rows'];cursor=-1;px=[];valid=[];indices=[];hw=None;affine=None
  def flush():
   if not px:return
   guard();f=encoder.images(torch.stack(px),torch.stack(valid)).cpu().numpy().astype('f2');callback(indices,f)
   px.clear();valid.clear();indices.clear()
  for i,rr in enumerate(ids):
   while cursor<rr:
    assert v.grab(),(path,rr);cursor+=1
   ok,bgr=v.retrieve();assert ok
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);x,m,af=preprocess(rgb);hw=list(rgb.shape[:2]);affine=af.tolist();px.append(x);valid.append(m);indices.append(i)
   if len(px)>=64:flush()
  flush();v.release();assert Path(path).stat().st_mtime_ns==s.st_mtime_ns
  return dict(camera=cam['camera'],original_hw=hw,affine=affine,source_path=path,source_unchanged=True,encoded_frames=len(ids))
 # Replay every original cached row on all six dev parents before any new extraction.
 oldplan=json.loads((T/'code_e86/artifacts/dev_v1/cache_plan.json').read_text());checks=[]
 for c in oldplan['captures']:
  for cam in c['cameras']:
   rows=np.load(old/c['canonical_id']/(cam['camera']+'_rows.npy'));arr=np.load(old/c['canonical_id']/(cam['camera']+'.npy'),mmap_mode='r');assert np.array_equal(rows,cam['rows']);errors=[]
   def compare(ii,ff):errors.append(float(np.max(np.abs(ff.astype('f4')-arr[ii].astype('f4')))))
   encode_cam(cam,compare);checks.append(dict(canonical_id=c['canonical_id'],camera=cam['camera'],frames=len(rows),max_abs_difference=max(errors)))
 maxerr=max(x['max_abs_difference'] for x in checks);atomic(R/'cache_replay.json',dict(parents=6,frames=sum(x['frames'] for x in checks),max_abs_difference=maxerr,exact=maxerr==0,checks=checks))
 with (T/'notes/e93_dev_diag_20261114.md').open('a') as f:f.write(f'\n### A 原6parent缓存重放\n{sum(x["frames"] for x in checks)}相机帧，fp16缓存最大绝对差={maxerr}；合同/本地权重SHA核验完成。\n')
 assert maxerr==0,'Reproduction mismatch: stop expansion and investigate, no tolerance-based continuation'
 plan=json.loads((R/'cache_plan.json').read_text());assert plan['estimated_bytes']<=30e9
 dest=R/'features';dest.mkdir(exist_ok=False);contract['plan_sha256']=sha(R/'cache_plan.json');atomic(dest/'contract.json',contract)
 for ci,c in enumerate(plan['captures']):
  out=dest/c['canonical_id'];out.mkdir(exist_ok=False);recs=[]
  for cam in c['cameras']:
   guard();ids=np.array(cam['rows'],dtype='i8');arr=np.lib.format.open_memmap(out/(cam['camera']+'.npy'),mode='w+',dtype='f2',shape=(len(ids),8,768));np.save(out/(cam['camera']+'_rows.npy'),ids)
   def save(ii,ff):arr[ii]=ff
   recs.append(encode_cam(cam,save));arr.flush();del arr
  atomic(out/'receipt.json',dict(canonical_id=c['canonical_id'],parent_component=c['parent_component'],split='dev',contract_sha256=sha(dest/'contract.json'),cameras=recs))
  atomic(R/'extract_status.json',dict(status='extracting',done=ci+1,total=len(plan['captures']),seconds=time.monotonic()-started));print(ci+1,len(plan['captures']),flush=True)
 atomic(R/'extract_status.json',dict(status='completed',captures=len(plan['captures']),seconds=time.monotonic()-started))
except BaseException as e:
 atomic(R/'extract_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
finally:
 atomic(ledger,dict(status='finished',seconds=time.monotonic()-started,GPUh=(time.monotonic()-started)/3600,gpu=device,brt=brt()))
