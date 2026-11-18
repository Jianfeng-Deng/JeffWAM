import os,sys,json,threading,subprocess,time,traceback
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode=True
import torch,numpy as np,cv2
from e103d_runtime import *
from e86.vision import FrozenEncoder,preprocess,PREPROCESS,MODEL
CAMS=['cam_high','cam_left_wrist','cam_right_wrist']; TOPICS=['/top-camera','/left-wrist-camera','/right-wrist-camera']
def encode_video(path,rows,encoder,guard,write):
 cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);assert cap.isOpened();j=-1;pixels=[];valid=[];ix=[]
 try:
  for i,r in enumerate(rows):
   while j<r:
    assert cap.grab(),(path,r,j);j+=1
   ok,bgr=cap.retrieve();assert ok;x,v,_=preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));pixels.append(x);valid.append(v);ix.append(i)
   if len(ix)==128:
    guard.check();write(ix,encoder.images(torch.stack(pixels),torch.stack(valid)).cpu().numpy().astype('f2'));pixels=[];valid=[];ix=[]
  if ix:guard.check();write(ix,encoder.images(torch.stack(pixels),torch.stack(valid)).cpu().numpy().astype('f2'))
 finally:cap.release()
def replay(enc,g):
 ready=json.loads((T/'code_e86/artifacts/E86-V3-20261113T144520/READY_V3_cache_windows.json').read_text());cache=Path(ready['cache']);records=[];parents=set()
 roots=[cache,Path(json.loads((cache/'overlay.json').read_text())['base_cache'])]
 for rp in [p for root in roots for p in sorted(root.glob('piper_*/receipt.json'))]:
  r=json.loads(rp.read_text())
  if r['split']!='train' or r['parent_component'] in parents or r['canonical_id']=='piper_4dc107335389f5711a0f65e0':continue
  paths=[Path(c['source_path'].replace('/mnt/workspace/colleague-a/Datasets/piper/','/mnt/workspace/dataset/piper/')) for c in r['cameras']]
  if not all(p.exists() for p in paths):continue
  assert all(p.stat().st_size==c['source_stat'][0] for p,c in zip(paths,r['cameras']))
  parents.add(r['parent_component'])
  for c in r['cameras']:
   rows=np.load(rp.parent/(c['camera']+'_rows.npy'))[:128];expected=np.load(rp.parent/(c['camera']+'.npy'),mmap_mode='r')[:len(rows)];out=np.empty_like(expected)
   source=str(Path(c['source_path'].replace('/mnt/workspace/colleague-a/Datasets/piper/','/mnt/workspace/dataset/piper/')))
   encode_video(source,rows,enc,g,lambda ix,v:out.__setitem__(ix,v));err=float(abs(out.astype('f4')-expected.astype('f4')).max());records.append(dict(parent=r['parent_component'],cid=r['canonical_id'],camera=c['camera'],frames=len(rows),max_abs=err,receipt_sha256=sha(rp),read_path=source,source_sha256=sha(source)))
  if len(parents)==2:break
 result=dict(parents=len(parents),records=records,max_abs=max(r['max_abs'] for r in records),preprocess=PREPROCESS);atomic(CONTROL/'piper_replay.json',result);report('## 两个 Piper parent 数值复现\n```json\n'+json.dumps(result,ensure_ascii=False)+'\n```');assert result['parents']==2 and result['max_abs']==0,'Cache replay mismatch: stop before expansion'
def encode_mcap(rec,k,encoder,guard,out):
 sys.path.append(str(T/'data/abc_inventory_v1/env/venv/lib/python3.12/site-packages'))
 from mcap.reader import make_reader
 from mcap_protobuf.decoder import DecoderFactory
 packets=[];ts=[]
 with open(rec['original']['path'],'rb') as f:
  r=make_reader(f,decoder_factories=[DecoderFactory()])
  for sc,ch,m,d in r.iter_decoded_messages(topics=[TOPICS[k]]):
   assert d.format=='h264';assert m.log_time==d.timestamp.seconds*10**9+d.timestamp.nanos;packets.append(d.data);ts.append(m.log_time)
 assert ts==rec['timestamps'][k];raw=b''.join(packets);del packets
 p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-f','h264','-i','pipe:0','-an','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 def feed():
  try:p.stdin.write(raw);p.stdin.close()
  except BrokenPipeError:pass
 th=threading.Thread(target=feed);th.start();rows=rec['rows'][k];want={r:i for i,r in enumerate(rows)};pixels=[];valid=[];ix=[];idx=0
 try:
  while True:
   b=p.stdout.read(640*480*3)
   if not b:break
   assert len(b)==640*480*3
   if idx in want:
    x,v,_=preprocess(np.frombuffer(b,np.uint8).reshape(480,640,3));pixels.append(x);valid.append(v);ix.append(want[idx])
    if len(ix)==128:
     guard.check();out[ix]=encoder.images(torch.stack(pixels),torch.stack(valid)).cpu().numpy().astype('f2');pixels=[];valid=[];ix=[]
   idx+=1
  th.join();err=p.stderr.read();assert p.wait()==0,err.decode()[:200];assert idx==len(ts),(idx,len(ts))
  if ix:guard.check();out[ix]=encoder.images(torch.stack(pixels),torch.stack(valid)).cpu().numpy().astype('f2')
 finally:
  if p.poll() is None:p.kill();p.wait()
  th.join(3)
def main():
 torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;guard=Guard('cache_02',19000,6000)
 try:
  enc=FrozenEncoder('cuda:0');replay(enc,guard)
  np.save(CACHE/'generic_text.npy',enc.text('Manipulate objects on a tabletop.').cpu().numpy().astype('f2'))
  if '--replay-only' in sys.argv:return
  plan=json.loads((CACHE/'plan.json').read_text());assert plan['summary']['estimated_cache_gpuh_218fps_margin25']<4.8
  dest=CACHE/'features';dest.mkdir(exist_ok=False);atomic(dest/'contract.json',dict(preprocess=PREPROCESS,encoder_files={str(p):sha(p) for p in sorted(Path(MODEL).glob('*')) if p.suffix in ['.safetensors','.json','.model']},plan_sha256=sha(CACHE/'plan.json'),camera_map=dict(zip(CAMS,TOPICS)),shape=[8,768],dtype='float16'))
  completed=[];total=0;start=time.monotonic()
  for i,r in enumerate(plan['records']):
   guard.check();cid=r['canonical_id'];d=dest/cid;d.mkdir(exist_ok=False);o=r['original'];paths=[o['path']] if r['source']=='abc' else [v['path'] for v in o['videos']];before={p:(os.stat(p).st_size,os.stat(p).st_mtime_ns) for p in paths}
   if r['source']=='abc':assert before[o['path']]==(o['size_bytes'],o['mtime_ns'])
   else:
    for v in o['videos']:
     assert before[v['path']]==(v['bytes'],v['mtime_ns'])
     with open(v['path'],'rb') as f:assert __import__('hashlib').sha256(f.read(65536)).hexdigest()==v['sha256_first_65536_bytes']
    for cam in ['top','left','right']:
     z=o['small_files_sha256'][cam+'-timestamp.npy'];assert sha(z['path'])==z['sha256']
   for k,cam in enumerate(CAMS):
    rows=r['rows'][k];out=np.lib.format.open_memmap(d/(cam+'.npy'),mode='w+',dtype='f2',shape=(len(rows),8,768));np.save(d/(cam+'_rows.npy'),np.array(rows,dtype='i8'))
    if r['source']=='abc':encode_mcap(r,k,enc,guard,out)
    else:
     p=next(v['path'] for v in o['videos'] if v['camera']==['top','left','right'][k]);encode_video(p,rows,enc,guard,lambda ix,v:out.__setitem__(ix,v))
    out.flush();del out;total+=len(rows)
   assert before=={p:(os.stat(p).st_size,os.stat(p).st_mtime_ns) for p in paths},'Source changed'
   atomic(d/'receipt.json',dict(cid=cid,source=r['source'],source_stat=before,files={p.name:sha(p) for p in d.glob('*.npy')}));completed.append(cid)
   atomic(CONTROL/'cache_status.json',dict(status='running',completed=len(completed),total=len(plan['records']),camera_frames=total,elapsed_s=guard.check(),throughput=total/(time.monotonic()-start),brt=brt()))
   if i%10==0:print('cached',len(completed),len(plan['records']),total,flush=True)
   if total>15000:
    projected=(plan['summary']['camera_frames']-total)/(total/(time.monotonic()-start));assert projected+used_seconds()+1200<21600,'Projected total GPU budget exceeds cap'
  result=dict(status='complete',completed=len(completed),camera_frames=total,elapsed_s=guard.check(),cache=str(dest),brt=brt());atomic(CONTROL/'cache_status.json',result);report('## 缓存提取完成\n'+json.dumps(result,ensure_ascii=False))
 finally:guard.finish()
if __name__=='__main__':
 try:main()
 except BaseException:atomic(CONTROL/'cache_error02.json',dict(error=traceback.format_exc(),brt=brt()));raise
