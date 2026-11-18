import os,sys,time,json,queue,threading,subprocess,re,traceback
sys.dont_write_bytecode=True
import numpy as np,torch,cv2
from concurrent.futures import ThreadPoolExecutor
from e103d_runtime import *
from e86.vision import FrozenEncoder,PREPROCESS,MODEL,preprocess
from e103d_cache import replay,encode_video
CAMS=['cam_high','cam_left_wrist','cam_right_wrist'];TOPICS=['/top-camera','/left-wrist-camera','/right-wrist-camera']
def packets(r):
 sys.path.append(str(T/'data/abc_inventory_v1/env/venv/lib/python3.12/site-packages'))
 from mcap.reader import make_reader
 from mcap_protobuf.decoder import DecoderFactory
 ps=[[] for _ in range(3)];ts=[[] for _ in range(3)];ix={c:k for k,c in enumerate(TOPICS)}
 with open(r['original']['path'],'rb') as f:
  for sc,ch,m,d in make_reader(f,decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=TOPICS):
   k=ix[ch.topic];assert d.format=='h264';assert m.log_time==d.timestamp.seconds*10**9+d.timestamp.nanos;ps[k].append(d.data);ts[k].append(m.log_time)
 assert ts==r['timestamps'];return [b''.join(p) for p in ps]
def encode_cameras(r,raws,enc,guard,arrays):
 q=queue.Queue(5);stop=threading.Event()
 def put(x):
  while not stop.is_set():
   try:q.put(x,timeout=.2);return
   except queue.Full:pass
 def worker(k):
  p=None
  try:
   p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','info','-threads','1','-filter_threads','1','-f','h264','-i','pipe:0','-vf','showinfo=checksum=0','-an','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   errs=[];bframes=[0];show=[0]
   def feed():
    try:p.stdin.write(raws[k]);p.stdin.close()
    except BrokenPipeError:pass
   def stderr():
    for line in iter(p.stderr.readline,b''):
     if b'showinfo' in line and b' n:' in line:
      show[0]+=1
      if re.search(rb'\btype:\s*B\b',line):bframes[0]+=1
     elif len(errs)<15:errs.append(line.decode(errors='replace').strip())
   inp=threading.Thread(target=feed);er=threading.Thread(target=stderr);inp.start();er.start();want={v:i for i,v in enumerate(r['rows'][k])};xs=[];vs=[];ids=[];frame=0
   while not stop.is_set():
    data=p.stdout.read(640*480*3)
    if not data:break
    assert len(data)==640*480*3
    if frame in want:
     x,v,_=preprocess(np.frombuffer(data,np.uint8).reshape(480,640,3));xs.append(x);vs.append(v);ids.append(want[frame])
     if len(ids)==128:put(('batch',k,ids,torch.stack(xs),torch.stack(vs)));xs=[];vs=[];ids=[]
    frame+=1
   if stop.is_set():return
   inp.join();assert p.wait()==0,errs;er.join();assert frame==len(r['timestamps'][k]),(frame,len(r['timestamps'][k]));assert show[0]==frame,(show[0],frame);assert bframes[0]==0,'B frames make packet/frame exposure mapping unsafe'
   if ids:put(('batch',k,ids,torch.stack(xs),torch.stack(vs)))
   put(('done',k,dict(decoded_frames=frame,b_frames=bframes[0],showinfo_frames=show[0])))
  except BaseException as e:put(('error',k,repr(e)))
  finally:
   if p and p.poll() is None:p.kill();p.wait()
 with ThreadPoolExecutor(max_workers=3) as pool:
  futs=[pool.submit(worker,k) for k in range(3)];done=0;stats={};last=0
  try:
   while done<3:
    if time.monotonic()-last>1:guard.check();last=time.monotonic()
    try:item=q.get(timeout=1)
    except queue.Empty:continue
    kind,k=item[:2]
    if kind=='error':raise RuntimeError(item[2])
    if kind=='done':stats[CAMS[k]]=item[2];done+=1;continue
    _,k,ids,x,v=item;arrays[k][ids]=enc.images(x,v).cpu().numpy().astype('f2')
  finally:stop.set()
 return stats

def main():
 torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;guard=Guard('cache_03',15500,6000)
 try:
  plan=json.loads((CACHE/'plan.json').read_text());assert json.loads((CONTROL/'piper_replay.json').read_text())['max_abs']==0;assert json.loads((CONTROL/'DATA_AUDIT.json').read_text())['status']=='passed';enc=FrozenEncoder('cuda:0');dest=CACHE/'features';assert dest.exists();total=0;fresh=0;completed=[];start=time.monotonic()
  for i,r in enumerate(plan['records']):
   cid=r['canonical_id'];d=dest/cid
   if (d/'receipt.json').exists():
    rc=json.loads((d/'receipt.json').read_text());assert all(sha(d/f)==v for f,v in rc['files'].items());total+=sum(len(a) for a in r['rows']);completed.append(cid);continue
   guard.check()
   if d.exists():
    quarantine=CACHE/('incomplete_cache02_'+cid);assert not quarantine.exists();d.rename(quarantine)
   d.mkdir(exist_ok=False);o=r['original'];paths=[o['path']] if r['source']=='abc' else [v['path'] for v in o['videos']];before={p:(os.stat(p).st_size,os.stat(p).st_mtime_ns) for p in paths};arrays=[]
   if r['source']=='abc':assert before[o['path']]==(o['size_bytes'],o['mtime_ns'])
   else:
    for v in o['videos']:
     assert before[v['path']]==(v['bytes'],v['mtime_ns'])
     with open(v['path'],'rb') as f:assert __import__('hashlib').sha256(f.read(65536)).hexdigest()==v['sha256_first_65536_bytes']
   for k,cam in enumerate(CAMS):
    rows=r['rows'][k];arrays.append(np.lib.format.open_memmap(d/(cam+'.npy'),mode='w+',dtype='f2',shape=(len(rows),8,768)));np.save(d/(cam+'_rows.npy'),np.array(rows,dtype='i8'))
   if r['source']=='abc':stats=encode_cameras(r,packets(r),enc,guard,arrays)
   else:
    stats={}
    for k,cam in enumerate(CAMS):
     p=next(v['path'] for v in o['videos'] if v['camera']==['top','left','right'][k]);encode_video(p,r['rows'][k],enc,guard,lambda ids,v:arrays[k].__setitem__(ids,v))
   for a in arrays:a.flush()
   del arrays;assert before=={p:(os.stat(p).st_size,os.stat(p).st_mtime_ns) for p in paths};atomic(d/'receipt.json',dict(cid=cid,source=r['source'],source_stat=before,decoder_stats=stats,files={p.name:sha(p) for p in d.glob('*.npy')}));count=sum(len(a) for a in r['rows']);total+=count;fresh+=count;completed.append(cid);rate=fresh/max(1,time.monotonic()-start);elapsed=guard.check()
   atomic(CONTROL/'cache_status.json',dict(status='running',phase='cache_03',completed=len(completed),total=len(plan['records']),camera_frames=total,elapsed_s=elapsed,throughput=rate,brt=brt()))
   if i%10==0:print('cached',len(completed),len(plan['records']),total,round(rate,1),flush=True)
   if fresh>15000:assert (plan['summary']['camera_frames']-total)/rate+used_seconds()+5400<21600,'Projected cache+training+evaluation exceed budget'
  result=dict(status='complete',completed=len(completed),camera_frames=total,fresh_camera_frames=fresh,elapsed_s=guard.check(),cache=str(dest),brt=brt());atomic(CONTROL/'cache_status.json',result);report('## 缓存提取完成\n'+json.dumps(result,ensure_ascii=False))
 finally:guard.finish()
if __name__=='__main__':
 try:main()
 except BaseException:atomic(CONTROL/'cache_error03.json',dict(error=traceback.format_exc(),brt=brt()));raise
