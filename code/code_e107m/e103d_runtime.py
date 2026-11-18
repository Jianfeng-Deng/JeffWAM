import os,time,json,threading,subprocess,datetime,hashlib
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu'); ROOT=T/'code_e107m'; CONTROL=T/'results/e107m_control_20261116_01'; CACHE=T/'data/e103d_cache'; REPORT=T/'notes/e107m_multimodal_head_20261116.md'
def brt():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(p,x):
 p=Path(p);tmp=p.with_name(p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));os.replace(tmp,p)
def report(s):
 with REPORT.open('a') as f:f.write('\n'+s+'\n')
def gpu_rows():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return [dict(index=int(a),uuid=b.strip(),total=float(c),free=float(d),util=float(e)) for a,b,c,d,e in (l.split(',') for l in s.strip().splitlines())]
def used_seconds():
 return sum(json.loads(p.read_text()).get('elapsed_s',0) for p in CONTROL.glob('gpu_*.json'))
class Guard:
 def __init__(self,phase,limit=7200,peak=6000):
  self.start=time.monotonic();self.peak=peak;self.phase=phase;self.stop=threading.Event();self.mutex=threading.RLock();self.prior=used_seconds();self.path=CONTROL/f'gpu_{phase}.json';assert not self.path.exists();self.row=next(r for r in gpu_rows() if r['uuid']==os.environ['CUDA_VISIBLE_DEVICES']);assert self.row['free']>=peak*1.5;self.limit=min(limit,7200-self.prior);self.meta=dict(phase=phase,pid=os.getpid(),gpu=self.row,started_brt=brt(),status='active',elapsed_s=0.,peak_estimate_mib=peak);atomic(self.path,self.meta);self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
 def monitor(self):
  while not self.stop.wait(2):
   try:
    r=next(r for r in gpu_rows() if r['uuid']==self.row['uuid']);assert r['free']>=self.peak*1.5,'Shared memory pressure'
    self.check()
   except BaseException as e:
    self.meta.update(status='stopped_guard',error=str(e),elapsed_s=time.monotonic()-self.start);atomic(self.path,self.meta);os._exit(88)
 def check(self):
  import torch
  elapsed=time.monotonic()-self.start;assert sum(p.stat().st_size for root in (ROOT,CONTROL,T/'results/e107m_m0_s90_20261116_01',T/'results/e107m_m1_s90_20261116_01') for p in root.rglob('*') if p.is_file())<9_900_000_000,'Disk cap';assert elapsed<self.limit,'GPU budget exhausted';assert not torch.cuda.is_initialized() or torch.cuda.max_memory_allocated()/2**20<=self.peak,'Peak estimate exceeded'
  with self.mutex:self.meta.update(elapsed_s=elapsed);atomic(self.path,self.meta)
  return elapsed
 def finish(self):
  self.stop.set();self.thread.join(5)
  with self.mutex:self.meta.update(status='closed',elapsed_s=time.monotonic()-self.start+30,finished_brt=brt());atomic(self.path,self.meta)
def wait_gpu(peak=6000):
 while True:
  rows=sorted(gpu_rows(),key=lambda r:r['free'],reverse=True)
  if rows[0]['free']>=peak*1.5:return rows[0]['uuid']
  atomic(CONTROL/'queue_status.json',dict(status='waiting_memory',peak=peak,brt=brt(),gpus=rows));time.sleep(20)
