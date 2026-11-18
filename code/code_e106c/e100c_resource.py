import os,time,json,fcntl,threading
import torch
from e99w2_common import CONTROL,ART,atomic,brt
from e86.resource import gpu_rows
PEAK_MIB=6144
class Guard:
 def __init__(self,phase,limit_seconds=1450):
  self.mutex=threading.RLock();self.started=time.monotonic();self.limit=limit_seconds;self.closed=False;self.error=None
  self.gpu=next(r for r in gpu_rows() if r['uuid']==os.environ['CUDA_VISIBLE_DEVICES'])
  assert self.gpu['total']-self.gpu['used']>=1.5*PEAK_MIB,'GPU admission failed'
  # Scheduler reserves full per-job budget under a session-wide lock.
  self.path=ART/f'gpu_{phase}.json';assert not self.path.exists()
  self.meta=dict(phase=phase,pid=os.getpid(),gpu=self.gpu,brt=brt(),status='active',elapsed_s=0.,peak_estimate_mib=PEAK_MIB);atomic(self.path,self.meta)
  self.stop=threading.Event();self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
 def monitor(self):
  while not self.stop.wait(2):
   try:
    r=next(r for r in gpu_rows() if r['uuid']==self.gpu['uuid'])
    if r['total']-r['used']<1.5*PEAK_MIB:self.error='Shared memory pressure'
    self.check()
   except BaseException as e:self.error=str(e)
   if self.error:
    import signal
    os.kill(os.getpid(),signal.SIGTERM)
    return
 def check(self):
  if self.error:raise RuntimeError(self.error)
  elapsed=time.monotonic()-self.started
  roots=[CONTROL.parent.parent/'code_e102a',*CONTROL.parent.glob('e102a_*')]
  total=sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
  assert total<=10*1024**3,'Disk budget exceeded'
  assert elapsed<self.limit,'GPU segment budget exhausted'
  if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()/2**20<=PEAK_MIB,'Peak exceeds estimate'
  with self.mutex:
   self.meta.update(elapsed_s=elapsed);atomic(self.path,self.meta)
  return elapsed
 def finish(self):
  if self.closed:return
  self.stop.set();self.thread.join(timeout=5);self.meta.update(elapsed_s=time.monotonic()-self.started,status='closed',error=self.error,finished_brt=brt());atomic(self.path,self.meta);self.closed=True
