"""E86-V3 separate <=2.5 GPUh ledger; shared devices with 1.5x peak free."""
import os,json,time,fcntl,threading
from pathlib import Path
from .common import ART,atomic,brt
from e86.resource import gpu_rows
SESSION=ART
BUDGET=7200.;PEAK_MIB=12288
class Guard:
 def __init__(self,phase,limit_seconds=21500):
  self.mutex=threading.RLock();self.started=time.monotonic();self.limit=min(float(limit_seconds),7100.);self.closed=False;self.error=None
  self.lock=(SESSION/'gpu.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.path=SESSION/'gpu_ledger.json';self.ledger=json.loads(self.path.read_text()) if self.path.exists() else dict(schema='e96a-gpu-walltime-v1',budget_s=BUDGET,elapsed_s=0.,segments=[])
  if self.ledger.get('active'):raise RuntimeError('Unclosed GPU segment requires accounting')
  good=[r for r in gpu_rows() if r['total']-r['used']>=1.5*PEAK_MIB]
  import torch
  if torch.cuda.is_initialized():good=[r for r in good if r["uuid"]==os.environ.get("CUDA_VISIBLE_DEVICES")]
  if not good:raise RuntimeError('No shared GPU with >=1.5x estimated peak free')
  self.limit=min(self.limit,BUDGET-self.ledger['elapsed_s'])
  if self.ledger['elapsed_s']>=BUDGET:raise RuntimeError('GPUh budget exhausted')
  self.gpu=min(good,key=lambda r:(r['util'],r['used']));os.environ['CUDA_VISIBLE_DEVICES']=self.gpu['uuid'];os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  self.ledger['active']=dict(phase=phase,started_brt=brt(),pid=os.getpid(),gpu=self.gpu,limit_s=self.limit,peak_estimate_mib=PEAK_MIB);atomic(self.path,self.ledger)
  self.stop=threading.Event();self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
 def monitor(self):
  while not self.stop.wait(2):
   try:
    r=next(r for r in gpu_rows() if r['uuid']==self.gpu['uuid'])
    if r['total']-r['used']<1.5*PEAK_MIB:self.error='Shared GPU free memory below conservative 1.5x peak; stop'
    self.check()
   except BaseException as e:self.error=str(e)
   if self.error:return
 def check(self):
  if self.error:raise RuntimeError(self.error)
  import torch
  if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()/2**20>PEAK_MIB:raise RuntimeError('Measured CUDA peak exceeds admission estimate; stop')
  elapsed=time.monotonic()-self.started
  if elapsed>=self.limit or self.ledger['elapsed_s']+elapsed>=BUDGET:raise RuntimeError('GPU wall-time budget exhausted')
  with self.mutex:
   self.ledger['active']['elapsed_s']=elapsed;atomic(self.path,self.ledger)
  return elapsed
 def finish(self):
  if self.closed:return
  self.stop.set();self.thread.join(timeout=10);elapsed=time.monotonic()-self.started;s=self.ledger.pop('active');s.update(elapsed_s=elapsed,finished_brt=brt(),error=self.error);self.ledger['segments'].append(s);self.ledger['elapsed_s']+=elapsed;atomic(self.path,self.ledger);self.closed=True;self.lock.close()
