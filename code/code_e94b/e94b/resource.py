from .common import *
import subprocess,threading
BUDGET=14400.;PEAK=8192
class Guard:
 def __init__(self,phase):
  self.path=R/'gpu_ledger.json';self.lock=(R/'gpu.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.ledger=json.loads(self.path.read_text()) if self.path.exists() else dict(elapsed_s=0.,budget_s=BUDGET,segments=[])
  if self.ledger.get('active'):raise RuntimeError('Unclosed GPU segment; reconcile before continuation')
  if self.ledger['elapsed_s']>=BUDGET:raise RuntimeError('4GPUh exhausted')
  cards=self.cards();good=[r for r in cards if r['free']>=PEAK*1.5]
  if not good:raise RuntimeError('No GPU with >=1.5x peak free')
  self.gpu=max(good,key=lambda r:r['free']);os.environ.update(CUDA_VISIBLE_DEVICES=self.gpu['uuid'],CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_CACHE_PATH=str(R/'cuda_cache'),CUDA_CACHE_MAXSIZE='67108864')
  self.started=time.monotonic();self.error=None;self.mutex=threading.RLock();self.stop=threading.Event()
  self.ledger['active']=dict(phase=phase,pid=os.getpid(),gpu=self.gpu,started_brt=brt(),peak_estimate_mib=PEAK);atomic(self.path,self.ledger)
  self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
 def cards(self):
  out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
  return [dict(index=int(a),uuid=b.strip(),free=int(c)) for a,b,c in (r.split(',') for r in out.splitlines())]
 def monitor(self):
  while not self.stop.wait(2):
   try:
    free=next(r['free'] for r in self.cards() if r['uuid']==self.gpu['uuid'])
    if free<PEAK*1.5:self.error='Other/shared memory pressure, stopping'
   except BaseException as e:self.error=str(e)
 def check(self):
  if self.error:raise RuntimeError(self.error)
  import torch
  peak=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0
  if peak>PEAK:raise RuntimeError('Peak exceeds estimate')
  elapsed=time.monotonic()-self.started
  if elapsed+self.ledger['elapsed_s']>=BUDGET:raise RuntimeError('4GPUh exhausted')
  with self.mutex:self.ledger['active'].update(elapsed_s=elapsed,peak_mib=peak);atomic(self.path,self.ledger)
  return elapsed
 def finish(self):
  self.stop.set();self.thread.join(timeout=3)
  s=self.ledger.pop('active');s.update(elapsed_s=time.monotonic()-self.started,finished_brt=brt(),error=self.error)
  self.ledger['segments'].append(s);self.ledger['elapsed_s']+=s['elapsed_s'];atomic(self.path,self.ledger);self.lock.close()
