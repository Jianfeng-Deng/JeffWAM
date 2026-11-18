import os,sys,time,json,threading,signal,inspect,traceback
from pathlib import Path
sys.dont_write_bytecode=True
from e102x_setup import *
import e102a_eval as ev
import torch
from e86.resource import gpu_rows
GROUP=os.environ['E102_GROUP'];ART=OUT/os.environ.get('E102X_ATTEMPT',GROUP)
class Guard:
 def __init__(self,phase,limit):
  self.mutex=threading.RLock();self.start=time.monotonic();self.stop=threading.Event();self.error=None;self.closed=False;self.path=ART/'gpu.json';assert not self.path.exists()
  self.gpu=next(r for r in gpu_rows() if r['uuid']==os.environ['CUDA_VISIBLE_DEVICES']);assert self.gpu['total']-self.gpu['used']>=9216
  self.meta=dict(status='active',group=GROUP,gpu=self.gpu,started=brt(),peak_estimate_mib=6144);atomic(self.path,self.meta)
  self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
 def check(self):
  with self.mutex:
   assert not self.error,self.error
   assert time.monotonic()-self.start<550,'segment budget'
   assert torch.cuda.max_memory_allocated()/2**20<=6144,'peak estimate exceeded'
   self.meta.update(elapsed_s=time.monotonic()-self.start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20);atomic(self.path,self.meta)
 def monitor(self):
  while not self.stop.wait(2):
   try:
    self.check();r=next(r for r in gpu_rows() if r['uuid']==self.gpu['uuid']);assert r['total']-r['used']>=9216,'shared memory pressure'
   except BaseException as e:
    self.error=str(e);self.meta.update(status='stopping',error=self.error);atomic(self.path,self.meta);os.kill(os.getpid(),signal.SIGTERM);return
 def finish(self):
  if self.closed:return
  self.stop.set();self.thread.join(5);self.meta.update(status='closed',elapsed_s=time.monotonic()-self.start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,error=self.error,finished=brt());atomic(self.path,self.meta);self.closed=True

def main():
 lock=json.loads((OUT/'BEFORE_FORWARD_LOCK.json').read_text());ART.mkdir(exist_ok=False)
 assert sha(SRC/'e102a_eval.py')==json.loads((OUT/'SOURCE_LOCK.json').read_text())['eval_sha']
 src=inspect.getsource(ev.evaluate)
 src=src.replace("assert len(f['train64'])==64 and len(f['dev'])==48", "assert len(f['dev'])==len(lock['entries'])")
 start=src.index(' c0=json.loads(');end=src.index(" cp=torch.load",start);src=src[:start]+src[end:]
 src=src.replace("[('dev',dev,f['dev']),('train64',tr,f['train64'])]","[('dev',dev,f['dev'])]")
 src=src.replace('     actions.append(item)',"     item.update(cid=r['canonical_id'],anchor=r['anchor'],true_event_ids=[[r['canonical_id'],k,r['anchor']+t,d] for k in range(2) for t,d in events(labels[:,k],init[k])]);actions.append(item)")
 src=src.replace('for h in (25,50,100):','for h in (10,25,50,100):\n       if h not in entry[\'horizons\']:continue')
 src=src.replace("seq=qq[a+1:a+101];assert len(seq)==100", "seq=qq[np.minimum(a+np.arange(1,101),len(qq)-1)]")
 src=src.replace("       sources=dict(gt_action", "       feat[~valid.numpy()]=0\n       sources=dict(gt_action")
 start=src.index("   b=move(dev.batch([f['dev'][0]");end=src.index(' finally:guard.finish()',start);src=src[:start]+src[end:]
 # Keep immutable E102-A evaluator mechanics; isolated globals route all outputs here.
 ns=dict(ev.__dict__);ns.update(CONTROL=OUT,ART=ART,Guard=Guard,lock=lock,load=datasets)
 (ART/'evaluator.py').write_text(src)
 exec(compile(src,str(T/'code_e102x/e102x_worker_evaluator.py'),'exec'),ns)
 ns['evaluate']()
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(OUT/(GROUP+'_failure.json'),dict(brt=brt(),traceback=traceback.format_exc()));raise
