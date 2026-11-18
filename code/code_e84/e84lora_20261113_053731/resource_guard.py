import os,time,subprocess,threading,traceback
from common import *
import torch
class Guard:
 def __init__(self):
  assert os.environ['CUDA_VISIBLE_DEVICES']=='4';self.last=0;self.finished=False;self.budget=None;self.start=time.time();self.ledger=P/'gpu_ledger';self.ledger.mkdir(exist_ok=True);self.path=self.ledger/f'{os.getpid()}_{int(self.start)}.json'
  self.initial=self.snapshot();assert self.initial['free_mib']>=18432,self.initial
  torch.cuda.set_per_process_memory_fraction(12288/self.initial['total_mib'],0)
  atomic(P/f'gpu_admission_{os.getpid()}.json',dict(timestamp_brt=brt(),selected=self.initial,estimate_peak_mib=12288,required_free_mib=18432,basis='375M fp32 model 1.43GiB; LoRA196608+Adam<0.01GiB; batch4 x4 images224, gradient only last4 blocks; text<=64 sequences; conservative activation+CUDA allowance10.5GiB; allocator capped12GiB',env={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF']}))
  self.record('running');threading.Thread(target=self.watch,daemon=True).start()
 def snapshot(self):
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);r=next([int(v.strip()) for v in l.split(',')] for l in raw.splitlines() if int(l.split(',')[0])==4)
  result=dict(zip(['gpu','total_mib','used_mib','free_mib','utilization_percent'],r));result['all_cards_csv']=raw.strip();return result
 def record(self,status):
  elapsed=time.time()-self.start;atomic(self.path,dict(status=status,start_utc=self.start,elapsed_s=elapsed,pid=os.getpid(),timestamp_brt=brt(),phase='lora'))
  total=PRIOR_LORA_GPU_HOURS*3600+sum(json.loads(x.read_text())['elapsed_s'] for x in self.ledger.glob('*.json'));update(gpu_hours=total/3600,lora_gpu_hours=total/3600,total_e84_gpu_hours=1.276657608018981+total/3600)
  return total
 def check(self,force=False):
  if not force and time.time()-self.last<10:return
  self.last=time.time();s=self.snapshot();s.update(timestamp_brt=brt(),allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
  with (P/'gpu_monitor.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
  assert s['free_mib']>=18432,f'GPU memory pressure {s}'
  assert self.record('running')<4*3600-60,'4GPUh budget stop'
 def watch(self):
  while not self.finished:
   time.sleep(10)
   try:self.check()
   except BaseException:
    atomic(P/'resource_stop.json',dict(timestamp_brt=brt(),reason=traceback.format_exc()));update(status='stopped_resource_or_budget');self.record('finished');os._exit(73)
 def finish(self):
  self.finished=True;self.record('finished');atomic(P/f'gpu_peak_{os.getpid()}.json',dict(timestamp_brt=brt(),allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20))
