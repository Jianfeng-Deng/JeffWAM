import os,time,subprocess,threading,traceback
from common import *
import torch
class Guard:
    def __init__(self):
        self.last=0;self.finished=False;self.budget=None
        assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
        self.initial=self.snapshot();assert self.initial['free_mib']>=12288,self.initial
        torch.cuda.set_per_process_memory_fraction(8192/self.initial['total_mib'],0)
        atomic(P/'gpu_shared_admission.json',{'timestamp_brt':brt(),'selected':self.initial,'estimate_peak_mib':8192,'required_free_mib':12288,'basis':'375187970 fp32 parameters=1431MiB; frozen no-grad AMP vision batch8 224x224; conservative activation/workspace/text and CUDA overhead allowance ~6.6GiB; allocator cap8192MiB. No optimizer on GPU.','policy':'User revision2: shared GPU allowed, one card, no MPS/preemption/process changes; stop on OOM or <12GiB free','env':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF']}})
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def snapshot(self):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        rows=[[int(v.strip()) for v in line.split(',')] for line in raw.splitlines()]
        d=next(r for r in rows if r[0]==4)
        return dict(zip(['gpu','total_mib','used_mib','free_mib','utilization_percent'],d))
    def check(self,force=False):
        if not force and time.monotonic()-self.last<10:return
        self.last=time.monotonic();s=self.snapshot()
        s.update(timestamp_brt=brt(),allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        with (P/'gpu_monitor.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
        assert s['free_mib']>=12288,f'GPU memory pressure stop: {s}'
        if self.budget:self.budget.heartbeat()
    def watch(self):
        while not self.finished:
            time.sleep(10)
            try:self.check()
            except BaseException:
                atomic(P/'resource_stop.json',{'timestamp_brt':brt(),'reason':traceback.format_exc()})
                update(status='stopped_resource_pressure',awaiting='resource pressure or GPU budget; inspect resource_stop.json')
                if self.budget:self.budget.close()
                os._exit(73)
    def finish(self):
        self.check(force=True);self.finished=True
        atomic(P/'gpu_observed_peak.json',{'timestamp_brt':brt(),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
