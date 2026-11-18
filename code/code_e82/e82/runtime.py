from __future__ import annotations
from pathlib import Path
import datetime,fcntl,json,os,subprocess,time
import numpy as np

def brt(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def atomic_json(path,d):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)); os.replace(tmp,path)

def idle_gpu():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[tuple(map(int,line.split(','))) for line in out.strip().splitlines()]
    available=[i for i,m,u in rows if m<512 and u<5]
    return (available[0] if available else None),rows

class Budget:
    def __init__(self,path,gpus=1,limit=48.,identity='unknown'):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.gpus=gpus; self.limit=limit; self.identity=identity; self.start=time.monotonic(); self.last=self.start
    def used(self):
        if not self.path.exists(): return 0.
        with self.path.open() as f: return sum(json.loads(x)['gpuh'] for x in f if x.strip())
    def tick(self):
        now=time.monotonic(); seconds=now-self.last; self.last=now
        entry=dict(brt=brt(),identity=self.identity,gpus=self.gpus,wall_seconds=seconds,gpuh=seconds*self.gpus/3600)
        with self.path.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX); f.write(json.dumps(entry)+'\n'); f.flush(); os.fsync(f.fileno())
        return self.used()
    def check(self):
        used=self.used()+(time.monotonic()-self.last)*self.gpus/3600
        if used>=self.limit: raise RuntimeError(f'GPUh budget stop: {used:.4f} >= {self.limit}')

class LatencyRecorder:
    def __init__(self): self.ms=[]
    def add(self,seconds): self.ms.append(seconds*1000)
    def summary(self):
        if not self.ms: return dict(n=0,qualified=False)
        a=np.asarray(self.ms); miss=int((a>333).sum()); n=len(a)
        # Wilson interval for hard-deadline miss fraction.
        p=miss/n; z=1.96; den=1+z*z/n
        mid=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return dict(n=n,p50_ms=float(np.quantile(a,.5)),p95_ms=float(np.quantile(a,.95)),p99_ms=float(np.quantile(a,.99)),deadline_misses=miss,deadline_miss_rate=p,deadline_miss_ci95=[max(0,mid-half),min(1,mid+half)],qualified=n>=1000 and np.quantile(a,.99)<=250 and p<.01,provisional=n<1000)

def cache_guard(root,limit_gb=100):
    # Only inventory this run's owned output; never touch source or old runs.
    total=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
    if total>=.9*limit_gb*1e9: raise RuntimeError('Owned artifact bucket reached 90%; stop writes')
    return total
