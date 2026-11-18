"""Single-device cooperative admission, persistent <=3 GPUh wall-time ledger."""
import subprocess,time,json,os,fcntl
from pathlib import Path
from .common import C,atomic,brt
BUDGET=3*3600
RESERVE=4096 # stop if other users exhaust this headroom
ESTIMATED_PEAK_MIB=16000

def gpu_rows():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(a),uuid=b.strip(),total=int(c),used=int(d),util=int(e)) for a,b,c,d,e in [line.split(',') for line in out.splitlines()]]
def choose(peak=ESTIMATED_PEAK_MIB):
    good=[r for r in gpu_rows() if r['total']-r['used']>=1.5*peak]
    if not good:raise RuntimeError('No GPU with free memory >=1.5 estimated peak')
    # Least utilization, then largest headroom; existing users remain untouched.
    return sorted(good,key=lambda r:(r['util'],r['used']))[0]
class Guard:
    def __init__(self,phase,limit_seconds):
        self.phase=phase;self.limit=min(limit_seconds,BUDGET);self.started=time.monotonic();self.closed=False
        self.path=C/'artifacts/gpu_ledger.json';self.lock=(C/'artifacts/gpu.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.ledger=json.loads(self.path.read_text()) if self.path.exists() else dict(schema='e86-gpu-walltime-v1',elapsed_s=0,segments=[])
        if self.ledger.get('active'):raise RuntimeError('Unclosed previous GPU segment: charge/reconcile before restart')
        self.gpu=choose();os.environ['CUDA_VISIBLE_DEVICES']=self.gpu['uuid'];self.ledger['active']=dict(phase=phase,started_brt=brt(),pid=os.getpid(),gpu=self.gpu,limit_s=self.limit);atomic(self.path,self.ledger)
    def check(self):
        elapsed=time.monotonic()-self.started
        self.ledger['active']['elapsed_s']=elapsed;atomic(self.path,self.ledger)
        if elapsed>self.limit or self.ledger['elapsed_s']+elapsed>=BUDGET:raise RuntimeError('GPU wall-time budget exhausted')
        r=next(r for r in gpu_rows() if r['uuid']==self.gpu['uuid'])
        if r['total']-r['used']<RESERVE:raise RuntimeError('Shared GPU memory pressure: immediate stop')
        return elapsed
    def finish(self):
        if self.closed:return
        elapsed=time.monotonic()-self.started;s=self.ledger.pop('active');s.update(elapsed_s=elapsed,finished_brt=brt());self.ledger['segments'].append(s);self.ledger['elapsed_s']+=elapsed;atomic(self.path,self.ledger);self.closed=True;self.lock.close()
