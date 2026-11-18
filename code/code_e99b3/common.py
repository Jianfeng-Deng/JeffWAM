import os,sys,json,time,hashlib,datetime,subprocess,threading,signal
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu'); R=T/'results/e99b3_20261115_01'; REPORT=T/'notes/e99b3_predicates_20261115.md'
sys.path.insert(0,str(T/'code_e98w'))
CAMS=('cam_high','cam_left_wrist','cam_right_wrist'); EXCLUDE='piper_4dc107335389f5711a0f65e0'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def put(name,x):
 p=R/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def read(name):return json.loads((R/name).read_text())
def status(stage,**kw):put('status.json',dict(stage=stage,brt=now(),**kw));print(stage,kw,flush=True)
def report(s):
 with REPORT.open('a') as f:f.write('\n'+s+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def jl(p):return [json.loads(s) for s in Path(p).open()]
def resolve(p):
 p=str(p);return p.replace('/mnt/workspace/colleague-a/Datasets/piper/','/mnt/workspace/dataset/piper/') if not Path(p).exists() else p
def gpu_rows():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
 return [[int(v.strip()) for v in l.split(',')] for l in s.splitlines()]
class Guard:
 def __init__(self,phase,peak_gb):
  self.phase=phase; self.peak=peak_gb*1024; self.started=time.monotonic();self.closed=False
  self.ledger=read('gpu_ledger.json') if (R/'gpu_ledger.json').exists() else [];self.used=sum(x['seconds'] for x in self.ledger);assert self.used<10800
  g=max(gpu_rows(),key=lambda r:r[1]);assert g[1]>=1.5*self.peak,(g,self.peak)
  self.gpu=g[0];os.environ['CUDA_VISIBLE_DEVICES']=str(self.gpu);self.initial=g
  self.entry=dict(phase=phase,gpu=self.gpu,initial_free_mib=g[1],estimated_peak_mib=self.peak,start_brt=now(),seconds=0,status='running');self.ledger.append(self.entry);put('gpu_ledger.json',self.ledger)
  self.stop=threading.Event();self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
 def check(self):
  elapsed=time.monotonic()-self.started
  if self.used+elapsed>10700:raise RuntimeError('GPU budget safety stop')
  if sum(p.stat().st_size for p in R.rglob('*') if p.is_file())>19_500_000_000:raise RuntimeError('Disk budget stop')
 def watch(self):
  while not self.stop.wait(3):
   try:
    free=next(r[1] for r in gpu_rows() if r[0]==self.gpu)
    if free<max(4096,self.peak*.5) or self.used+time.monotonic()-self.started>10750:
     self.entry['abort_reason']='shared_memory_pressure_or_budget';self.finish();os.kill(os.getpid(),signal.SIGTERM);return
   except Exception:pass
 def finish(self):
  if self.closed:return
  self.closed=True;self.stop.set();self.entry.update(seconds=time.monotonic()-self.started,end_brt=now(),status='closed');put('gpu_ledger.json',self.ledger)
