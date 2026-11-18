import os,sys,json,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from e94a.common import T,ART,atomic,brt,sha
runs=[('M1','e94a_m1_p1_20261114_01'),('M2','e94a_m2_p1_20261114_01')]
status=ART/'pipeline_status.json';started=time.monotonic()
try:
 assert (ROOT/'READY_readout.json').exists()
 ready=json.loads((ROOT/'READY_readout.json').read_text())
 for p,h in ready['sha256'].items():assert sha(ROOT/p)==h,('Frozen interface changed',p)
 for recipe,run in runs:assert not (T/'results'/run).exists(),run
 for recipe,run in runs:
  atomic(status,dict(status='training',recipe=recipe,run=run,brt=brt()))
  with (ART/(recipe+'_training.log')).open('x') as f:subprocess.run([sys.executable,'-u','-m','e94a.train',run,'--recipe',recipe],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 for recipe,run in runs:
  atomic(status,dict(status='evaluating',recipe=recipe,run=run,brt=brt()))
  with (ART/(recipe+'_evaluation.log')).open('x') as f:subprocess.run([sys.executable,'-u','eval_e94a.py',recipe,run],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 atomic(status,dict(status='completed',runs=runs,seconds=time.monotonic()-started,brt=brt()))
except BaseException as e:atomic(status,dict(status='stopped',error=str(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started,brt=brt()));raise
