from pathlib import Path
import time,json,subprocess,sys,traceback,os
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from e96a.common import ART,atomic,brt
try:
 atomic(ART/'finalization_waiter_status.json',dict(status='waiting_evaluation',brt=brt()))
 deadline=time.monotonic()+3500
 while True:
  x=json.loads((ART/'evaluation_waiter_status.json').read_text())
  if x['status']=='stopped':raise RuntimeError('Evaluation stopped')
  if x['status']=='completed':break
  if time.monotonic()>deadline:raise RuntimeError('Wait timeout')
  time.sleep(10)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 with (ART/'finalization_01.log').open('x') as f:
  subprocess.run([sys.executable,'-u','finalize_e96a.py'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
 atomic(ART/'finalization_waiter_status.json',dict(status='completed',brt=brt()))
except BaseException as e:
 atomic(ART/'finalization_waiter_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc(),brt=brt()));raise
