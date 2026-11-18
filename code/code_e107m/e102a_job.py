import os,sys,signal,subprocess,traceback
from e99w2_common import *
def stop(sig,frame):raise SystemExit('E102-A own-process stop signal')
signal.signal(signal.SIGTERM,stop)
try:
 assert json.loads((CONTROL/'CPU_TESTS_03.json').read_text())['status']=='passed'
 assert (CONTROL/'PRETRAIN_LOCK.json').exists()
 from e99w2_data import load
 from e99w2_train import train
 assert Path(train.__code__.co_filename).resolve().parent==ROOT
 _,_,tr,_=load();train(tr)
 # Evaluate in same process so scheduler's own process termination always applies.
 from e102a_eval import evaluate
 evaluate();atomic(ART/'JOB.json',dict(status='complete',brt=brt()))
except BaseException:
 atomic(ART/'JOB.json',dict(status='failed',traceback=traceback.format_exc(),brt=brt()));raise
