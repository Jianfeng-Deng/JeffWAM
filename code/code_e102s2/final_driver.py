from common import *
import traceback
try:
 put('final_driver_status.json',dict(stage='waiting_for_training_locks',brt=now()))
 while not (R/'CONTROLLER_LOCK.json').exists():
  failures=[p for pat in ['DRIVER_FAILURE_*.json','TRAIN_FAILURE_*.json','TUNE_FAILURE_*.json'] for p in R.glob(pat)]
  if failures:raise RuntimeError('training failure; not evaluating dev: '+','.join(p.name for p in failures))
  time.sleep(20)
 for script in ['evaluate.py','summarize.py']:
  put('final_driver_status.json',dict(stage=script,brt=now()))
  with (R/(script.replace('.py','')+'_01.log')).open('xb') as log:
   subprocess.run([sys.executable,'-u',str(T/'code_e102s'/script)],check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
 put('final_driver_status.json',dict(stage='COMPLETE',brt=now()))
except Exception:
 put('FINAL_DRIVER_FAILURE_'+str(int(time.time()))+'.json',dict(brt=now(),traceback=traceback.format_exc()));raise
