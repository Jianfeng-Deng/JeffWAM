from common import *
import traceback
try:
 put('driver_status.json',dict(stage='waiting_for_features',brt=now()))
 while not (R/'FEATURES_COMPLETE.json').exists():
  # Extraction safety exits and failures must not silently hang the orchestration.
  if (R/'gpu_ledger.json').exists():
   led=read('gpu_ledger.json')
   if led and led[-1]['status']=='closed':raise RuntimeError('extraction ended without FEATURES_COMPLETE; inspect retained log')
  time.sleep(20)
 for module,lock in [('train','HEADS_LOCK.json'),('tune','CONTROLLER_LOCK.json')]:
  if (R/lock).exists():continue
  put('driver_status.json',dict(stage=module,brt=now()))
  subprocess.run([sys.executable,'-u',str(T/'code_e102s2'/ (module+'.py'))],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
 put('driver_status.json',dict(stage='ready_for_dev',brt=now()))
except Exception:
 put('DRIVER_FAILURE_'+str(int(time.time()))+'.json',dict(brt=now(),traceback=traceback.format_exc()));raise
