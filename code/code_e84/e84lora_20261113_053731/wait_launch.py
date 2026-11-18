import time,subprocess,traceback
from common import *
try:
 started=time.time()
 while not (L/'READY_training_data.json').exists():
  assert time.time()-started<1800,'training preparation timeout; inspect prepare log'
  atomic(P/'launch_wait.json',dict(timestamp_brt=brt(),waiting='READY_training_data.json',elapsed_s=time.time()-started));time.sleep(10)
 subprocess.run(['/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python','-u',str(C/'driver_gpu.py')],cwd=T,check=True)
except BaseException:
 atomic(P/'launch_failure.json',dict(timestamp_brt=brt(),traceback=traceback.format_exc()));raise
