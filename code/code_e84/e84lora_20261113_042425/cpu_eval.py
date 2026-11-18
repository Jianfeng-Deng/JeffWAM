import os,sys,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
from common import *
if __name__=='__main__':
 v=sys.argv[1]
 try:
  from run_pixel import main as pixel
  from run_motion import main as motion
  from run_behavior import main as behavior
  pixel(v);motion(v);behavior(v);atomic(P/f'READY_eval_{v}.json',dict(timestamp_brt=brt(),status='completed',variant=v))
 except BaseException:
  atomic(P/f'eval_failure_{v}.json',dict(timestamp_brt=brt(),traceback=traceback.format_exc()));raise
