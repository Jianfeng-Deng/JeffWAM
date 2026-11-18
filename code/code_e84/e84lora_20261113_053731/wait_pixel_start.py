import os,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from common import *
try:
 start=time.time()
 while not (R/'features'/RUN/'shuffled/READY_pixel_features.json').exists():
  assert time.time()-start<300,'pixel features not ready';time.sleep(5)
 from run_pixel import main
 main('shuffled');atomic(P/'early_pixel_shuffled_completed.json',dict(timestamp_brt=brt(),status='completed'))
except BaseException:
 atomic(P/'early_pixel_failure.json',dict(timestamp_brt=brt(),reason=traceback.format_exc()));raise
