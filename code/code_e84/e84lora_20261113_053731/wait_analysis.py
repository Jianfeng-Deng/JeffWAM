import os,time,subprocess,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from common import *
try:
 started=time.time()
 while not all((P/f'READY_eval_{v}.json').exists() for v in ['lora','shuffled']):
  failed=[p.name for p in P.glob('*failure*.json')]
  assert not failed,failed
  assert not (P/'resource_stop.json').exists(),'resource_stop'
  assert time.time()-started<8*3600,'CPU/result wait timeout'
  atomic(P/'analysis_wait.json',dict(timestamp_brt=brt(),elapsed_s=time.time()-started,waiting='both evaluation READY'));time.sleep(30)
 py='/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python'
 for name in ['analyze.py','figures.py','report.py']:
  subprocess.run([py,'-u',str(C/name)],cwd=T,check=True)
 update(status='results_and_report_draft_ready')
except BaseException:
 atomic(P/'analysis_failure.json',dict(timestamp_brt=brt(),traceback=traceback.format_exc()));raise
