import os,sys,subprocess,json,traceback
from e99w2_common import *
try:
 for group in ('A0','A1','A2'):
  env=dict(os.environ,E102_GROUP=group,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
  with (CONTROL/f'cpu_{group}_03.log').open('x') as f:subprocess.run([sys.executable,'-u','e102a_test.py'],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 atomic(CONTROL/'CPU_TESTS_03.json',dict(status='passed',brt=brt(),source_hash=source_hash()))
except BaseException:
 atomic(CONTROL/'CPU_TESTS_03.json',dict(status='failed',traceback=traceback.format_exc(),brt=brt()));raise
