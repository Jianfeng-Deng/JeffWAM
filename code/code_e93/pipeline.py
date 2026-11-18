from common import *
import subprocess,traceback
try:
 while True:
  s=json.loads((R/'extract_status.json').read_text())
  if s['status']=='completed':break
  if s['status']=='stopped':raise RuntimeError(s)
  time.sleep(5)
 for stage in ('extract_supp','cpu_eval','gpu_diag'):
  atomic(R/'pipeline_status.json',dict(status='running',stage=stage,brt=brt()))
  with (R/(stage+'.log')).open('x') as f:subprocess.run([sys.executable,'-u',stage+'.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
 atomic(R/'pipeline_status.json',dict(status='completed',brt=brt()))
except BaseException as e:atomic(R/'pipeline_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc(),brt=brt()));raise
