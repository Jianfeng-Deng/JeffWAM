import os,time,sys,subprocess,json
from e88.common import *
status=ART/'finish_driver_status.json'
if status.exists():raise FileExistsError(status)
while True:
 p=ART/'after_pipeline_status.json';d=json.loads(p.read_text()) if p.exists() else {}
 if d.get('status')=='completed':break
 if d.get('status')=='stopped':raise RuntimeError(d)
 time.sleep(2)
for phase,script in [('conditional_inference','inference_conditions.py'),('finalize','finalize.py')]:
 atomic(status,dict(status='running',phase=phase,brt=brt()))
 with (ART/('finish_'+phase+'.log')).open('x') as f:p=subprocess.run([sys.executable,script],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:atomic(status,dict(status='stopped',phase=phase,returncode=p.returncode,brt=brt()));sys.exit(p.returncode)
atomic(status,dict(status='completed',brt=brt()))
