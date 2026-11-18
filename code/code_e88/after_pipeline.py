import time,sys,subprocess,os
from e88.common import *
status=ART/'after_pipeline_status.json'
if status.exists():raise FileExistsError(status)
while True:
 p=ART/'pipeline_02_status.json';d=json.loads(p.read_text()) if p.exists() else {}
 if d.get('status')=='completed':break
 if d.get('status')=='stopped':raise RuntimeError(d)
 time.sleep(2)
atomic(status,dict(status='profiling_language_conditions',brt=brt()))
with (ART/'profile_conditions_01.log').open('x') as f:
 p=subprocess.run([sys.executable,'profile_conditions.py'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
atomic(status,dict(status='completed' if p.returncode==0 else 'stopped',returncode=p.returncode,brt=brt()))
