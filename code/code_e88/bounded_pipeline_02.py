import os,sys,time,subprocess,json
from pathlib import Path
from e88.common import ART,atomic,brt,report
start=time.monotonic();status=ART/'pipeline_02_status.json'
if status.exists():raise FileExistsError(status)
for _ in range(180):
 text=(ART/'contracts_06.log').read_text() if (ART/'contracts_06.log').exists() else ''
 if '108 passed' in text:break
 if 'failed' in text or 'ERROR' in text:raise RuntimeError('CPU contracts failed')
 time.sleep(1)
else:raise TimeoutError('CPU contract completion not observed')
commands=[('smoke_03',['-m','e88.train','smoke_03']),('resume_P1',['-m','e88.replay','smoke_03','P1']),('resume_P3_short',['-m','e88.replay','smoke_03','P3_short']),('profile_01',['-m','e88.profile','smoke_03'])]
for phase,args in commands:
 atomic(status,dict(status='running',phase=phase,updated_brt=brt()))
 with (ART/('pipeline2_'+phase+'.log')).open('x') as f:
  child=subprocess.Popen([sys.executable]+args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
  atomic(status,dict(status='running',phase=phase,child_pid=child.pid,updated_brt=brt()));code=child.wait()
 if code:
  atomic(status,dict(status='stopped',phase=phase,returncode=code,updated_brt=brt()));sys.exit(code)
 report('### 工程流水线阶段完成 '+phase+'\nBRT='+brt()+'；详细数据已写入code_e88/artifacts/smoke_03。')
atomic(status,dict(status='completed',elapsed_s=time.monotonic()-start,updated_brt=brt()))
