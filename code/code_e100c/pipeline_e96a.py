import os,sys,json,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from e96a.common import T,ART,atomic,brt,sha,report
run='e96a_m1_p1_20261114_01';status=ART/'pipeline_status.json';started=time.monotonic()
try:
 assert not (T/'results'/run).exists()
 # The independently launched CPU suite must finish successfully before GPU use.
 deadline=time.monotonic()+900
 while True:
  log=(ART/'contracts_01.log').read_text()
  if 'failed' in log or 'ERRORS' in log:raise RuntimeError('Contract failure; inspect contracts_01.log')
  if 'passed in' in log:
   final=log.strip().splitlines()[-1];break
  if time.monotonic()>deadline:raise RuntimeError('CPU contract suite timeout')
  atomic(status,dict(status='waiting_contracts',brt=brt()));time.sleep(5)
 atomic(ART/'contracts_status.json',dict(status='completed',summary=final,GPUh=0,brt=brt()))
 report('\n## 合同完成与正式训练启动\n'+brt()+'：'+final+'。含旧套件、E96a读出合同与全部camera×cell原位旁路合同；CPU离线，不下载数据/权重。')
 atomic(status,dict(status='training',run=run,brt=brt()))
 with (ART/'training_01.log').open('x') as f:
  subprocess.run([sys.executable,'-u','-m','e96a.train',run,'--recipe','M1'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 atomic(status,dict(status='training_completed',run=run,seconds=time.monotonic()-started,brt=brt()))
except BaseException as e:
 atomic(status,dict(status='stopped',error=str(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started,brt=brt()));raise
