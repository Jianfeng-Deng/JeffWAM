import subprocess,json,os,time,traceback
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';P='/mnt/workspace/dengjianfeng/venvs/e29/bin/python'
try:
 for script,args,name in [('analyze_v2.py',[],'analysis'),('sensitivity.py',[],'sensitivity'),('event_exposure_v2.py',['score'],'event_score')]:
  (O/'pipeline_status.json').write_text(json.dumps(dict(status='running',stage=name,GPUh=0)))
  with (O/(name+'.log')).open('x') as log:subprocess.run([P,'-B','-u',str(T/'code_e109g'/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
 (O/'PIPELINE_COMPLETE.json').write_text(json.dumps(dict(status='complete',GPUh=0)))
except BaseException:(O/'PIPELINE_FAILED.json').write_text(traceback.format_exc());raise
