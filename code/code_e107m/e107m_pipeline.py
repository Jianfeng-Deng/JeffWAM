import os,sys,subprocess,traceback,time
from e103d_runtime import *
PYTHON='/mnt/workspace/dengjianfeng/venvs/e29/bin/python'
def run(script,args,gpu=False):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 if gpu:env['CUDA_VISIBLE_DEVICES']=wait_gpu(6144)
 name=script.replace('.py','')+'_'+'_'.join(args)
 with (CONTROL/(name+'.log')).open('x') as f:
  child=subprocess.Popen([PYTHON,'-u',script]+args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);atomic(CONTROL/'PIPELINE.json',dict(status='running',phase=name,pid=child.pid,brt=brt()));rc=child.wait()
 assert rc==0,(name,rc)
 if gpu:assert used_seconds()<7100

def main():
 while not (CONTROL/'CPU_TESTS.json').exists():
  if (CONTROL/'CPU_TEST_FAILED.json').exists():raise RuntimeError('CPU tests failed')
  time.sleep(5)
 for group in ('M0','M1'):run('e107m_train.py',[group],True)
 for group in ('BASE','M0','M1'):run('e107m_eval.py',[group],True)
 run('e107m_analyze.py',[])
 atomic(CONTROL/'PIPELINE.json',dict(status='complete',GPUh=used_seconds()/3600,brt=brt()))
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(CONTROL/'PIPELINE.json',dict(status='failed',traceback=traceback.format_exc(),GPUh=used_seconds()/3600,brt=brt()));report('## 管线停止\n'+traceback.format_exc());raise
