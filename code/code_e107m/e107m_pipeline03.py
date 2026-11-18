import os,subprocess,traceback
from e103d_runtime import *
PYTHON='/mnt/workspace/dengjianfeng/venvs/e29/bin/python'
def run(script,args,gpu=False):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 if gpu:env['CUDA_VISIBLE_DEVICES']=wait_gpu(6144)
 name=script.replace('.py','')+'_'+'_'.join(args)
 with (CONTROL/(name+'.log')).open('x') as f:
  child=subprocess.Popen([PYTHON,'-u',script]+args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);atomic(CONTROL/'PIPELINE03.json',dict(status='running',phase=name,pid=child.pid,brt=brt()));rc=child.wait()
 assert rc==0,(name,rc)
 assert used_seconds()<7100

def main():
 for g in ('M0','M1'):assert json.loads((T/f'results/e107m_{g.lower()}_s90_20261116_01/summary.json').read_text())['status']=='complete'
 for group in ('BASE','M0','M1'):run('e107m_eval03.py',[group],True)
 run('e107m_analyze03.py',[])
 atomic(CONTROL/'PIPELINE03.json',dict(status='complete',GPUh=used_seconds()/3600,brt=brt()))
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(CONTROL/'PIPELINE03.json',dict(status='failed',traceback=traceback.format_exc(),GPUh=used_seconds()/3600,brt=brt()));report('## 评估管线03停止\n'+traceback.format_exc());raise
