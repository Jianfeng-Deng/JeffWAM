import os,sys,subprocess,time,traceback,json
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from e103d_runtime import *
def run(script,args,phase,gpu=False):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=wait_gpu(6144) if gpu else '',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 atomic(CONTROL/'PIPELINE.json',dict(status='running',phase=phase,brt=brt(),gpu=env['CUDA_VISIBLE_DEVICES']))
 with (CONTROL/(phase+'.log')).open('x') as f:
  p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/script)]+args,env=env,stdout=f,stderr=subprocess.STDOUT);atomic(CONTROL/'child.json',dict(pid=p.pid,phase=phase));rc=p.wait()
 assert rc==0,(phase,rc)
def main():
 assert not (CONTROL/'PIPELINE.json').exists()
 run('e106c_preflight.py',[],'preflight');run('e106c_train.py',['F2'],'train',True)
 for g in ('A2','F0','F2','S15000','S20000'):run('e106c_eval.py',[g],'eval_'+g,True)
 run('e106c_finalize.py',[],'finalize');atomic(CONTROL/'PIPELINE.json',dict(status='complete',brt=brt(),GPUh_upper=used_seconds()/3600))
if __name__=='__main__':
 try:main()
 except BaseException:
  report('## 管线停止\n'+traceback.format_exc());atomic(CONTROL/'PIPELINE.json',dict(status='failed',error=traceback.format_exc(),brt=brt()));raise
