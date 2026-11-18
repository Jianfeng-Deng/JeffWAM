import os,sys,subprocess,time,traceback
from common import *
py='/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python'
def run(args,log,env=None):
 with (P/log).open('a') as f:
  subprocess.run([py,'-u',*map(str,args)],cwd=T,env=env or os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 assert (L/'READY_training_data.json').exists()
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4',PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.8',PYTHONDONTWRITEBYTECODE='1')
 run([C/'train_lora.py'],'train.log',env)
 for variant in ['lora','shuffled']:
  env['VARIANT']=variant
  run([C/'extract.py','full','cuda:0',L/variant/'adapter.pt'],f'extract_{variant}.log',env)
  with (P/f'eval_{variant}.log').open('a') as f:
   subprocess.Popen([py,'-u',str(C/'cpu_eval.py'),variant],cwd=T,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 update(status='gpu_complete_cpu_evaluating',gpu_complete_brt=brt())
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(P/'gpu_pipeline_failure.json',dict(timestamp_brt=brt(),traceback=traceback.format_exc()));update(status='gpu_pipeline_failed');raise
