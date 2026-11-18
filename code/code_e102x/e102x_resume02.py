import os,sys,time,json,subprocess,traceback
from pathlib import Path
sys.dont_write_bytecode=True
from e102x_setup import *
from e86.resource import gpu_rows

def main():
 ledger=json.loads((OUT/'STATUS.json').read_text());assert ledger['status']=='failed'
 atomic(OUT/'SCHEDULER_failed01.json',ledger)
 ledger['status']='resuming';ledger.pop('traceback',None)
 for group in ('A1','A2'):
  assert ledger['GPUh_upper']*3600+580<=1800,'total budget reservation failed'
  avail=[r for r in gpu_rows() if r['total']-r['used']>=9216];assert avail,'no admissible GPU'
  gpu=sorted(avail,key=lambda r:(r['util'],r['used']))[0]
  env=dict(os.environ,E102_GROUP=group,CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',E102X_ATTEMPT=group+'_retry02')
  start=time.monotonic()
  with (OUT/(group+'_retry02.log')).open('x') as log:
   proc=subprocess.Popen([sys.executable,'-u',str(T/'code_e102x/e102x_worker.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   ledger.update(active=dict(group=group,pid=proc.pid,gpu=gpu,brt=brt()));atomic(OUT/'STATUS.json',ledger)
   try:rc=proc.wait(timeout=550)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,15)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,9);proc.wait()
    rc=-15
  elapsed=time.monotonic()-start;ledger.pop('active');ledger['segments'].append(dict(group=group,rc=rc,elapsed_s=elapsed,upper_s=elapsed+30));ledger['GPUh_upper']=sum(x['upper_s'] for x in ledger['segments'])/3600;atomic(OUT/'STATUS.json',ledger)
  assert rc==0,f'{group} failed {rc}'
 ledger.update(status='inference_complete',finished=brt());atomic(OUT/'STATUS.json',ledger)
 print(json.dumps(ledger),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=OUT/'STATUS.json';s=json.loads(p.read_text()) if p.exists() else {};s.update(status='failed',traceback=traceback.format_exc(),brt=brt());atomic(p,s);raise
