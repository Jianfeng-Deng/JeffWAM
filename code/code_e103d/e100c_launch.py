import os,sys,time,subprocess,json,traceback
from e99w2_common import *
from e86.resource import gpu_rows

def main():
 assert json.loads((CONTROL/'CPU_TESTS.json').read_text())['status']=='passed'
 assert not (CONTROL/'PRETRAIN_LOCK.json').exists()
 for g in ('C0','C1','C2'):
  assert not (T/f'results/e100c_{g.lower()}_s90_20261115_03').exists()
  (CONTROL/f'{g}_s90').mkdir(exist_ok=False)
 atomic(CONTROL/'PRETRAIN_LOCK.json',dict(brt=brt(),seed=90,seeds_per_group=1,user_amendment='one seed per group before any GPU training',source_hash=source_hash(),eval_sha=sha(ROOT/'e100c_eval.py'),test_sha=sha(ROOT/'e100c_test.py'),groups=['C0','C1','C2'],steps=5000,init_sha=sha(INIT),bootstrap=2000,bootstrap_seed=90,dev='fixed48 windows/6parent; read-only E93 supplement',budget_s=10800))
 jobs=[];used=set();pending=['C0','C1','C2'];started=time.monotonic();endstate=[]
 while pending or jobs:
  good=sorted([r for r in gpu_rows() if r['uuid'] not in used and r['total']-r['used']>=6144],key=lambda r:(r['used'],r['util']))
  while pending and good and len(jobs)<3:
   g=pending.pop(0);r=good.pop(0);uuid=r['uuid'];used.add(uuid);a=CONTROL/f'{g}_s90';env=dict(os.environ,E100_GROUP=g,E100_SEED='90',CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');log=(a/'driver.log').open('x');p=subprocess.Popen([sys.executable,'-u','e100c_job.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(dict(group=g,p=p,log=log,start=time.monotonic(),uuid=uuid));atomic(a/'LAUNCH.json',dict(pid=p.pid,gpu=uuid,brt=brt(),estimated_peak_mib=4096,free_at_admission_mib=r['total']-r['used']))
  for j in list(jobs):
   elapsed=time.monotonic()-j['start']
   if elapsed>3000 and j['p'].poll() is None:
    # Only own process group; includes failed/cleanup time in accounting.
    import signal
    os.killpg(j['p'].pid,signal.SIGTERM)
   rc=j['p'].poll()
   if rc is not None:
    j['log'].close();used.remove(j['uuid']);jobs.remove(j);endstate.append(dict(group=j['group'],rc=rc,elapsed_s=elapsed,upper_s=elapsed+30));atomic(CONTROL/'SCHEDULER.json',dict(status='running',finished=endstate,active=[x['group'] for x in jobs],brt=brt()))
  assert sum(x['upper_s'] for x in endstate)+sum(time.monotonic()-j['start']+30 for j in jobs)<10800,'Session budget'
  time.sleep(5)
 atomic(CONTROL/'SCHEDULER.json',dict(status='complete' if all(x['rc']==0 for x in endstate) else 'failed',finished=endstate,GPUh_upper=sum(x['upper_s'] for x in endstate)/3600,brt=brt()))
if __name__=='__main__':
 try:main()
 except BaseException:atomic(CONTROL/'SCHEDULER_FAILURE.json',dict(traceback=traceback.format_exc(),brt=brt()));raise
