import os,sys,time,subprocess,signal,traceback,json
from e99w2_common import *
from e86.resource import gpu_rows
PRIOR_GPU_S=json.loads((CONTROL/'PRIOR_GPU_USAGE.json').read_text())['prior_GPUh_upper']*3600
BUDGET=7200-PRIOR_GPU_S

def main():
 assert not (CONTROL/'PRETRAIN_LOCK.json').exists()
 assert json.loads((CONTROL/'CPU_TESTS_03.json').read_text())['status']=='passed'
 assert sha(INIT)=='a432001ffeb7d99fa2980ef337379464948d8c22678bb37dbfea0ab6708da99b'
 for g in ('A0','A1','A2'):
  assert not (T/f'results/e102a_{g.lower()}_s90_20261115_03').exists();(CONTROL/f'{g}_s90').mkdir(exist_ok=False)
 lock=dict(brt=brt(),groups=['A0','A1','A2'],steps=STEPS,seed=90,source_hash=source_hash(),init=str(INIT),init_sha=sha(INIT),contract_sha=sha(CONTROL/'GRIPPER_CONTRACT.json'),windows_sha=sha(CONTROL/'fixed_windows.json'),dev='fixed48/6parent; short human instruction only',train64='E101-A fixed64',lr='constant3e-4 all groups',mode_probabilities=[.25]*4,chain='C2 continuation global step5000+local; full .1; predicted reentry0.8; same all groups',FM='A0 all14; A1/A2 joints12 weight1; A2 deterministic delta MSE weight1 preserving FM short-W coupling',gripper='target-free t0 A-only BCE weight1, no event weighting; fixed openings from train',A2_history='164D current/causal -24/-18/-12/-6 q12, binary observed grip states, FK18, flags; no deletion in action_only, no state-drop deletion',evaluation='parent bootstrap2000 seed90; joint both prefixes L1 improvement CI>0; both A1/A2 F1@10 minus A0 CI>0 and lower false switches; world N1-minus-no-traj skill CI>0 at25/50/100; final only',budget_gpu_s=BUDGET,prior_gpu_s=PRIOR_GPU_S,disk_bytes=10*1024**3,peak_mib=6144)
 atomic(CONTROL/'PRETRAIN_LOCK.json',lock);report('## 最终训练前预注册\n'+json.dumps(lock,ensure_ascii=False))
 pending=['A0','A1','A2'];jobs=[];done=[];used=set()
 try:
  while pending or jobs:
   elapsed=sum(x['upper_s'] for x in done)+sum(time.monotonic()-x['start']+30 for x in jobs)
   assert elapsed<BUDGET,'Total GPUh budget exhausted'
   good=sorted([r for r in gpu_rows() if r['total']-r['used']>=1.5*6144 and r['uuid'] not in used],key=lambda r:(r['util'],r['used']))
   while pending and good and len(jobs)<3:
    g=pending.pop(0);r=good.pop(0);uuid=r['uuid'];used.add(uuid);art=CONTROL/f'{g}_s90';env=dict(os.environ,E102_GROUP=g,E102_SEED='90',CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');log=(art/'driver.log').open('x');p=subprocess.Popen([sys.executable,'-u','e102a_job.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(dict(group=g,p=p,log=log,start=time.monotonic(),uuid=uuid));atomic(art/'LAUNCH.json',dict(pid=p.pid,gpu=uuid,brt=brt(),estimated_peak_mib=6144,free_at_admission_mib=r['total']-r['used']))
   for j in list(jobs):
    rc=j['p'].poll();elapsed=time.monotonic()-j['start']
    if elapsed>2300 and rc is None:j['p'].terminate()
    if rc is not None:
     j['log'].close();used.remove(j['uuid']);jobs.remove(j);done.append(dict(group=j['group'],rc=rc,elapsed_s=elapsed,upper_s=elapsed+30))
   atomic(CONTROL/'SCHEDULER.json',dict(status='running' if jobs else 'queued' if pending else 'finishing',finished=done,active=[dict(group=j['group'],pid=j['p'].pid,elapsed_s=time.monotonic()-j['start']) for j in jobs],brt=brt()))
   if jobs or pending:time.sleep(5)
  upper=(PRIOR_GPU_S+sum(x['upper_s'] for x in done))/3600
  if all(x['rc']==0 for x in done):
   subprocess.run([sys.executable,'-u','e102a_finalize.py'],cwd=ROOT,check=True)
   status='complete'
  else:status='failed'
  atomic(CONTROL/'SCHEDULER.json',dict(status=status,finished=done,GPUh_upper=upper,brt=brt()));report('## 资源与执行终态\n'+json.dumps(dict(status=status,GPUh_upper=upper,finished=done,new_bytes=sum(p.stat().st_size for root in [ROOT,*CONTROL.parent.glob('e102a_*')] for p in root.rglob('*') if p.is_file()),old_checkpoint_unchanged=sha(INIT)==lock['init_sha']),ensure_ascii=False))
  with (T/'docs/experiment_log/EXPERIMENT_LOG.md').open('a') as f:
   import fcntl;fcntl.flock(f,fcntl.LOCK_EX);f.write('\n### E102-A 终态 '+brt()+' BRT\n'+json.dumps(dict(status=status,command='setsid nohup .../e29/bin/python -u e102a_launch.py',directory=str(CONTROL),changes='A1 binary gripper BCE; A2 causal proprio + deterministic delta MSE with FM joint auxiliary; all C2 world plans fixed openings',results=str(CONTROL/'SUMMARY.json'),conclusion=json.loads((CONTROL/'DECISION.json').read_text()) if status=='complete' else 'see failure status',decision=str(CONTROL/'DECISION.json'),GPUh_upper=upper),ensure_ascii=False)+'\n')
  with (T/'notes/worklog.md').open('a') as f:f.write('\n- '+brt()+' BRT E102-A '+status+'；GPUh上界='+str(upper)+'；报告 notes/e102a_action_fix_20261115.md。\n')
 finally:
  for j in jobs:
   if j['p'].poll() is None:j['p'].terminate()
   j['log'].close()
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(CONTROL/'SCHEDULER_FAILURE.json',dict(traceback=traceback.format_exc(),brt=brt()));raise
