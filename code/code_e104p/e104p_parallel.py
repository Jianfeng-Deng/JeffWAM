import os,sys,time,json,signal,subprocess,traceback
from e104p_pipeline import PY,disk,shared,state,run
from e103d_runtime import *
def main():
 lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text())
 for name in ['e104p_train.py','e104p_eval.py','e104p_finalize.py','e99w2_model.py','e99w2_objective.py','e103d_model.py','e103d_data.py']:assert sha(ROOT/name)==lock['code_hashes'][name],name
 old=json.loads((CONTROL/'PIPELINE.json').read_text());assert old['pid']==1884687 and old['args']==['F2'];assert b'e104p_pipeline.py' in Path('/proc/1884687/cmdline').read_bytes()
 f2=json.loads((CONTROL/'gpu_train_F2.json').read_text());pid2=f2['pid'];assert b'e104p_train.py' in Path(f'/proc/{pid2}/cmdline').read_bytes();assert f2['status']=='active'
 assert not (T/'results/e104p_f1_s90_20261115_03').exists();assert not (CONTROL/'F1_train.log').exists();assert not (CONTROL/'ORCHESTRATION_LOCK.json').exists()
 atomic(CONTROL/'SERIAL_ORCHESTRATION_BEFORE.json',old)
 # Stop only the idle serial parent, leaving its independently running F2 child untouched.
 os.kill(1884687,signal.SIGTERM)
 row=sorted([r for r in gpu_rows() if r['uuid']!=f2['gpu']['uuid'] and r['free']>=9216],key=lambda r:r['free'],reverse=True)[0]
 orch=dict(brt=brt(),change='Only parallel orchestration; no change to training/eval/data/loss/seed/optimizer/checkpoints/decision',serial_parent_stopped=1884687,existing_F2_pid=pid2,F2_gpu=f2['gpu'],F1_gpu=row,code_hashes=lock['code_hashes'],orchestrator_sha=sha(Path(__file__)),global_GPU_seconds_stop=10400,includes_all_prior_failures=True)
 atomic(CONTROL/'ORCHESTRATION_LOCK.json',orch);report('## 仅调度并行化（模型/配方锁不变）\n'+json.dumps({k:v for k,v in orch.items() if k!='code_hashes'},ensure_ascii=False))
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',E102_GROUP='A2',CUDA_VISIBLE_DEVICES=row['uuid'])
 log=(CONTROL/'F1_train.log').open('x');p1=subprocess.Popen([PY,'-u','e104p_train.py','F1'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 atomic(CONTROL/'PARALLEL_LAUNCH.json',dict(F1_pid=p1.pid,F2_pid=pid2,brt=brt(),F1_free_at_admission_mib=row['free'],peak_estimate_mib=6144))
 try:
  while True:
   total=used_seconds()+30*len(list(CONTROL.glob('gpu_*.json')));assert total<10400,'aggregate GPUh reserve stop';assert disk()<9e9
   p2=CONTROL/'gpu_train_F2.json';s2=json.loads(p2.read_text());rc1=p1.poll();done2=s2['status']=='closed'
   state('parallel_training',F1_pid=p1.pid,F2_pid=pid2,F1_done=rc1 is not None,F2_done=done2)
   if rc1 is not None and rc1!=0:raise RuntimeError('F1 failed rc='+str(rc1))
   if (CONTROL/'train_F2_error.json').exists():raise RuntimeError('F2 failed')
   if rc1==0 and done2:break
   if not Path(f'/proc/{pid2}').exists() and not done2:raise RuntimeError('F2 vanished without closed guard')
   time.sleep(2)
 finally:
  log.close()
  if sys.exc_info()[0] is not None:
   for pid in [pid2,p1.pid]:
    try:
     if b'e104p_train.py' in Path(f'/proc/{pid}/cmdline').read_bytes():os.kill(pid,signal.SIGTERM)
    except FileNotFoundError:pass
 a=json.loads((T/'results/e104p_f2_s90_20261115_03/summary_05000.json').read_text());b=json.loads((T/'results/e104p_f1_s90_20261115_03/summary_05000.json').read_text());checks={k:a[k]==b[k] for k in ['P3_samples_sha256','P1_samples_sha256','P3_flow_sha256','modes','P3_draws','P1_draws']};assert all(checks.values()),checks;atomic(CONTROL/'PAIRED_STREAM_AUDIT.json',dict(status='passed',checks=checks,F0=a,F1=b));report('## 同Piper数据逐样本核验\n'+json.dumps(checks,ensure_ascii=False))
 for group in ['A2','F1-pre','F0','F1','F2']:run('e104p_eval.py',[group],group+'_eval.log',True)
 run('e104p_finalize.py',[],'finalize.log');res=json.loads((CONTROL/'DECISION.json').read_text());upper=(used_seconds()+30*len(list(CONTROL.glob('gpu_*.json'))))/3600;assert upper<=3 and disk()<=10e9
 report('## 资源与最终状态\n'+json.dumps(dict(status='COMPLETE',brt=brt(),GPUh_upper=upper,new_bytes=disk(),segments=[json.loads(p.read_text()) for p in CONTROL.glob('gpu_*.json')],checkpoint_cache_old_code_readonly=True),ensure_ascii=False)+'\n未完成必需项：无。需要用户决定：无。外部有效性严格按预注册判定；不部署。');shared('COMPLETE',res['verdict']+f'；GPUh上界{upper:.4f}；新增{disk()/1e9:.3f}GB；全部固定终点训练/评估完成。');state('complete',decision=res)
if __name__=='__main__':
 try:main()
 except BaseException:
  err=traceback.format_exc();state('stopped',error=err);report('## 并行调度停止\n```\n'+err+'\n```\n必需步骤未全部完成，不作外部有效结论。失败计入GPUh，旧输出保留。');shared('STOPPED','必需项未全部完成，详见PIPELINE.json；不作增益结论。');raise
