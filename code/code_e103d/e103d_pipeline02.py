import os,sys,json,time,threading,subprocess,traceback,fcntl,hashlib
from e103d_runtime import *
PY='/mnt/workspace/dengjianfeng/venvs/e29/bin/python'
def disk():
 roots=[ROOT,CACHE,CONTROL,*T.glob('results/e103d_d[01]_s90_20261115_*')]
 return sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file() and not p.is_symlink())
def state(status,**kw):atomic(CONTROL/'PIPELINE.json',dict(status=status,pid=os.getpid(),pipeline='02',brt=brt(),**kw))
def run(script,args,logname,gpu=False):
 assert disk()<40e9;env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=wait_gpu() if gpu else '')
 if args and args[0] in ['D0','D1']:env['E103_GROUP']=args[0]
 state('running',script=script,args=args,gpu=env['CUDA_VISIBLE_DEVICES'],gpu_seconds=used_seconds())
 with (CONTROL/logname).open('x') as log:subprocess.run([PY,'-u',script,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def shared(status,msg):
 for p,txt in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',f'\n## E103-D | {brt()} BRT | {status}\n目标：ABC/yam无动作视频对Piper世界预测增益；C2只读起点，seed90各5000步，P3+等权P1世界辅助成对比较。代码code_e103d；缓存data/e103d_cache；控制results/e103d_control_20261115_01；有效训练results/e103d_d0_s90_20261115_02、e103d_d1_s90_20261115_02。命令：setsid nohup e29/bin/python -u e103d_pipeline02.py；e103d_train.py D0/D1；e103d_eval.py D0/D1/finish。{msg}\n报告notes/e103d_external_data_20261115.md。失败/中断保留并计入GPUh。\n'),(T/'notes/worklog.md',f'\n- {brt()} BRT E103-D {status}：{msg}；有效run02，报告notes/e103d_external_data_20261115.md。\n')]:
  with p.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(txt);f.flush()
def resource_test():
 import e103d_runtime as rt,torch
 old=rt.gpu_rows
 def forbidden():raise AssertionError('Per-step nvidia-smi remains')
 rt.gpu_rows=forbidden;g=rt.Guard.__new__(rt.Guard);g.start=time.monotonic();g.limit=60;g.peak=6000;g.mutex=threading.RLock();g.path=CONTROL/'CPU_RESOURCE_CHECK_DETAIL.json';g.meta=dict(status='CPU_test');errors=[]
 def worker():
  try:
   for _ in range(50):g.check()
  except BaseException as e:errors.append(repr(e))
 ts=[threading.Thread(target=worker) for _ in range(4)]
 try:
  for t in ts:t.start()
  for t in ts:t.join()
  assert not errors,errors;assert not torch.cuda.is_initialized()
  g.start-=100
  try:g.check();raise AssertionError('Budget guard failed')
  except AssertionError as e:assert 'GPU budget exhausted' in str(e)
 finally:rt.gpu_rows=old
 result=dict(status='passed',concurrent_checks=200,no_CLI_in_step=True,elapsed_budget_check=True,CUDA_initialized=False,background_memory_monitor_seconds=2,brt=brt());atomic(CONTROL/'CPU_RESOURCE_TEST_02.json',result);return result

def main():
 assert not (CONTROL/'PIPELINE.json').exists();state('CPU_resource_test');resource_test()
 assert json.loads((CONTROL/'cache_status.json').read_text())['status']=='complete';assert json.loads((CONTROL/'CPU_TESTS.json').read_text())['status']=='passed';assert json.loads((CONTROL/'CACHE_COVERAGE.json').read_text())['status']=='passed'
 oldlock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text())
 for name in ['e103d_model.py','e103d_data.py','e99w2_model.py','e99w2_objective.py','e100c_chain.py','e90/data.py','e88/data.py','e99w2_data.py']:assert sha(ROOT/name)==oldlock['code_hashes'][name],name
 lock=dict(oldlock);lock.update(brt=brt(),run_suffix='20261115_02',prior_lock_sha256=sha(CONTROL/'PRETRAIN_LOCK.json'),runtime_only_revision='Removed synchronous per-update nvidia-smi; background2s memory pressure check matches C2. Time/peak checks every update. Model/data/objectives/seed/N/optimizer/eval/decision unchanged. Aborted D0 run01 at300 status steps, all391.358s charged; no dev scoring performed.',CPU_resource_test_sha256=sha(CONTROL/'CPU_RESOURCE_TEST_02.json'),code_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py'))})
 assert not (CONTROL/'PRETRAIN_LOCK_02.json').exists();atomic(CONTROL/'PRETRAIN_LOCK_02.json',lock);report('## 运行时优化后同预注册重启（无评分变更）\nD0 run01同步nvidia-smi每次实测0.458s，300步后中断，完整391.358s计入；日志、代码快照、旧锁保留。新run02只将空闲显存检查移至2秒后台监控，每步仍检查预算/峰值，与C2相同监控方式。200次四线程CPU检查和超时拒绝通过，无CUDA初始化；模型/数据/损失/优化器/seed/N/判据不变。PRETRAIN_LOCK_02.json在新训练前固定；无任何dev模型评分。')
 for group in ['D0','D1']:run('e103d_train.py',[group],group+'_train02.log',True)
 ss=[json.loads((T/f'results/e103d_{g.lower()}_s90_20261115_02/train_summary.json').read_text()) for g in ['D0','D1']];assert ss[0]['P3_samples_sha256']==ss[1]['P3_samples_sha256'];assert ss[0]['modes']==ss[1]['modes'];assert all(x['steps']==5000 for x in ss)
 for group in ['D0','D1']:run('e103d_eval.py',[group],group+'_eval02.log',True)
 run('e103d_eval.py',['finish'],'finalize02.log');result=json.loads((CONTROL/'FINAL_RESULTS.json').read_text());used=used_seconds();size=disk();assert used<=21600 and size<=40e9;resources=dict(gpu_seconds=used,gpu_hours=used/3600,new_bytes=size,new_decimal_GB=size/1e9,segments=[json.loads(p.read_text()) for p in sorted(CONTROL.glob('gpu_*.json'))]);atomic(CONTROL/'RESOURCES.json',resources)
 msg=f"{result['decision']['verdict']}；GPU {used/3600:.4f}h（含失败/中断段），新增{size/1e9:.3f}GB；D0/D1各5000步。";report('## 资源与最终状态\n'+json.dumps(resources,ensure_ascii=False)+'\n\n状态：COMPLETE。'+msg+'\n未完成必需项：无。扩展+100仅208窗/26parent，其余扩展时域211窗/27parent；不补造缺帧。动作真值为既有Piper next-frame follower反馈，不声称发送命令或实机成功。单seed、固定主集6parent；无测试集选型。无需用户新增决定。');shared('COMPLETE',msg);state('complete',decision=result['decision'],gpu_hours=used/3600,new_bytes=size)
if __name__=='__main__':
 try:main()
 except BaseException:
  err=traceback.format_exc();state('stopped',error=err,gpu_seconds=used_seconds(),new_bytes=disk());report('## 管线02停止（不得作增益结论）\n```\n'+err+'\n```\n必需项未完成，见PIPELINE.json；GPU段全部计入。');shared('STOPPED','必需步骤未全部完成，不能判定增益；详见PIPELINE.json。');raise
