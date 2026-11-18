import os,sys,json,time,subprocess,traceback,fcntl,hashlib
from e103d_runtime import *
PY='/mnt/workspace/dengjianfeng/venvs/e29/bin/python'
def disk():
 roots=[ROOT,CACHE,CONTROL,*T.glob('results/e103d_d[01]_s90_20261115_01')]
 return sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file() and not p.is_symlink())
def state(status,**kw):atomic(CONTROL/'PIPELINE.json',dict(status=status,pid=os.getpid(),brt=brt(),**kw))
def run(script,args,logname,gpu=False):
 assert disk()<40e9;env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=wait_gpu() if gpu else '')
 if args and args[0] in ['D0','D1']:env['E103_GROUP']=args[0]
 state('running',script=script,args=args,gpu=env['CUDA_VISIBLE_DEVICES'],gpu_seconds=used_seconds())
 with (CONTROL/logname).open('x') as log:subprocess.run([PY,'-u',script,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def append_shared(status,msg):
 for p,txt in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',f'\n## E103-D | {brt()} BRT | {status}\n目标：ABC/yam 无动作视频对 Piper 世界预测增益。基线C2固定ckpt；新代码code_e103d。命令：setsid nohup e29/bin/python -u e103d_pipeline.py；缓存e103d_cache.py；训练e103d_train.py D0/D1；评估e103d_eval.py D0/D1/finish。目录results/e103d_control_20261115_01、results/e103d_d0_s90_20261115_01、results/e103d_d1_s90_20261115_01、data/e103d_cache。{msg}\n报告notes/e103d_external_data_20261115.md。\n'),(T/'notes/worklog.md',f'\n- {brt()} BRT E103-D {status}：{msg}；报告notes/e103d_external_data_20261115.md。\n')]:
  with p.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(txt);f.flush()
def main():
 assert not (CONTROL/'PIPELINE.json').exists();state('waiting_cache')
 while True:
  p=CONTROL/'cache_status.json';d=json.loads(p.read_text()) if p.exists() else {}
  if d.get('status')=='complete':break
  g=CONTROL/'gpu_cache_03.json'
  if g.exists():
   meta=json.loads(g.read_text());assert meta['status']=='active','Cache GPU phase stopped before completion'
   try:
    st=Path(f"/proc/{meta['pid']}/stat").read_text().split()[2];assert st!='Z'
   except FileNotFoundError:raise RuntimeError('Cache process exited without completion')
  assert disk()<40e9
  time.sleep(15)
 assert json.loads((CONTROL/'DATA_AUDIT.json').read_text())['status']=='passed';assert json.loads((CONTROL/'piper_replay.json').read_text())['max_abs']==0
 run('e103d_train.py',['contracts'],'contracts.log')
 from e99w2_common import INIT
 assert sha(INIT)=='a432001ffeb7d99fa2980ef337379464948d8c22678bb37dbfea0ab6708da99b'
 lock=dict(brt=brt(),initial_checkpoint=str(INIT),initial_sha256=sha(INIT),steps_each=5000,seed=90,lr=3e-4,schedule='constant; restore identical C2 optimizer; P3 global steps5001..10000, full C2 chain .1, predicted reentry probability .8',P3='Both arms identical Piper P3 32-window update every optimizer step, four modes equal probability; original sampler seed90 and demo3 cap .45',P1='Both arms additional batch32 no-robot/no-action deterministic world auxiliary, coefficient .1; four fixed horizons. D0 all Piper; D1 20 Piper +6 yam +6 ABC; missing source slots back to Piper. P1 has independent per-step RNG; P3 RNG unaffected by source substitution.',ratio_ambiguity_disclosure='C2 training code is P3-only, no historical standalone P1 ratio exists. Added paired P1 auxiliary to implement explicit user P1+P3 request; disclosed before training; optional clarification had no answer at implementation time.',P1_source_fractions_D1={'piper':.625,'yam':.1875,'abc':.1875},P1_sampler='Piper parent equal demo3 cap.45 seed10390; external task round robin, parent uniform, valid window uniform; source fixed slots. ABC 5 tasks, each20% external draws asymptotically; no single task >25% physical hours',text='Piper original train instruction/fallback; ABC/yam fixed generic Manipulate objects on a tabletop. No model narratives. dev fixed human default only.',world_loss='same normalized deterministic world MSE +.1 cosine +.1 reentry adapter; no external action/alignment/phase/chain loss',eval='Fixed Piper dev48 windows/6parent all4 horizons gt_action/no_traj/pred_N1/pred_N8; 4Euler N1/N8 seeds90..97; E93 extended cached valid dev no_traj; 2000 paired parent bootstrap seed90; fixed endpoints no checkpoint selection',criterion='D1-D0 no_traj +25/+50/+100 >=2 CI lower>0pp; all3 gt_action CI lower>-1pp; N1 joint_L1_rad and gripper_L1_mm at8/50 point relative degradation<=5% (all4). Otherwise 无增益.',plan_sha256=sha(CACHE/'plan.json'),cache_contract_sha256=sha(CACHE/'features/contract.json'),data_audit_sha256=sha(CONTROL/'DATA_AUDIT.json'),CPU_tests_sha256=sha(CONTROL/'CPU_TESTS.json'),code_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py'))},GPU_seconds_cap=21600,disk_bytes_cap=40000000000)
 assert not (CONTROL/'PRETRAIN_LOCK.json').exists();atomic(CONTROL/'PRETRAIN_LOCK.json',lock);report('## 结果前预注册锁定\n```json\n'+json.dumps({k:v for k,v in lock.items() if k!='code_hashes'},ensure_ascii=False)+'\n```')
 for group in ['D0','D1']:run('e103d_train.py',[group],group+'_train.log',True)
 ss=[json.loads((T/f'results/e103d_{g.lower()}_s90_20261115_01/train_summary.json').read_text()) for g in ['D0','D1']];assert ss[0]['P3_samples_sha256']==ss[1]['P3_samples_sha256'];assert ss[0]['modes']==ss[1]['modes'];assert all(x['steps']==5000 for x in ss)
 for group in ['D0','D1']:run('e103d_eval.py',[group],group+'_eval.log',True)
 run('e103d_eval.py',['finish'],'finalize.log');result=json.loads((CONTROL/'FINAL_RESULTS.json').read_text());used=used_seconds();size=disk();assert used<=21600 and size<=40e9;resources=dict(gpu_seconds=used,gpu_hours=used/3600,new_bytes=size,new_decimal_GB=size/1e9,segments=[json.loads(p.read_text()) for p in sorted(CONTROL.glob('gpu_*.json'))]);atomic(CONTROL/'RESOURCES.json',resources)
 msg=f"{result['decision']['verdict']}；GPU {used/3600:.4f}h（含失败段），新增{size/1e9:.3f}GB；D0/D1各5000步。";report('## 资源与最终状态\n'+json.dumps(resources,ensure_ascii=False)+'\n\n状态：COMPLETE。'+msg+'\n未完成必需项：无。无真实机器人执行，未做测试集选型；单seed与有限dev parent边界见上文。无需用户新增决定。');append_shared('COMPLETE',msg);state('complete',decision=result['decision'],gpu_hours=used/3600,new_bytes=size)
if __name__=='__main__':
 try:main()
 except BaseException:
  err=traceback.format_exc();state('stopped',error=err,gpu_seconds=used_seconds(),new_bytes=disk());report('## 管线停止（不得作增益结论）\n```\n'+err+'\n```\n必需项未完成，见PIPELINE.json；此前GPU段全部计入。');append_shared('STOPPED','必需步骤未全部完成，不能判定增益；详见PIPELINE.json。');raise
