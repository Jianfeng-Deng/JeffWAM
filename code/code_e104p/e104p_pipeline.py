import os,sys,time,json,subprocess,traceback,fcntl
from e103d_runtime import *
PY='/mnt/workspace/dengjianfeng/venvs/e29/bin/python'
def disk():return sum(p.stat().st_size for root in [ROOT,*T.glob('results/e104p_*')] for p in root.rglob('*') if p.is_file() and not p.is_symlink())
def shared(status,msg):
 for p,txt in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',f'\n## E104-P | {brt()} BRT | {status}\n目标：ABC+yam先外部预训练再全量Piper微调，A2固定基线；F2 Piper10000(5000=F0)、F1外部5000+Piper5000，seed90恒定lr3e-4连续AdamW。改动code_e104p；只读复用data/e103d_cache；目录results/e104p_control_20261115_03、e104p_f1_s90_20261115_03、e104p_f2_s90_20261115_03。命令setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e104p_pipeline.py。{msg}\n报告notes/e104p_pretrain_finetune_20261115.md。\n'),(T/'notes/worklog.md',f'\n- {brt()} BRT E104-P {status}：{msg}；报告notes/e104p_pretrain_finetune_20261115.md。\n')]:
  with p.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(txt);f.flush()
def state(status,**kw):atomic(CONTROL/'PIPELINE.json',dict(status=status,pid=os.getpid(),brt=brt(),GPUh_upper=(used_seconds()+30*len(list(CONTROL.glob('gpu_*.json'))))/3600,new_bytes=disk(),**kw))
def run(script,args,name,gpu=False):
 assert disk()<9_000_000_000;assert used_seconds()+30*len(list(CONTROL.glob('gpu_*.json')))<10800
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',E102_GROUP='A2',CUDA_VISIBLE_DEVICES=wait_gpu(6144) if gpu else '')
 state('running',script=script,args=args,gpu=env['CUDA_VISIBLE_DEVICES'])
 with (CONTROL/name).open('x') as f:subprocess.run([PY,'-u',script,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 assert not (CONTROL/'PIPELINE.json').exists();shared('PREPARING','新路径检查通过，报告骨架已建；预算3GPUh/10GB；预注册CPU核验后锁定，尚无本轮评分。');state('preflight');run('e104p_preflight.py',[],'preflight.log')
 for group in ['F2','F1']:run('e104p_train.py',[group],group+'_train.log',True)
 a=json.loads((T/'results/e104p_f2_s90_20261115_03/summary_05000.json').read_text());b=json.loads((T/'results/e104p_f1_s90_20261115_03/summary_05000.json').read_text());checks={k:a[k]==b[k] for k in ['P3_samples_sha256','P1_samples_sha256','P3_flow_sha256','modes','P3_draws','P1_draws']};assert all(checks.values()),checks;atomic(CONTROL/'PAIRED_STREAM_AUDIT.json',dict(status='passed',checks=checks,F0=a,F1=b));report('## 同Piper数据逐样本核验\n'+json.dumps(checks,ensure_ascii=False))
 for group in ['A2','F1-pre','F0','F1','F2']:run('e104p_eval.py',[group],group+'_eval.log',True)
 run('e104p_finalize.py',[],'finalize.log');res=json.loads((CONTROL/'DECISION.json').read_text());upper=(used_seconds()+30*len(list(CONTROL.glob('gpu_*.json'))))/3600;assert upper<=3 and disk()<=10_000_000_000
 report('## 资源与最终状态\n'+json.dumps(dict(status='COMPLETE',brt=brt(),GPUh_upper=upper,new_bytes=disk(),segments=[json.loads(p.read_text()) for p in CONTROL.glob('gpu_*.json')],checkpoint_cache_old_code_readonly=True),ensure_ascii=False)+'\n未完成必需项：无。需要用户决定：无。外部有效性严格按预注册判定；不部署。');shared('COMPLETE',res['verdict']+f'；GPUh上界{upper:.4f}；新增{disk()/1e9:.3f}GB；全部固定终点训练/评估完成。');state('complete',decision=res)
if __name__=='__main__':
 try:main()
 except BaseException:
  err=traceback.format_exc();state('stopped',error=err);report('## 管线停止\n```\n'+err+'\n```\n必需步骤未全部完成，不作外部有效结论。失败计入GPUh，旧输出保留。');shared('STOPPED','必需项未全部完成，详见PIPELINE.json；不作增益结论。');raise
