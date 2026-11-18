from common import *
import traceback

ATTEMPT=os.environ.get('E99B3_ATTEMPT','01')
def run(script,*args):
 log=R/(script.replace('.py','')+'_'+('_'.join(args) if args else 'run')+'_'+ATTEMPT+'.log')
 with log.open('x') as f:
  p=subprocess.run([sys.executable,'-u',str(T/'code_e99b3'/script),*args],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ))
 if p.returncode:raise RuntimeError(f'{script} {args} exit {p.returncode}; see {log.name}')
try:
 status('pipeline_waiting_train_features')
 while True:
  ledger=read('gpu_ledger.json')
  if ledger and ledger[0]['status']=='closed':
   if not (R/'O_contract_train.json').exists():raise RuntimeError('train feature process stopped before O contract')
   break
  time.sleep(5)
 run('contracts.py');run('vlm.py','smoke');run('heads.py','fit');run('controller.py','lock')
 with (R/'DEV_ONCE_LOCK.json').open('x') as f:json.dump(dict(brt=now(),heads=read('HEADS_LOCK.json'),b4_sha256=sha(R/'B4_LOCK.json'),vlm_smoke_sha256=sha(R/'VLM_SMOKE.json'),protocol_sha256=sha(R/'PROTOCOL_LOCK.json'),all_config_locked=True),f,ensure_ascii=False,indent=2)
 report('## dev 锁定开始 '+now()+'\nDEV_ONCE_LOCK.json 已冻结 heads、B4、提示模板及配置；只运行一次各方法前向，后续表格/CI只读冻结分数。')
 run('features.py','dev');run('heads.py','dev');run('vlm.py','dev');run('metrics.py');run('controller.py','evaluate')
 put('PIPELINE_COMPLETE.json',dict(brt=now(),status='awaiting_final_audit'));status('pipeline_complete_awaiting_final_audit')
except BaseException:
 put('PIPELINE_FAILURE_'+ATTEMPT+'.json',dict(brt=now(),traceback=traceback.format_exc()));status('pipeline_failed');report('## 执行异常 '+now()+'\n'+traceback.format_exc());raise
