from e103d_runtime import *
import subprocess,ast
assert json.loads((CONTROL/'OPENLOOP_DTYPE_TEST.json').read_text())['status']=='passed';assert not (CONTROL/'PIPELINE03.json').exists()
lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());fix=dict(brt=brt(),reason='E93 extended-dev plan_action float64 promoted open_second trajectory to float64; cast only robot condition to model float32 at projection input. Physical plan values/indices/labels and metric definitions unchanged.',failed_eval='eval_BASE_02',GPU_seconds=json.loads((CONTROL/'gpu_eval_BASE_02.json').read_text())['elapsed_s'],new_eval_suffix='_03',CPU_contract_test='OPENLOOP_DTYPE_TEST.json')
lock.setdefault('execution_corrections',[]).append(fix)
for n in ('e107m_eval03.py','e107m_analyze03.py','e107m_pipeline03.py','e107m_test_openloop.py'):
 ast.parse((ROOT/n).read_text());lock['source_sha256'][n]=sha(ROOT/n)
atomic(CONTROL/'PRETRAIN_LOCK.json',lock);atomic(CONTROL/'FAILED_EVAL02.json',fix);report('## 开环dtype故障与新run03\n'+json.dumps(fix,ensure_ascii=False)+'\nrun02已完成211窗单跳，但未保存最终评估，不作结论；失败GPU段已计入。BASE/M1在k25/k50的CPU第二跳模型接口与有限值测试通过。保存原eval_BASE_02与日志，三组统一重新执行eval_*_03；不重训、不改终点、阈值或判据。\n')
with (CONTROL/'pipeline03.log').open('x') as f:
 child=subprocess.Popen(['nohup','/mnt/workspace/dengjianfeng/venvs/e29/bin/python','-u','e107m_pipeline03.py'],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
atomic(CONTROL/'LAUNCH03.json',dict(pid=child.pid,brt=brt(),GPUh_prior=used_seconds()/3600));print('pipeline03 PID',child.pid)
