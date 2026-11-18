from e103d_runtime import *
import shutil,subprocess,ast
assert not (CONTROL/'PIPELINE02.json').exists();p=ROOT/'e102x_setup.py';old=CONTROL/'e102x_setup_locked_initial.py';assert not old.exists();shutil.copyfile(p,old);p.write_text(p.read_text().replace("SRC=T/'code_e106c'","SRC=T/'code_e107m'"))
lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());lock.setdefault('execution_corrections',[]).append(dict(brt=brt(),file=p.name,before=lock['source_sha256'][p.name],after=sha(p),reason='Copied extended-dev loader SRC still pointed to old module path and failed contract lookup before CUDA; bind copied new source. No model/data/metric/threshold/checkpoint change.',failed_eval='eval_BASE',failed_GPU_seconds=0,new_eval_suffix='_02'))
lock['source_sha256'][p.name]=sha(p)
for n in ('e107m_eval02.py','e107m_analyze02.py','e107m_pipeline02.py'):
 ast.parse((ROOT/n).read_text());lock['source_sha256'][n]=sha(ROOT/n)
atomic(CONTROL/'PRETRAIN_LOCK.json',lock);atomic(CONTROL/'FAILED_EVAL01.json',lock['execution_corrections'][-1]);report('## 评估启动故障与新run02（0 GPU秒）\n'+json.dumps(lock['execution_corrections'][-1],ensure_ascii=False)+'\n保留eval_BASE/evaluator.py与失败日志；不重训、不覆盖旧结果。有效评估统一eval_BASE_02/eval_M0_02/eval_M1_02，之后CPU汇总。\n')
with (CONTROL/'pipeline02.log').open('x') as f:
 child=subprocess.Popen(['nohup','/mnt/workspace/dengjianfeng/venvs/e29/bin/python','-u','e107m_pipeline02.py'],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('pipeline02 PID',child.pid)
