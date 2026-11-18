import time,subprocess,os,json,traceback
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e98w_control_20261114_01';C=T/'code_e98w';PY='/mnt/workspace/dengjianfeng/venvs/e29/bin/python'
def state(**kw):(R/'waiter_status.json').write_text(json.dumps(dict(brt=time.strftime('%Y-%m-%dT%H:%M:%S%z',time.localtime()),**kw),indent=2))
try:
 os.environ['TZ']='America/Sao_Paulo';time.tzset();start=time.monotonic();state(status='waiting_fixed_endpoints')
 while not (R/'TRAIN_COMPLETE.json').exists():
  st=json.loads((R/'STATUS.json').read_text())
  if st.get('status')=='stopped':raise RuntimeError('Training stopped; no evaluation authorized from incomplete checkpoints')
  if time.monotonic()-start>7300:raise RuntimeError('Training wait deadline')
  time.sleep(5)
 state(status='evaluating_fixed_endpoints')
 with (R/'evaluation_01.log').open('x') as f:subprocess.run([PY,'-u','e98w_evaluate.py'],cwd=C,stdout=f,stderr=subprocess.STDOUT,check=True)
 state(status='cpu_final_audit')
 with (R/'finalize_01.log').open('x') as f:subprocess.run([PY,'-u','e98w_finalize.py'],cwd=C,stdout=f,stderr=subprocess.STDOUT,check=True)
 state(status='completed')
except BaseException:
 state(status='stopped',traceback=traceback.format_exc());raise
