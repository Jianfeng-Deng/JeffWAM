import os,time,json,subprocess
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e97a_robot_env_20261114_01'
start=time.monotonic()
while not (R/'FIT_LOCK.json').exists() or not (R/'calibration_inventory.json').exists():
 if (R/'failure.json').exists():raise RuntimeError('fit failure; do not evaluate')
 if time.monotonic()-start>1800:raise RuntimeError('waiter timed out')
 time.sleep(5)
code=subprocess.call(['/mnt/workspace/dengjianfeng/venvs/e29/bin/python','-u',str(T/'code_e97a/e97a_evaluate.py')],env=os.environ.copy())
(R/'waiter_status.json').write_text(json.dumps(dict(status='completed' if code==0 else 'stopped',returncode=code)))
