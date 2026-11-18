import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import sys,subprocess,time,traceback,json
from common import *
PYTHON=sys.executable
previous=json.loads((P/'cpu_pipeline_status.json').read_text()) if (P/'cpu_pipeline_status.json').exists() else {}
if previous.get('status')=='failed':atomic(P/('cpu_pipeline_failure_'+str(previous['pid'])+'.json'),previous)
state={'started_brt':previous.get('started_brt',brt()),'resumed_brt':brt(),'pid':os.getpid(),'status':'running','completed':previous.get('completed',[]),'previous_attempt_pid':previous.get('pid')}

def save():atomic(P/'cpu_pipeline_status.json',state)
def run(script):
    if script in state['completed']:return
    state.update(current=script);save();update(status='cpu_'+script)
    with (P/(script+'.log')).open('a') as log:
        subprocess.run([PYTHON,'-u',str(C/(script+'.py'))],stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ)
    state['completed'].append(script);save()
try:
    save();run('language_audit');run('run_pixel')
    state.update(current='waiting_READY_features');save()
    wait=time.monotonic()
    while not (R/'features'/RUN/'frozen/READY_features.json').exists():
        if (P/'extract_failure.json').exists() or (P/'resource_stop.json').exists():raise RuntimeError('GPU extraction stopped; inspect resource/extract status')
        if time.monotonic()-wait>6*3600:raise RuntimeError('Feature readiness timeout; GPU budget protects background worker')
        time.sleep(10)
    run('run_motion');run('run_behavior')
    state.update(status='completed',finished_brt=brt());save();update(status='experiments_complete_await_summary',experiments_completed_brt=brt())
except BaseException:
    state.update(status='failed',finished_brt=brt(),traceback=traceback.format_exc());save();update(cpu_pipeline_failed=True);raise
