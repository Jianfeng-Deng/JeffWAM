from .common import *
import subprocess,traceback
PYTHON='/mnt/workspace/dengjianfeng/venvs/e29/bin/python'
def command(module,args=(),label=None,wait=True):
 label=label or module.rsplit('.',1)[-1];f=(R/(label+'.log')).open('x');p=subprocess.Popen([PYTHON,'-u','-m',module,*args],cwd=C,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=os.environ.copy());f.close();atomic(R/(label+'_process.json'),dict(pid=p.pid,command=[PYTHON,'-u','-m',module,*args],started_brt=brt()))
 if wait:
  code=p.wait()
  if code:raise RuntimeError(label+' failed, exit='+str(code))
 return p
try:
 atomic(R/'pipeline_status.json',dict(stage='waiting_extract',brt=brt()))
 while not (R/'extract_complete.json').exists():
  s=json.loads((R/'status.json').read_text())
  if s['stage'].startswith('stopped'):raise RuntimeError('extraction stopped: '+s.get('error',''))
  time.sleep(15)
 while json.loads((R/'gpu_ledger.json').read_text()).get('active'):time.sleep(2)
 command('e94b.stats',label='stats');command('e94b.validate',label='implementation_checks')
 ridge=command('e94b.ridge',['8'],label='ridge_8',wait=False)
 command('e94b.train',['b1'],label='b1_train')
 command('e94b.evaluate',['b1'],label='b1_eval')
 if ridge.wait():raise RuntimeError('ridge8 failed')
 command('e94b.ridge',['49'],label='ridge_49')
 # Earliest next READY poll is >=10min after this process starts. Once b1/ridge finish, poll at >=600s.
 started=time.monotonic();ready=T/'code_e94a/READY_readout.json'
 while not ready.exists():
  atomic(R/'pipeline_status.json',dict(stage='waiting_E94a_READY',wait_s=time.monotonic()-started,brt=brt()))
  if time.monotonic()-started>=14400:raise RuntimeError('E94a READY wait exceeded4h')
  time.sleep(600)
 atomic(R/'E94a_ready_observed.json',dict(path=str(ready),sha256=sha(ready),ready=json.loads(ready.read_text()),brt=brt()))
 # Manual code review/import adapter must be prepared before running b2; wait for our own reviewed marker.
 while not (C/'E94b_readout_import_reviewed.json').exists():
  if time.monotonic()-started>=14400:raise RuntimeError('readout integration exceeds4h wait')
  atomic(R/'pipeline_status.json',dict(stage='waiting_reviewed_readout_import',brt=brt()));time.sleep(15)
 command('e94b.train',['b2'],label='b2_train');command('e94b.evaluate',['b2'],label='b2_eval')
 atomic(R/'pipeline_status.json',dict(stage='completed_training_evaluation',brt=brt()))
except BaseException as e:
 atomic(R/'pipeline_status.json',dict(stage='stopped',error=str(e),traceback=traceback.format_exc(),brt=brt()));raise
