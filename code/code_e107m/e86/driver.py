"""Wait for this session's cache, then bounded dev extraction/smoke/profile/latency only."""
import time,sys,json,subprocess,os,traceback
from .common import *
def main():
    status=C/'artifacts/driver_v2_status.json';started=time.monotonic();run='smoke_v1'
    try:
        while True:
            cache=C/'artifacts/features_v3';state=json.loads((cache/'status.json').read_text()) if (cache/'status.json').exists() else {}
            if state.get('status')=='stopped':raise RuntimeError('Feature cache stopped: '+state.get('reason',''))
            if (cache/'READY_features.json').exists() and (C/'artifacts/READY_windows_v3.json').exists() and (C/'artifacts/language_v1/READY_language.json').exists():break
            if time.monotonic()-started>6600:raise RuntimeError('Timed out waiting for E86 prerequisites')
            atomic(status,dict(status='waiting_own_prerequisites',cache_captures=state.get('completed_captures'),updated_brt=brt()));time.sleep(20)
        time.sleep(2) # extractor exits Guard immediately after READY
        commands=[['-m','e86.cache_fast','artifacts/dev_v1/cache_plan.json','artifacts/dev_features_v1'],['-m','e86.train',run,'--cache',str(cache),'--clean-aux','.25'],['-m','e86.profile_train',run],['-m','e86.eval_cli',run]]
        for i,args in enumerate(commands):
            atomic(status,dict(status='running',step=i,command=args,updated_brt=brt()))
            with (C/'artifacts'/('driver_v2_step_'+str(i)+'.log')).open('x') as out:subprocess.run([sys.executable,'-u']+args,cwd=C,stdout=out,stderr=subprocess.STDOUT,check=True)
        atomic(status,dict(status='completed',formal_training_started=False,elapsed_s=time.monotonic()-started,updated_brt=brt()))
    except BaseException as e:atomic(status,dict(status='stopped',reason=str(e),traceback=traceback.format_exc(),updated_brt=brt()));raise
if __name__=='__main__':main()
