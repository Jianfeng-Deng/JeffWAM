import sys,subprocess,json,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e86.common import *
def main():
    status=C/'artifacts/driver_v3_status.json';assert not status.exists();atomic(status,dict(status='running',step='GPU_regression',updated_brt=brt()))
    commands=[['scripts/gpu_regression_v1.py'],['-m','e86.train','smoke_v2','--cache',str(C/'artifacts/features_v3'),'--clean-aux','.25'],['-m','e86.profile_train','smoke_v2'],['-m','e86.eval_cli','smoke_v2']]
    try:
        assert (C/'artifacts/dev_features_v1/READY_features.json').exists()
        for i,args in enumerate(commands):
            atomic(status,dict(status='running',step=i,command=args,updated_brt=brt()))
            with (C/'artifacts'/('driver_v3_step_'+str(i)+'.log')).open('x') as out:subprocess.run([sys.executable,'-u']+args,cwd=C,stdout=out,stderr=subprocess.STDOUT,check=True)
        atomic(status,dict(status='completed',formal_training_started=False,updated_brt=brt()))
    except BaseException as e:atomic(status,dict(status='stopped',reason=str(e),traceback=traceback.format_exc(),updated_brt=brt()));raise
if __name__=='__main__':main()
