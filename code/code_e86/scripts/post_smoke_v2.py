import sys,os,json,time,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e86.common import *
def main():
    status=C/'artifacts/post_smoke_v2_status.json';assert not status.exists();atomic(status,dict(status='waiting_main',updated_brt=brt()))
    try:
        while True:
            d=json.loads((C/'artifacts/driver_v3_status.json').read_text())
            if d['status']=='stopped':raise RuntimeError('Main smoke/profile/latency driver stopped')
            if d['status']=='completed':break
            time.sleep(10)
        while json.loads((C/'artifacts/watchdog_v2_status.json').read_text()).get('status')=='monitoring':time.sleep(2)
        atomic(status,dict(status='running',step='external_resume',updated_brt=brt()))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',E86_DRIVER_STATUS=str(status),E86_WATCHDOG_STATUS=str(C/'artifacts/watchdog_v3_status.json'))
        with (C/'artifacts/watchdog_v3.log').open('x') as out:subprocess.Popen([sys.executable,'-u','-m','e86.watchdog'],cwd=C,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        commands=[['-c',"import runpy; import e86.common; import sys; sys.argv=['profile_fullpool_v1.py','smoke_v2']; runpy.run_path('scripts/profile_fullpool_v1.py',run_name='__main__')"],['-m','e86.resume','artifacts/smoke_v2/P3_resume.pt','resume_cli_smoke_v2','--extra-steps','1'],['-c',"import runpy; import e86.common; runpy.run_path('scripts/fixed_train_diag_v1.py',run_name='__main__')"]]
        for i,args in enumerate(commands):
            atomic(status,dict(status='running',step=i,command=args,updated_brt=brt()))
            with (C/'artifacts'/('post_smoke_v2_step_'+str(i)+'.log')).open('x') as out:subprocess.run([sys.executable,'-u']+args,cwd=C,stdout=out,stderr=subprocess.STDOUT,check=True)
        diag=json.loads((C/'artifacts/fixed_train_diag_v1.json').read_text());resume=json.loads((C/'artifacts/resume_cli_smoke_v2/status.json').read_text())
        append_report('### 独立进程恢复与固定训练batch诊断\n'+json.dumps(resume,ensure_ascii=False,indent=2)+'\n'+json.dumps(diag,ensure_ascii=False,indent=2)+'\n固定训练batch/同噪声仅检验数值优化，未用于checkpoint选型；所有阶段仍取预先固定终点。')
        with (C/'artifacts/final_audit_v1.log').open('x') as out:subprocess.run([sys.executable,'-u','-m','e86.audit'],cwd=C,stdout=out,stderr=subprocess.STDOUT,check=True)
        full=json.loads((C/'artifacts/smoke_v2/profile_fullpool_v1/READY_profile.json').read_text())
        append_report('### 正式GPUh估算：全池随机窗口（以本节为主）\n'+json.dumps(full,ensure_ascii=False,indent=2)+'\n以上使用全部可读P1 107057窗/P3 V1 24477窗按parent等权随机抽样、各100个实测batch32更新；前述小窗profile仅为诊断。正式训练尚未启动。')
        atomic(C/'artifacts/E86_M_FULLPOOL_PROFILE.json',dict(status='ready',profile_path=str(C/'artifacts/smoke_v2/profile_fullpool_v1/READY_profile.json'),profile_sha256=sha(C/'artifacts/smoke_v2/profile_fullpool_v1/READY_profile.json'),report_sha256=sha(T/'notes/e86_M_v3_smoke_20261113.md')))
        atomic(status,dict(status='completed',formal_training_started=False,updated_brt=brt()))
    except BaseException as e:atomic(status,dict(status='stopped',reason=str(e),traceback=traceback.format_exc(),updated_brt=brt()));raise
if __name__=='__main__':main()
