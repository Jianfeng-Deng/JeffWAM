import os,time,json,subprocess,traceback
from common import *
PY='/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python'
def main():
    atomic(P/'launcher_status.json',{'status':'waiting_gpu_or_2h_cpu','pid':os.getpid(),'timestamp_brt':brt()})
    while True:
        s=json.loads((P/'progress.json').read_text())
        if s.get('status')=='blocked_missing_qualified_pixel_gt':return
        if s.get('status')=='stopped_missing_split':return
        if s.get('split_audit_complete') and (P/'selection_full_v4.json').exists() and (P/'synthetic_checks.json').exists():
            gpu=s.get('gpu_admission',{})
            if gpu.get('eligible'):
                mode,device='full','cuda:0';env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['eligible'][0]),PYTHONDONTWRITEBYTECODE='1');break
            if gpu.get('cpu_fallback_due'):
                mode,device='cpu','cpu';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');break
        time.sleep(30)
    update(compute_started=True,compute_mode=mode,compute_device=device)
    append_report('## 计算启动\n\n'+brt()+' 选择 '+mode+' / '+device+'；GPU入场信息/连续空闲等待记录见wait_events.jsonl；独立进程再次核验入场。')
    with (P/('extract_'+mode+'.log')).open('a') as f:
        proc=subprocess.Popen([PY,'-u',str(C/'extract.py'),mode,device],env=env,cwd=C,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(P/'launcher_status.json',{'status':'extracting','pid':proc.pid,'launcher_pid':os.getpid(),'mode':mode,'device':device,'timestamp_brt':brt()})
        code=proc.wait()
    atomic(P/'launcher_status.json',{'status':'extracted' if code==0 else 'extraction_failed','returncode':code,'timestamp_brt':brt(),'mode':mode,'device':device})
    if code: update(status='extraction_failed');return
    update(status='features_ready_wait_gt')
if __name__=='__main__':
    try:main()
    except Exception:
        atomic(P/'launcher_failure.json',{'timestamp_brt':brt(),'traceback':traceback.format_exc()});raise
