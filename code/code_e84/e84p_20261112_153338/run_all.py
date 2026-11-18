import os,sys,time,json,subprocess,pathlib,traceback
from common import *
PY='/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python'
def run(script,*args,env=None):
    name=script.replace('.py','')+'_'+('_'.join(args) or 'default');log=P/(name+'.log')
    with log.open('a') as f:
        p=subprocess.Popen([PY,'-u',str(C/script),*args],cwd=C,env=env or dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(P/'evaluation_job.json',{'script':script,'args':args,'pid':p.pid,'timestamp_brt':brt(),'status':'running'})
        code=p.wait()
    atomic(P/'evaluation_job.json',{'script':script,'args':args,'pid':p.pid,'timestamp_brt':brt(),'status':'completed' if code==0 else 'failed','exit_code':code})
    if code:raise RuntimeError(f'{script} exited {code}; see {log}')
def trigger_lora():
    reasons=[]
    for arm in ['left','right']:
        p=P/'frozen'/f'motion_robot_{arm}_h50.json'
        if not p.exists():continue
        r=json.loads(p.read_text())
        if r.get('status')!='completed':continue
        zero=r['zero_shot']['zh']['difference']['direction_top1'];linear=r['learned']['linear']['metrics']['direction_top1'];random=r['baseline']['random']['direction_top1']
        if zero['estimate']<=.25 and linear['ci95'][0]>random['ci95'][1]:reasons.append({'arm':arm,'zero_direction':zero,'linear_direction':linear,'random_direction':random})
    return reasons
def device_for_lora():
    s=json.loads((P/'progress.json').read_text())
    if s.get('compute_mode')=='cpu':return 'cpu',dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    started=time.monotonic()
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        free=[r[0] for r in [[int(x.strip()) for x in line.split(',')] for line in rows.splitlines()] if r[1]<512 and r[2]<5]
        if free:return 'cuda:0',dict(os.environ,CUDA_VISIBLE_DEVICES=str(free[0]),PYTHONDONTWRITEBYTECODE='1')
        if time.monotonic()-started>=7200:
            update(gpu_admission={'cpu_fallback_due':True,'continuous_no_free_s':time.monotonic()-started,'eligible':[],'last_brt':brt()})
            return 'cpu',dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        update(status='waiting_lora_gpu',lora_gpu_wait_s=time.monotonic()-started)
        for _ in range(10):time.sleep(30)
def main():
    update(evaluation_runner_pid=os.getpid())
    while True:
        s=json.loads((P/'progress.json').read_text())
        if s.get('status')=='blocked_missing_qualified_pixel_gt':return
        if s.get('status') in ['stopped_missing_split','stopped_missing_fk_gt','extraction_failed']:raise RuntimeError(s.get('blocker',s['status']))
        if (R/'features'/RUN/'frozen/READY_features.json').exists() and s.get('fk_gt_audit_complete') and (P/'synthetic_learning_checks.json').exists() and (P/'synthetic_lora_check.json').exists():break
        time.sleep(30)
    for script in ['run_localization.py','run_motion.py','run_behavior.py']:
        update(status='running_'+script.replace('.py',''));run(script,'frozen')
    reasons=trigger_lora();atomic(P/'lora_trigger.json',{'triggered':bool(reasons),'reasons':reasons,'rule':'zero-shot own-base direction<=25%; linear probe CI above random CI; unsupported pixel GT never triggers LoRA','timestamp_brt':brt()})
    if reasons:
        append_report('## LoRA触发依据\n\n'+json.dumps(reasons,ensure_ascii=False)+'。依据FK运动的冻结零样本失败且线性可学；精确左右/锁头像素GT不足不用于触发。')
        update(status='lora_training');device,env=device_for_lora();run('train_lora.py',device,env=env)
        # Reacquire after training process exits; no use of an occupied device.
        if device.startswith('cuda'):device,env=device_for_lora()
        mode=json.loads((P/'progress.json').read_text())['compute_mode'];run('extract.py',mode,device,str(R/'lora'/RUN/'adapter.pt'),env=env)
        for script in ['run_localization.py','run_motion.py','run_behavior.py']:run(script,'lora')
        run('drift_and_figures.py')
    else:append_report('## LoRA未触发\n\n按预注册FK方向条件未触发；不根据未认证像素proxy决定微调。')
    run('summarize.py');update(status='completed_scoped_with_pixel_gt_blocker')
if __name__=='__main__':
    try:main()
    except Exception:
        atomic(P/'run_all_failure.json',{'timestamp_brt':brt(),'traceback':traceback.format_exc()});update(status='evaluation_runner_failed');raise
