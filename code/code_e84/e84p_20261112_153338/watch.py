import subprocess, time, json, hashlib
from common import *
state=json.loads((P/'progress.json').read_text()); start=state['started_utc']
split_deadline=start+2*3600; gt_deadline=start+6*3600
log=P/'wait_events.jsonl'; next_split=0; next_gt=0; next_gpu=0
split=bool(state.get('split_ready')); gt=bool(state.get('fk_gt_ready')); no_gpu_start=start; free_seen=False
if log.exists():
    for line in log.read_text().splitlines():
        e=json.loads(line);stamp=datetime.datetime.fromisoformat(e['timestamp_brt']).timestamp()
        if e['event']=='split_check':next_split=stamp+300
        if e['event']=='gt_check':next_gt=stamp+600
        if e['event']=='gpu_check':
            next_gpu=stamp+300
            if e['eligible']:no_gpu_start=None
            elif no_gpu_start is None:no_gpu_start=stamp
def event(obj):
    obj['timestamp_brt']=brt()
    with log.open('a') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
event({'event':'watcher_started','pid':os.getpid(),'split_interval_s':300,'gt_interval_s':600})
while True:
    now=time.time()
    if not split and now>=next_split:
        path=R/'READY_split.json'; exists=path.is_file()
        event({'event':'split_check','exists':exists})
        if exists:
            digest=hashlib.sha256(path.read_bytes()).hexdigest(); split=True
            update(split_ready={'path':str(path),'sha256':digest,'detected_brt':brt()},status='split_ready_waiting_operator_schema_audit')
            append_report('## Split就绪\n\n'+brt()+' 检测到READY_split.json，SHA256='+digest+'。待校验F合同后读样本。')
        elif now>=split_deadline:
            update(status='stopped_missing_split',blocker='READY_split.json absent after 2h; no samples read')
            append_report('## 等待停止\n\n'+brt()+' READY_split.json等待已达2h；不读样本，停止依赖工作。缺少split锁定交付。')
            break
        next_split=now+300
    if not gt and now>=next_gt:
        path=R/'READY_fk_gt.json'; exists=path.is_file()
        event({'event':'gt_check','exists':exists})
        if exists:
            digest=hashlib.sha256(path.read_bytes()).hexdigest(); gt=True
            update(fk_gt_ready={'path':str(path),'sha256':digest,'detected_brt':brt()})
            append_report('## FK/GT就绪\n\n'+brt()+' 检测到READY_fk_gt.json，SHA256='+digest+'。待校验真值/目视抽查与适用范围。')
        elif now>=gt_deadline:
            update(status='stopped_missing_fk_gt',blocker='READY_fk_gt.json absent after 6h')
            append_report('## FK/GT等待停止\n\n'+brt()+' READY_fk_gt.json等待已达6h；无法报告有效对齐指标。')
            break
        next_gt=now+600
    if now>=next_gpu:
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,check=True)
            rows=[[int(x.strip()) for x in line.split(',')] for line in result.stdout.splitlines()]
            free=[row[0] for row in rows if row[1]<512 and row[2]<5]
            event({'event':'gpu_check','gpus':rows,'eligible':free})
            if free: no_gpu_start=None; free_seen=True
            elif no_gpu_start is None: no_gpu_start=now
            update(gpu_admission={'last_brt':brt(),'eligible':free,'continuous_no_free_s':0 if no_gpu_start is None else now-no_gpu_start,'cpu_fallback_due':no_gpu_start is not None and now-no_gpu_start>=7200})
        except Exception as e: event({'event':'gpu_query_error','error':str(e)})
        next_gpu=now+300
    if split and gt and json.loads((P/'progress.json').read_text()).get('compute_started'):
        event({'event':'all_ready_compute_started'}); break
    time.sleep(min(30,max(1,min(next_split if not split else gt_deadline,next_gt if not gt else gt_deadline,next_gpu)-time.time())))
event({'event':'watcher_exit'})
