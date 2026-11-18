#!/usr/bin/env python3
"""Read READY every 10 minutes, for <=4h; never train or allocate a GPU."""
import datetime,json,os,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
state=root/'artifacts'/'data_wait.json'
start=time.time(); deadline=start+4*3600
while True:
    data=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1')
    found={n:(data/n).is_file() for n in ('READY_parent_split.json','READY_data.json')}
    status='ready' if found['READY_data.json'] else ('timeout' if time.time()>=deadline else 'waiting')
    record=dict(status=status,start_unix=start,deadline_unix=deadline,checked_brt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat(),**found)
    tmp=state.with_suffix('.tmp'); tmp.write_text(json.dumps(record,indent=2)); os.replace(tmp,state)
    with (state.parent/'data_wait_checks.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    if status!='waiting': break
    time.sleep(min(600,max(0,deadline-time.time())))
