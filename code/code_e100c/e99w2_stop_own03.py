import os,signal,time,json,datetime
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');A=T/'results/e99w2_control_20261115_01';p=A/'gpu_ledger.json';d=json.loads(p.read_text());s=d['active'];pid=s['pid'];assert pid==2758573 and 'e99w2_train.py' in Path(f'/proc/{pid}/cmdline').read_text();os.kill(pid,signal.SIGTERM)
for _ in range(30):
 if not Path(f'/proc/{pid}').exists():break
 time.sleep(.2)
assert not Path(f'/proc/{pid}').exists()
d=json.loads(p.read_text());s=d.pop('active');now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3)));elapsed=max(s.get('elapsed_s',0),(now-datetime.datetime.fromisoformat(s['started_brt'])).total_seconds())+30;s.update(elapsed_s=elapsed,finished_brt=now.isoformat(),error='Own early implementation stop: ensure shifted chain cannot even read beyond original H50 action plan; +30s cleanup charged');d['elapsed_s']+=elapsed;d['segments'].append(s);p.write_text(json.dumps(d,indent=2))
(A/'implementation_stop_03.json').write_text(json.dumps(s,indent=2))
with (T/'notes/e99w2_planned_traj_20261115.md').open('a') as f:f.write('\n### 失败与实现修正（无dev评分）\n前两轮CUDA不可用共3.4435秒；第三轮在导入torch前设置CUDA_VISIBLE_DEVICES后成功。第三轮主动早停，修正链式临时样本，确保连读取也只用原动作块后缀+末端保持，不读取越出原H50的动作标签。整段及30秒清理余量入账；新run04从原E98-W A重新开始。\n')
print('stopped own E99W2 only; charged',elapsed,'seconds')
