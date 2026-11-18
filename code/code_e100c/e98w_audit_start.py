from pathlib import Path
import hashlib,json,fcntl
from datetime import datetime,timezone,timedelta
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e98w_control_20261114_01'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
paths=[]
for name in ('code_e96a','code_e97a'):
 paths += [p for p in (T/name).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
paths += [T/'results/e96a_m1_p1_20261114_01/ckpt_05000.pt']
paths += list((T/'results/e97a_robot_env_20261114_01').glob('R[012]_h*.npz'))
paths += [T/'results/e93_dev_diag_20261114_01'/n for n in ('dev/windows_full.json','dev/captures_full.json','train64_selection.json')]
assert not (R/'protected_inputs.json').exists()
(R/'protected_inputs.json').write_text(json.dumps({str(p):dict(sha256=sha(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths},indent=2))
now=datetime.now(timezone(timedelta(hours=-3))).isoformat()
entry=f'''\n## E98-W — {now} BRT — running\n- 目标：E97-A1 §11 的 A/B：未来机器人轨迹条件 vs 显式 R1 剩余目标；仅 Piper。\n- 改动：独占 code_e98w（由 code_e96a 小源码复制），101 原行 oracle q/FK token 只供 W；A/B 同参数、seed90、batch32、AdamW3e-4、固定5000步。B 用5折parent交叉拟合R1，dev加回E97-A1训练全量冻结R1，保留原目标尺度。\n- 命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .../venvs/e29/bin/python -u code_e98w/e98w_prepare.py / e98w_run.py。\n- 目录：results/e98w_control_20261114_01；正式候选e98w_a/b_20261114_03；失败准入_01和20步profile_02保留；notes/e98w_robot_conditioned_20261114.md。\n- 结果：CPU合同通过，旧参数初始化逐位一致、O条件隔离0差，R1复现最大1.2e-7；dev未前向；前20步profile预算停止41.19秒入账。\n- 结论/状态：训练内状态/FK预热后继续预算内profile，禁止dev选模型；GPU总额7200秒、新增≤8GB。\n'''
for path,text in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',entry),(T/'notes/worklog.md',f'\n- {now} BRT | E98-W | 实现oracle轨迹条件A/B、5折parent R1、合同通过；训练中，预算≤2GPUh/8GB；见 notes/e98w_robot_conditioned_20261114.md。\n')]:
 with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text)
print(json.dumps(dict(protected_files=len(paths),status='logged_running',brt=now)))
