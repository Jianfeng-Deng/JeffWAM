from pathlib import Path
import datetime,fcntl,os
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
text=f'\n## E101-S | {now} | RUNNING\n- 目标：用训练parent袋外谓词分数调时间平滑阶段控制器，dev一次冻结评估。\n- 改动：新建code_e101s、results/e101s_20261115_01与notes/e101s_stage_controller_20261115.md；旧代码/标签/结果只读。\n- 命令：ssh -o BatchMode=yes <gpu-server>；CUDA_VISIBLE_DEVICES为空、PYTHONDONTWRITEBYTECODE=1、setsid nohup e29/bin/python -u code_e101s/run.py fit。\n- 当前：60 train parent 5折b/d谓词分数完成；仅12训练parent具全帧缓存用于调参。GPUh=0。首段unknown阶段映射异常已修正并保留日志，无dev评分。\n- 结果/结论：待冻结dev；状态RUNNING。\n'
p=T/'docs/experiment_log/EXPERIMENT_LOG.md'
with p.open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);f.write(text);f.flush();os.fsync(f.fileno())
print(now)
