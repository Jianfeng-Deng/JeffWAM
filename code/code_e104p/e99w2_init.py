from pathlib import Path
from datetime import datetime,timezone,timedelta
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu'); now=datetime.now(timezone(timedelta(hours=-3))).isoformat()
p=T/'notes/e99w2_planned_traj_20261115.md'
with p.open('x') as f:f.write(f'''# E99-W2 动作块计划轨迹条件
开始：{now} BRT。状态：动作合同核查与实现中。
独占code_e99w2、results/e99w2_*；旧包与数据只读；无仿真/retarget/下载；永久排除ep142；demo3累计前缀≤45%。AGENTS旧天机54D规定由本轮明确Piper14D授权覆盖，不启动旧planner。

## 1. 预注册（结果前锁定）
E98-W A固定5000初始化；P3_ready_V3；seed90、5000步终点、batch32、AdamW lr3e-4；sync/independent/action_only/clean各25%；确定性均值世界头、空间旁路与原链日程/适配器门。不按dev选型。
同一终点仅切世界轨迹来源：oracle、GT action、pred action N1和N8世界预测均值、no-traj、GT加同幅噪声。固定4 Euler、seed90起；parent bootstrap2000 seed90。N1相对no-traj在+25/+50/+100 skill改善95%CI下界全部>0判为可部署条件有效。保留率=(pred−no)/(oracle−no)；分母≤0则NA，原差值仍报告。
H50之外计划扩展在训练与评分前锁定，不读取真实未来延长计划。eval禁MHA fastpath并拒绝非有限值。dev只用既有P3 V3固定窗口/人工短指令；模型标签不作测试真值/阈值/选点。
GPU≤4GPUh含失败，新增≤10GB；free≥估算峰值×1.5才准入；OOM/显存压力立即停止；分离后台任务与状态文件。

## 2. 动作合同与时间对齐
待完成。

## 3. 实现与验证
待完成。

## 4. 训练与资源
待完成。

## 5. 固定评估与判定
待完成。

## 6. 命令、结论、未完成与决策
待完成。
''')
for path,line in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',f'\n## E99-W2 | {now} BRT\n目标：动作计划条件下P3完整v3.1。改动：新code_e99w2；目录results/e99w2_*；命令/结果/结论见notes/e99w2_planned_traj_20261115.md；状态进行中。\n'),(T/'notes/worklog.md',f'\n- {now} BRT E99-W2：P3完整v3.1计划轨迹条件，进行中；notes/e99w2_planned_traj_20261115.md。\n')]:
 with path.open('a') as f:
  import fcntl
  fcntl.flock(f,fcntl.LOCK_EX);f.write(line)
