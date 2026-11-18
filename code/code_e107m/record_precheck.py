from pathlib import Path
import fcntl,datetime,json
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
with (T/'notes/e107m_multimodal_head_20261116.md').open('a') as f:f.write('\n## CPU预检故障记录（0 GPUh）\ncpu_test01/02/03在z=0比较退出；两模型参数与buffer一致，但CPU冻结与非冻结参数计算路径造成4.35114e-6差异。diagnose.py确认各模型自身重复逐位一致，冻结原参考模型后跨模型逐位一致。cpu_test04统一冻结推理状态重跑。没有GPU初始化，没有训练更新到正式模型，所有日志保留。\n')
for path,text in [(T/'docs/experiment_log/EXPERIMENT_LOG.md','\n## E107-M | '+stamp+' BRT\n目标：世界头M0/M1多峰小实验；目录code_e107m、results/e107m_control_20261116_01；起点E106C固定20000；固定3000步；CPU测试前置，GPU≤2h/新增≤10GB。命令：e107m_test.py（setsid nohup分离）；状态：PRECHECK，GPU未启动。报告notes/e107m_multimodal_head_20261116.md。\n'),(T/'notes/worklog.md','\n- '+stamp+' BRT E107-M PRECHECK：新目录已创建；M0/M1固定3000步，CPU测试前置；报告 notes/e107m_multimodal_head_20261116.md。\n')]:
 with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text)
p=T/'code_e107m/e103d_runtime.py';s=p.read_text().replace("T/'results/e107m_m0_s90_20261116_01')","T/'results/e107m_m0_s90_20261116_01',T/'results/e107m_m1_s90_20261116_01')");p.write_text(s)
print('appended logs; guard counts both runs')
