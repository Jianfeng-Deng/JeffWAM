"""Append the E82-M delivery record once; only authorized project-owned paths."""
from pathlib import Path
import datetime,fcntl,hashlib,json
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu')
code=T/'code_e82'; report=T/'notes/e82_M_model_pipeline_20261112.md'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
s=report.read_text()
s=s.replace('开始：10:19 BRT。','开始：10:19 BRT；最终审计：'+now+'。')
s=s.replace('待填写；正式训练禁止执行。','已完成代码交付；正式训练未执行，真实GPU profile资源阻塞，见§9–10。')
s=s.replace('待填写。','已在下方对应进展与§7–10完成记录。')
s=s.replace('### 进展 A：来源与资源（10:26 BRT）','### 进展 A：来源与资源（10:21 BRT）')
s=s.replace('首次 GPU 快照所有卡显存均被占用，未用 GPU','首次 GPU 快照所有卡显存均被占用，未做CUDA模型计算（早期RNG上下文风险见§9.1）')
s=s.replace('CPU，未使用 GPU。','CPU模型计算，资源审计偏差见§9.1。')
s=s.replace('profile单卡桶.95 GPUh','profile单卡桶.4 GPUh')
report.write_text(s)
readme=code/'README.md'
s=readme.read_text()
s+='\n最终验证（2026-11-12 BRT）：29项测试全部通过；2,476个READY文件哈希验证与真实生产尺寸DINO CPU前向通过，缓存误差0。真实GPU profile因8卡均占用未运行，48 GPUh与P99仍未实测；P3_ready=0、腕ROI未审，正式训练未启动。见 T/notes/e82_M_model_pipeline_20261112.md。\n'
if '最终验证（2026-11-12 BRT）' not in readme.read_text():readme.write_text(s)
files={}
for p in sorted(code.rglob('*')):
 if p.is_file() and not any(x in ('artifacts','__pycache__','.pytest_cache') for x in p.relative_to(code).parts):
  files[str(p.relative_to(code))]=hashlib.sha256(p.read_bytes()).hexdigest()
implementation=dict(id='E82-M',brt=now,source_files=files,report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),tests='29 passed, 1 warning in 21.38s',formal_training_started=False,real_GPU_profile='blocked_no_idle_gpu',data_ready_sha256='18b5d27d047f670bc2d2af09dbf68ac7051fde4749c633028c9a3ec2ce8e9844')
(code/'artifacts/implementation_release.json').write_text(json.dumps(implementation,indent=2,ensure_ascii=False))
log=T/'docs/experiment_log/EXPERIMENT_LOG.md'
entry=f'''\n\n### E82-M | Piper统一模型与训练评估流水线 | 2026-11-12 BRT\n\n- 审计时间：{now}\n- 目标：E79/E81 Piper O/A/W统一flow及P1→P3/R→P3流水线；本轮仅合同/小样本/profile，禁止正式训练。\n- 改动：新建code_e82（core、原始本地DINO双塔/LoRA、Piper adapter、损失/噪声流、部署两路径、manifest/实际D数据入口、配置、训练恢复止损、网格、预算、时延、父等权/簇bootstrap、链式启动脚本）；不改旧工具。\n- 命令：scripts/run_contracts.sh；python -m compileall -q e82 tests；scripts/fixture_smoke.sh；scripts/real_cpu_smoke.py；scripts/run_bounded_profile.sh（无空闲卡退出75）。\n- 目录：T/code_e82/；报告T/notes/e82_M_model_pipeline_20261112.md；证据code_e82/artifacts/implementation_release.json、contracts_release_final.txt、real_cpu_forward_v1.json、gpu_accounting.json。\n- 结果：29 tests通过，core4,739,072、P3可训5,618,446、LoRA49,152；O187/A50/W273；2476文件hash通过；真实DINO生产尺寸CPU缓存误差0，cuda_initialized=false；合成P1/P3/resume与100更新fixture profile通过。\n- 偏差：早期CPU检查点CUDA RNG读取可能初始化上下文，已修复并加CPU禁CUDA回归；无CUDA模型计算。按全部相关CPU墙钟×8卡保守上下文GPUh上界0.458525，真实profile桶降.4，合计上界.858525<1。不能宣称严格0上下文。\n- 结论：真实数据已接线；8卡均占用，真实100步GPU profile/阶段GPUh与P99未测，48h可行性未决；P3_ready=0、ROI待人工审核，正式gate拒绝放行。\n- 状态：DELIVERED_CODE_CPU_VERIFIED / GPU_PROFILE_BLOCKED；未启动正式训练。需安排空闲H20短时窗口；天机/R0按用户要求不做。\n'''
with log.open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
 if '### E82-M | Piper统一模型与训练评估流水线 | 2026-11-12 BRT' not in f.read():f.write(entry)
line=f'\n- {now} | E82-M | 新建code_e82：29合同测试/真实DINO CPU接入通过（O187/A50/W273，core4.739M）；真实GPU profile无空闲卡阻塞、48GPUh/P99未决；早期CUDA RNG潜在上下文偏差已修复并记保守上界；正式训练未启动。报告 notes/e82_M_model_pipeline_20261112.md。\n'
with (T/'notes/worklog.md').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
 if '| E82-M | 新建code_e82：29合同测试' not in f.read():f.write(line)
print(json.dumps(dict(brt=now,source_files=len(files),report=str(report),status='delivered_GPU_profile_blocked'),ensure_ascii=False))
