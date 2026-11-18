from pathlib import Path
import json,shutil,datetime
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e98w_control_20261114_01'
assert not (T/'results/e98w_a_20261114_02').exists() and not (T/'results/e98w_b_20261114_02').exists()
shutil.copyfile(R/'STATUS.json',R/'admission_failure_01.json')
(R/'gpu_ledger.json').write_text(json.dumps(dict(schema='e98w-gpu-walltime-v1',budget_s=7200,elapsed_s=.7,segments=[dict(phase='admission_failure_01',elapsed_s=.7,GPU_initialized=False,reason='Inherited idle-utilization restriction rejected busy shared GPUs before CUDA; corrected to user free-memory rule; conservative whole attempt charged')]),indent=2))
with (T/'notes/e98w_robot_conditioned_20261114.md').open('a') as f:
 f.write('\n### GPU准入与判定解释补充（结果前）\n首次准入因沿用旧包util==0限制而在CUDA初始化前拒绝，0前向/0训练，保留e98w_a_20261114_01与admission_failure_01；保守计0.7秒。修复为用户规定的空闲显存≥12GiB估算峰值×1.5，允许共享Default模式卡；监控余量，压力/OOM立即停。新run为A/B _02。B显式分离判定中“完整skill不退化”采用保守充分证据：B−A完整skill配对CI下界≥0；各h同时报告点差与区间，禁止结果后重释。A/B从相同seed的E96a架构初始参数训练，基线只读既有5000终点，未用基线终点继续训练。\n')
print('relaunch_receipt_written')
