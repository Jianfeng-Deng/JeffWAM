from pathlib import Path
import shutil
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e98w_control_20261114_01'
assert not (T/'results/e98w_a_20261114_04').exists() and not (T/'results/e98w_b_20261114_04').exists()
shutil.copyfile(R/'STATUS.json',R/'profile_stop_03.json')
with (T/'notes/e98w_robot_conditioned_20261114.md').open('a') as f:f.write('\n### 共享GPU profile预算停止\nA_03预热后100步最后20步median0.707秒，投影3535秒/run，仍超过3150秒预留，88.78秒完整入账并停止。运行期间共享GPU4有81%利用率。当前同卡利用率降至0，按相同准入规则新run A/B_04再测；只增加采样/数据/拷贝/计算分段计时，不改模型、损失、采样、精度和5000步合同。依然无dev前向。若无法预算内完成则如实交付未完成，不改设计凑结果。\n')
