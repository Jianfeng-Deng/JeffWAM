from pathlib import Path
import json,shutil
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e98w_control_20261114_01'
assert not (T/'results/e98w_a_20261114_03').exists() and not (T/'results/e98w_b_20261114_03').exists()
shutil.copyfile(R/'STATUS.json',R/'profile_stop_02.json')
with (T/'notes/e98w_robot_conditioned_20261114.md').open('a') as f:f.write('\n### 训练内profile停止与I/O预热\nA_02前20步median1.283秒，保守投影6415秒/run超过单run预留3150秒，自动停止；未产生dev预测，不保存/选择模型。全部41.19秒GPU段入账。检查发现profile包括新capture状态/FK冷读；下一新run A/B_03在GPU准入前CPU预热全部546 capture状态/FK，profile固定取前100步的最后20步（仍计入实际训练，不重启/改配置）。共享卡在满足显存条件后按最低利用率选择，继续监控；模型、目标、优化器、5000步、采样和预注册判定均不变。\n')
print('new run paths clear')
