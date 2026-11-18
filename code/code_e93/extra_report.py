import json,fcntl
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e93_dev_diag_20261114_01';a=json.loads((R/'aux_cos_results.json').read_text());l=json.loads((R/'train_language_results.json').read_text());lines=['\n### B 额外CPU诊断：辅助余弦头与原训练语言','短时域的world_cos_head是独立读出头（共享观测表征），并非rollout的world_head；其方向损失不能直接解释为rollout残差方向已学好。下面用零W输入、t=0读取纯因果O条件的辅助头；不传真实未来作网络输入。余弦按整窗token/channel展平后计算，和rollout方向报告一致。它只接受方向监督，输出模长任意，因此不把其原始输出MSE当作世界性能。','|池|+帧|辅助头残差方向cos [95% parent CI]|','|---|---:|---|']
for pool,dd in a.items():
 for h,r in dd.items():
  ci=r['ci95']['cosine'];lines.append(f'|{pool}|{h}|{r["values"]["cosine"]:.5f} [{ci[0]:.5f}, {ci[1]:.5f}]|')
lines+=['该头方向略有信号，但比线性参照弱；训练与dev接近，未显示仅dev突然失效。不能把“.1短时余弦辅助”当成对采样均值的直接MSE监督。','\n为排除train64用固定短指令导致表面欠拟合，从同64个训练parent各取已有选中窗口中位的1窗（seed/索引原样配对），另用该窗原本训练时的instruction评v1。54/64窗有原E83训练指令，其余原本就使用固定短指令。只在TRAIN读取原训练文本，dev/test语言完全未变。','|+帧|配对窗/parent|固定短指令skill %|原训练instruction skill %|','|---|---|---:|---:|']
for h,r in l['original_instruction'].items():lines.append(f'|{h}|{r["windows"]}/{r["parents"]}|{100*l["fixed_human_short"][h]["values"]["skill_copy"]:.3f}|{100*r["values"]["skill_copy"]:.3f}|')
lines+=['原训练指令没有消除负skill；不能将本轮训练侧失败主要归为缺少原模型任务文本。该表只有每parent1窗，不与主507窗表混合或拿它选语言/ckpt。全部CPU、无参数更新，结果见aux_cos_results.json与train_language_results.json。','\n固定时域训练FM均值补充：','|时域|3001–4000|4001–4500|4501–5000|','|---|---:|---:|---:|']
for k,v in json.loads((R/'curve_horizon_means.json').read_text()).items():lines.append('|'+k+'|'+'|'.join(f'{x:.5f}' for x in v.values())+'|')
with (T/'notes/e93_dev_diag_20261114.md').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'.join(lines)+'\n')
