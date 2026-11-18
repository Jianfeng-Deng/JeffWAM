from common import *
p=C/'report.py';s=p.read_text();s=s.replace('batch4（3个FK运动对+1个E83 caption）','batch4（每8步一次3个FK+1个E83，其余为4个FK对）').replace('每parent最多2条E83，固定hash/最近锚点≤15源步，不看val选样','E83仅保留原标注source_frame_index与锁定anchor_source_row完全相同的11条；固定hash，不看val选样').replace('E83训练caption完整唯一句331','E83训练caption完整唯一句11').replace('caption完整唯一句331','caption完整唯一句11')
needle=" s.append(conclusion)";s=s.replace(needle,needle+"\n s.append(f'严格配对修正：先导run `{PILOT}` 的331条E83正配对有320条近邻锚点，为严格遵守不平移标签，先导结果仅保留诊断，不作为本报告正式证据；停止自己的未完成任务后，新run仅保留11条同锚点caption，FK1483对、原selection/split与所有真值均不变。E83每8步一次，避免11条文本过度重复，固定768步，不按任何val指标选checkpoint。此前调试/训练/复提{PRIOR_LORA_GPU_HOURS:.6f} GPUh全部计入同一4GPUh预算。11条E83覆盖不足限制对完整阶段一方案的外推。')")
s=s.replace('本轮LoRA含调试/训练/两组全局复提','本LoRA阶段含先导作废/调试/训练/两组全局复提');p.write_text(s)
p=C/'record_training.py';s=p.read_text().replace('1483对FK+331对E83','1483对FK+11对严格同锚点E83').replace('首个设备放置错误发生在优化步0，已归档，GPU时间计入；未OOM','先导近邻E83已作废，其GPU时间全计入；本次无设备错误/未OOM');p.write_text(s)
p=C/'finalize.py';s=p.read_text().replace('训练1483FK+331E83/191parent','训练1483FK+11同锚点E83/191parent').replace('首次设备放置错误在优化步0已修正，CPU准备性能/字段修正均留档，调试GPU时间全计入','先导近邻E83配对不作正式证据并留档，本次只用11同锚点caption；所有先导/调试GPU时间全计入');p.write_text(s)
# Copying immutable train-cache rows is a subset operation, not action/label retiming.
update(status='strict_exact_training_ready',strict_e83_exact_anchor=True,e83_pairs=11,e83_sampling='one caption every8steps, fixed768steps')
print('strict_configured',RUN,PRIOR_LORA_GPU_HOURS)
