from pathlib import Path
import json,collections,numpy as np
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';R=T/'notes/e109g_gripper_generalization_20261116.md';m=json.loads((O/'PARENT_METADATA_RESOLVED.json').read_text())['records'];full=json.loads((O/'FULL_CAPTURE_STATS.json').read_text());cov=json.loads((O/'COVERAGE.json').read_text());ex=json.loads((O/'EXPOSURE.json').read_text());lines=['## 4 已完成：parent来源与物理分布（预测评分独立）','172个代表canonical的原始HDF5均只读核验。原colleague-a路径已整体迁移至/mnt/workspace/dataset/piper；只在本轮RAM解析路径，视频补提核对旧receipt的size/mtime，原始元数据核对帧数及首帧日期，不改旧manifest。录制日取原timestamp的UTC自然日（另存BRT）；所有执行记录用BRT。数据所有者不等于操作者。原始attrs中无operator/scene/人工阶段字段，无法证实同操作者或同真实场景；source_collection、camera serial仅作来源/设备代理，不作场景真值。dev不读入模型语言，不用模型阶段/结果标签分层。','|UTC录制日|训练parent|dev parent|','|---|---:|---:|']
for field in ['recording_date','source_collection']:
 if field=='source_collection':lines+=['|录制来源（场景代理，非场景真值）|训练parent|dev parent|','|---|---:|---:|']
 counts={s:collections.Counter(r[field]['UTC'] if field=='recording_date' else r[field] for r in m if r['split']==s) for s in ('train_aligned','dev')}
 for key in sorted(set(counts['train_aligned'])|set(counts['dev'])):lines.append(f"|{key}|{counts['train_aligned'][key]}|{counts['dev'][key]}|")
lines+=['dev所有录制日均出现在训练侧；但demo4 inference来源dev有5/27parent、训练代表canonical为0，存在来源组成差异。同日并不等于同场景或同操作者。完整父级物理统计在FULL_CAPTURE_STATS/COVERAGE.json，逐parent附表在后续章节。','|集合|parent|窗数|重复计数事件|唯一覆盖事件|窗口目标覆盖帧/全帧|有事件parent|','|---|---:|---:|---:|---:|---:|---:|']
for s,pp in cov.items():
 vs=list(pp.values());lines.append(f"|{s}|{len(vs)}|{sum(v['windows'] for v in vs)}|{sum(v['window_events'] for v in vs)}|{sum(v['unique_covered_events'] for v in vs)}|{sum(v['unique_frames'] for v in vs)}/{sum(v['full_frames'] for v in vs)}|{sum(v['window_events']>0 for v in vs)}|")
lines+=['|集合|完整录制事件/分钟 parent中位数[IQR]|完整事件间隔秒中位数[IQR]|30帧净开合幅度mm中位数[IQR]|','|---|---|---|---|']
for s in ('train_aligned','dev'):
 fs=[f for f in full.values() if f['split']==s];freq=[f['event_frequency_per_min'] for f in fs];intervals=sum([f['interval_s'] for f in fs],[]);amps=[a[1] for f in fs for a in f['amplitude_30frame_mm']];fmt=lambda a:str(np.quantile(a,[.5,.25,.75]).round(3).tolist()) if a else '无';lines.append(f'|{s}|{fmt(freq)}|{fmt(intervals)}|{fmt(amps)}|')
lines+=['幅度定义预先固定为onset前1帧至后30帧净位移绝对值，只在同连续段且状态有效时计入；不是完整一次开合行程。事件间隔按同爪/同连续段计算。完整录制统计与被窗口覆盖事件是不同分母，重叠窗重复事件不会当独立真事件。','## 3 已完成：训练采样器与夹爪损失核对','原样重放MultiSourceSampler的P3 seed90/P1 seed10390、parent_equal、demo3累计cap=.45。A2 2000步P3样本SHA、Piper前10000步P3/P1 SHA，以及10001..15000/20000段两池SHA全部吻合实际日志。A2直接夹爪监督64000抽样；Piper20k再640000，共704000次P3窗口暴露。P1另640000次、242parent，仅世界/余弦/adapter辅助权重.1，没有夹爪BCE；不能把P1次数等同夹爪训练次数。P3池34777窗/145parent；父级等权后父内窗口等权，非事件平衡。','夹爪BCE是原有target-free t=0动作分支的每窗50步×2爪平均、权重1，无event/class权重。关节FM/MSE另有时间权重4/2/1，未施加到夹爪BCE。四模式均计算该BCE；训练loss下降本身不证明事件F1提高。']
vals=[v for v in ex['parents'].values() if v['P3_windows']];lines.append('直接夹爪parent暴露数 min/Q25/median/Q75/max='+str(np.quantile([v['direct_grip_draws'] for v in vals],[0,.25,.5,.75,1]).tolist())+'；每P3候选窗暴露数对应分位='+str(np.quantile([v['draws_per_P3_window'] for v in vals],[0,.25,.5,.75,1]).tolist())+'。后者反映父内候选窗数差异，不是精确物理事件见到次数。模型初始化前的更早视觉/世界训练暴露未计为夹爪目标暴露。')
assert not (O/'METADATA_REPORT_COMPLETE.json').exists()
with R.open('a') as f:f.write('\n'+'\n'.join(lines)+'\n')
(O/'METADATA_REPORT_COMPLETE.json').write_text('{"complete":true}')
print('\n'.join(lines[-4:]))
