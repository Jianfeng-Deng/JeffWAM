# E91b ABC uncertain 复查（2026-11-14 BRT）

状态：已交付v2并通过校验；部分逐段补帧密度受24张上限限制，详情见§5.2。CPU-only，训练文本，非真值。保留 v1，源数据只读。

## 1. 输入与 uncertain 统计

## 2. 补帧与证据范围

## 3. 逐项判断与因果限制

## 4. 独立盲复核

## 5. 结果、限制与交付

### 1.1 初始审计 2026-11-14T04:15:58.883382-03:00
128 uncertain 项，75/116 episodes。task_instruction_en 11/116，process_narration 89/348，cleaned_original_segments 28/2484，scene_description_en 0/116，events 0/230。递归扫描 uncertain=true 与文本单词 uncertain，contains_uncertain 为汇总标记而非独立项。
逐段抽帧与每episode24张在多段episode冲突，已向用户询问；先处理无冲突部分。叙述是单帧观察点，扩展为截止帧前0.5秒的6点与前后各2点，再加截止时左右腕视，总12张/观察点。

### 2.1 补帧完成 2026-11-14T04:29:32.668623-03:00
62个仅叙述episode生成816张，12个多类型episode生成287张，另episode78汇总标记补查12张，共75 episodes / 1115张JPEG，640×480，CPU解码，单episode最多24张，当前新增约55.06MB。联系表仅在内存拼接与显示，不落盘。三路原位读取；源stat前后核对通过。
初始128个显式项实际分布在74 episodes；v1 contains_uncertain共75条，额外ordinal78只有汇总标记，叙述称袋内遮挡。已补看该点并保留汇总uncertain，不将汇总标记混入各类文本项分母。
12个多类型episode的时间窗采用共享帧并严格保持24张上限，部分短段未达到逐段6张要求；逐项实际覆盖写入coverage_audit.json。用户尚未答复取舍问题，因此未突破硬上限。

### 3.1 首次判断冻结
128/128显式项已逐项查看补帧并判断，47项resolved。decisions_frozen.json与SHA在读取盲复核前冻结，后续仅做结构装配。另ordinal78袋内仍被塑料遮挡，汇总标记保留。10条任务指令改为zip tie；ordinal108的深色条带头部细节仍不足。花束ordinal35确认印字布带，但原段23原文语义异常，保留uncertain。

### 4.1 盲复核结果
random.Random(91)从128显式uncertain项抽13项（10.15625%）；独立上下文显式gpt-6-astra/low，读取v1与补帧、不读首次结论。resolved判定13/13一致；解决的4项文本语义兼容（zip tie/cable tie同义），其余9项均保留uncertain。这是同模型家族共享图像证据复核，非人工真值、非准确率。主会话runtime模型身份未独立暴露，用户指定身份与实际可验证身份分开记录。

## 5. 最终交付与复核 2026-11-14T04:33:25.718149-03:00

状态：v2已生成，结构、v1回溯、源时间边界和因果证据校验通过。逐段抽帧密度存在下述未满足项。

|类型|总数|v1 uncertain|v2 uncertain|比例变化|
|---|---:|---:|---:|---:|
|task_instruction_en|116|11|1|9.48% → 0.86%|
|scene_description_en|116|0|0|0.00% → 0.00%|
|process_narration|348|89|78|25.57% → 22.41%|
|cleaned_original_segments|2484|28|2|1.13% → 0.08%|
|events|230|0|0|0.00% → 0.00%|

显式uncertain由128项降到81项，解决47项（36.72%）。按原episode汇总标记口径，75/116（64.66%）→69/116（59.48%）；仅显式项覆盖episode为74→68，额外ordinal78的汇总标记单独保留。

### 5.1 修改与典型证据
- 总指令修改10条，ordinal 97、98、101、102、105、106、107、109、110、112。统一从“Gather the wires and secure them with the narrow fastener (type uncertain).”改为“Gather the wires and secure them with a zip tie.”。这表达任务目标。
- ordinal109原尺寸腕部frame2405显示黑色矩形锁头及孔口；ordinal106 frame2432及中点腕视显示浅色锁头和齿纹。ordinal108头部与线缆/夹爪重叠，仍保留类型uncertain。
- ordinal102的清晰锁头证据在frame2392，晚于中点叙述截止，因此仅更新总指令及允许后见确认的原段。该中点叙述继续保持类型uncertain。
- ordinal35确认宽棕色印字布带，8个原段解除材料歧义；原段23“Pick up the center of the board...”语义异常仍保留。叙述中的结扣状态继续uncertain。原段解除标记仅代表原先材料/类型歧义得到解决，源动作没有获得稠密独立认证。
- 其余仍不确定的主要原因包括夹爪/布料/袋体遮挡、锁口尺寸小、黑色条带低对比、卷轴挡住绕线层、线缆交叠无法追踪，以及静态图无法确认紧度/强度/密封。

### 5.2 抽帧覆盖限制与用户取舍
- 硬上限全部遵守，75 episodes、1115张原尺寸JPEG，最多24张/episode，JPEG合计53.434MB；本轮data新增约60.952MB，远低于1GB与60GB。GPUh=0。
- 12个多类型episode合并时间窗共享证据。按严格±2.000秒计数，71/128项满足段内≥6张、前后各≥2张；57项不满足，包含24叙述、22原段、11总指令。总指令采用整episode范围，片头前/片尾后自然没有源帧；其余主要受24张上限约束。
- 上述57项中ordinal37、42、51各1个叙述项仅有一张前帧因向前取原生帧而略早于严格2秒边界，仍接近用户要求的约2秒；其余54项均在12个共享帧episode。实际slot与时间范围保存在coverage_audit.json，未伪称逐段6–12张全部达标。
- 需要用户决定的仅是是否放宽这些episode的24张上限，以补齐逐段密度。当前交付使用严格24张上限，未等待决定而停下可完成的复查。

### 5.3 复核与格式
- 独立盲复核13/128（10.16%），resolved判定13/13一致（100%）；解决的4项语义兼容，另9项均保留uncertain。复核只覆盖抽样项。同模型家族、非人工真值，不能解释为100%准确。
- 主会话用户指定gpt-6-astra/low，runtime身份未独立暴露，未声称已核验；复核agent显式配置gpt-6-astra/low。
- lang_abc_v2.jsonl保留116行及v1原字段。每个被复查的叙述/原段增加uncertain_resolved、resolved_by_frames、resolution_notes、confidence、v1_value；字符串总指令的元信息在field_resolutions.task_instruction_en。uncertain_reviews提供统一逐项索引。
- 行级uncertain_resolved仅在原本有不确定项且全部解决时为true；未复查的非uncertain行=false并明确标记未重评。行级v1_value保存完整原行，逐项v1_value保存原对象或原文本。历史快照中的uncertain不能计入v2当前统计。
- resolved_by_frames记录用于解决的帧，未解决项为空；reviewed_by_frames记录全部查看帧。叙述的causal_evidence_frames严格不晚于原timestamp和source frame截止。后续帧只进入复查说明及允许后见确认的字段。

### 5.4 校验、目录与命令
- v1 SHA256仍为 7713852cb82403990f5b83c0b98a98d5237d4de3df78ea59c5477325e9d40abc。
- v2 SHA256为 9e451cc14902d3d6f785474929cd4d0f03feb512fee5aa9f23070ac85c439e7f。116个完整v1快照逐行相等，128显式项+1汇总标记均有复查记录。原段边界、叙述截止、original_text_en与events保持一致。
- 交付文件为data/e91_abc_lang_v1/lang_abc_v2.jsonl；证据/脚本/审计在keyframes_v2/；状态在keyframes_v2/state/final_status.json。
- 复现脚本为extract.py、extract_complex.py、extract_flag_only.py、view.py、assemble_validate.py。抽帧脚本通过setsid nohup env PYTHONDONTWRITEBYTECODE=1 data/abc_inventory_v1/env/venv/bin/python分离执行；装配使用python3。已有输出排他保护，原样重跑会拒绝覆盖，应使用新授权run路径。
- 未复制MCAP、未下载权重或数据、未训练、未访问GPU、未触碰他人进程；源文件stat前后核对通过。所有新增文件在指定data目录与本报告，公共日志仅追加。
