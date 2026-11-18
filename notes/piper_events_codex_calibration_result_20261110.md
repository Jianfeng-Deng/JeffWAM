# Piper Codex 结构化事件标注结果（2026-11-10）
阶段0完成：173 episodes、154父组、211571帧源时间映射、230拼缝；建议train/dev/sealed_test为146/9/18 episodes（131/8/15父组）。跨集父组已在建议中整体隔离，8个人工父组封存；未修改任何旧划分。
阶段1完成：读原始三相机视频，ffmpeg CPU抽帧并实际查看101张contact sheet；先提交8条盲标和SHA256，再读人工答案。104个事件存在性记录+2个unknown.event候选，所有帧边界仍标unknown并给不确定区间；没有宣称精确点已确定。
主容差±15：P=0.2308，R=0.2400，F1=0.2353；±8 F1=0.1275，±30 F1=0.4118。24个±15匹配事件arm/object各100%，attempt83.33%；无时间容差诊断arm91.21%、object90.11%。LCS顺序一致率75%。
handover召回62.5%，thread_done11.11%；failure/retry在±15均0检出。事件F1与关键类型均未达预设门槛，阶段2停止，未标24个新父组。严格将unknown.frame全视拒答时，确定精确帧预测为0、F1为0。
主要问题：操作位语义放错臂和阶段、穿入完成早标、首次微调和拉紧边界不一致、失败与attempt分配错误。高置信度不是点位准确率。详细指标与限制见校准报告。
文件：
- manifest/parent_groups.csv、source_time_map.csv、proposed_split.json、normalization_mapping.json、STAGE0_SUMMARY.md
- calibration/PREREGISTERED_PROTOCOL.md、codex_blind_{0,4,10,38,41,42,45,76}.jsonl、BLIND_COMMIT.json、HUMAN_UNSEAL_RECEIPT.json
- calibration/AGREEMENT_REPORT.md、agreement_metrics.json、matched_events.csv、unbounded_diagnostic_matches.csv、STAGE1_SUMMARY.md
- calibration/review_queue.jsonl（全部106条待复核，已排序）、frontend_exports/decisions_piper_v2_<ep>.jsonl、FRONTEND_REVIEW.md
- calibration/VISUAL_REVIEW_LOG.json、QA_RESULT.json、evidence/（原视频抽帧与拼图）、scripts/（抽帧与评分代码）
- STOPPED_STAGE2.md
复核顺序：ep10最初失败/重试及ep41失败/重试 → 全部thread_done和handover → 操作位/接近语义 → aligned/adjust/tighten。前端复核前先备份浏览器列表，导入会替换列表；派生note含来源审计标记，完整字段以原盲标为准。
核验：8条盲标哈希未变，人工文件自解封后的哈希未变；211571源映射行与父分组隔离检查通过；匹配程序200个小样本穷举核对通过。ep42一个奇数帧锚点25有效但前端stride2可能舍入，已单列说明。
所有服务器产出仅在本次新目录及notes新文件，未启动训练、使用GPU、操作他人进程或访问互联网。
**模型辅助标注，需人工复核，不得作为测试真值；不得用于阈值选择或模型选型，不能充当独立人工标注者。**

产出根目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/piper_events_codex_20261110
