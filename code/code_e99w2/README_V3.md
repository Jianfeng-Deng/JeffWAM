# E86-V3 Piper 接口

本轮 ID：E86-V3-20261113T144520。仅小规模工程验证，正式训练未授权启动。

## 冻结入口

- P3 默认入口：`data/e82_piper_v1/P3_ready_V3.json`，代码固定校验 READY 本身、源池、用户采用收据和永久排除清单的 SHA256。
- V3 源池原字节保持不变；284 个 entry / 34,777 个窗口 / 145 parent。历史 pool 内 proposed/approved=false 由独立用户批准收据授权，不能修改源 JSON。
- 永久排除：`data/e82_piper_v1/exclusions_episode142_E86_V3_20261113.json`；canonical `piper_4dc107335389f5711a0f65e0` 在 Package、build_window、CachedDataset 和训练采样器中均拒绝。
- 可读 P1=107,057 窗；V3 的 284 条以 entry_id/cycle_id 区分，禁止按 canonical 去重而丢轮次。只用显式 anchor，不扩张区间；原动作是 next-frame 实测反馈。
- `code_e86/artifacts/E86-V3-20261113T144520/READY_V3_cache_windows.json` 绑定最终窗口、训练统计、增量缓存合同。原缓存只读复用，新增帧位于同目录 `features_increment_02/`，`overlay.json` 指向旧缓存。

## 验证与入口命令

解释器 `/mnt/workspace/dengjianfeng/venvs/e29/bin/python`，在 `code_e86` 下执行，`PYTHONPATH=.`。

```bash
python -m pytest -q tests
python -m e86.v3_train smoke E86-V3-20261113T144520/NEW_UNIQUE_SMOKE_ID
python -m e86.v3_train profile /ABS/NEW_UNIQUE_SMOKE_RUN
python -m e86.v3_train replay /ABS/NEW_UNIQUE_SMOKE_RUN --stage P3
```

以上命令仅描述接口；再次执行须另行计入 GPU 预算，不能覆盖本轮产物。`python -m e86.train` 默认也转向新的 V3 smoke 入口。旧 driver、eval_cli、resume/profile_train 等脚本属于 V1 历史工具，新的 V3 run 使用 v3_train 中的 profile/replay 和固定 stage-end dev 评估；不把旧路径当 V3 入口。

可选参数：`--sampler parent_equal|stratified`、`--stratum-weights '{"demo3":1,"demo4":1,"demo6early":1,"demo6native":1}'`、`--demo3-cap 0.45`、`--mode-probabilities 0.3 0.3 0.3 0.1`、`--clean-aux 0.25`。demo3 cap 约束累计抽取窗口数的每个前缀，不是 entry 数/parent 数，不能超过45%；受 cap 截断后，原始分层权重不再是精确实际份额，run 日志记录实际抽样数。

默认配方维持主模式30/30/30/10；同步/独立W调用另以25%概率加干净动作辅助，预期每更新干净条件世界暴露25%。若用户选择“干净动作是互斥主模式25%”，一种相对比例不变的配置为 `--mode-probabilities 0.25 0.25 0.25 0.25 --clean-aux 0`；具体口径仍待用户锁定。

## V1 回退

`Package(version='V1')` 仍固定验证原 `P3_ready_V1.json` 和原池 SHA256，并同样落实永久排除。回退数据统计必须采用原 V1 统计，不将 V3 stats 冒充 V1 stats。本轮55项既有合同保留为明确 V1 回退/原模型回归，新增 V3 合同另测。原始源码归档为本轮 artifact 下 `source_before.tar.gz`；历史 checkpoint 的严格代码hash限制保留，不能以新代码声称逐位复现旧运行。

模型审核仅定义训练池和训练目标，不是人工真值；dev 固定使用既有真实反馈48窗和原人工任务指令，不生成语言标注；test/sealed_test不打开。dev 参考与训练损失不作 checkpoint 选择。
