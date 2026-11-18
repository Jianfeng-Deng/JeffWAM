# E90 完整 v3.1 与仅 Piper P1

仅执行 `e90` 入口。复制的 `e86/`、`e88/` 保留原108项合同回归及原模型参照，旧写入命令禁止运行。旧code_e82/e86/e88、数据和缓存只读。

Python：`/mnt/workspace/dengjianfeng/venvs/e29/bin/python`；cwd=`T/code_e90`；`PYTHONPATH=.`。

```sh
python -m pytest -q tests
python -m e90.preflight
python -m e90.train e90_smoke_20261114_01 --smoke
python -m e90.train e90_smoke_20261114_01 --replay
mkdir -p ../results
python -m e90.train e90_p1_20261114_01 --mode-probabilities .25 .25 .25 .25 --clean-aux 0
```

所有输出目录必须不存在。恢复到新run：`--resume /absolute/ckpt_XXXXX.pt`，严格验证代码、配置、数据hash，恢复模型/优化器/调度器/各RNG/采样器。新阶段仅权重初始化：`--init-ckpt /absolute/ckpt_XXXXX.pt`，与精确恢复分开；本轮仅单测该入口，不启动步骤2或P3。

`e90.model.WorldModel`：继承冻结SigLIP2缓存的完整v3.1，6×256/8头/FFN1024，当前3×8、历史4×8、文本、NULL，新增来源O token（Piper/yam/ABC）。O仅读O；短W读长W，长W不读短W，也不读可能携带短W的A（保留E88防间接泄漏合同）；状态只在A。动作默认action_only；12关节相对当前q、夹爪signed absolute、50×14 flow matching；对齐头及因果阶段/进度头保留。

世界固定10/25/50为每路2×4，100为每路global；连续时域嵌入，默认再抽10..100真实整数偏移；200/300在`sample_horizons(maximum=300)`和Data接口保留，正式配置默认100。未来减当前，训练parent统计归一化，FM满768维噪声旁路，短时域余弦，长时域无余弦。

`config_for`固定P1 5000步、batch32、AdamW 3e-4/1e-4、clip1、恒定LR，与E88建议一致。前3000世界损失（FM + .1短余弦 + .1重入蒸馏辅助），不做动作、对齐、阶段训练。适配器独立64 train parent拟合、16 train parent门检；最初与3000边界蒸馏，NMSE≤.1且cos≥.95，失败则本run关闭链；每16步复查当前门。3000..4000链0→.1，4000..5000保持.1；每次合格更新抽一链、最多2窗、2Euler，完整系数，不乘四分之一暴露率。首跳与直接端停梯度；第二跳监督真实+2k及直接一致性，预测重入概率.2→.8；真实历史因果后移，无状态/动作。

P1读图像/时序/文本，不装配动作/状态数组，不抽取新特征。已有E83任务指令只在原有效区间，缺项固定人工短指令；无过程叙述。dev仅原固定短指令。缺+100帧按该时域mask，不丢弃同窗有效短时域监督，不移动标签。

MultiSourceSampler支持源权重与按累计窗口每个前缀的硬上限；源内parent等权、demo3≤45%。须至少一个无上限正权重来源保证第一个前缀可行（通常Piper）；本轮仅Piper。只保留只读mmap句柄与行索引于RAM，不复制特征或新增磁盘缓存。

GPU统一台账`artifacts/gpu_ledger.json`≤21600秒（失败计入）；准入util0、free≥8192MiB×1.5，实测峰值超估算/共享余量不足/OOM停止。固定500步保存，不用dev选型。最终`results/e90_p1_20261114_01/ckpt1_dev_eval.json`包含4Euler、2000次paired parent bootstrap、copy/linear/skill、demo及demo6两细层、链与真实增量可加性。已有dev仅6/51 parent、48窗，100仅1窗；不重建缓存，稀疏层的NA与单parent退化区间如实记录。

本轮5000步后自动评估因结果父目录缺失退出；训练/10份checkpoint完整保留。`eval_tail_e90.py`在新进程只读加载同一5000步权重完成评估；失败段照计GPUh，原状态保留为`training_driver_stopped.json`。训练包与checkpoint代码hash保持不变。以后运行前按上面命令先创建results父目录。
