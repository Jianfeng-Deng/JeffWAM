# E93 dev/cache/checkpoint diagnostics

本轮不训练神经模型。所有脚本只读 E90、E82/E86/E88、共享数据和冻结 checkpoint；仅 E93 工具/结果、报告及共享日志追加可写。Python 使用已有 `/mnt/workspace/dengjianfeng/venvs/e29/bin/python`，禁用 pyc，未安装/下载任何依赖或权重。

固定结果根：`T/results/e93_dev_diag_20261114_01`；报告 `T/notes/e93_dev_diag_20261114.md`。这些是本轮专用输出，重新运行须用新 run 并统一更改各脚本的 R，不要覆盖已有结果。脚本中频繁更新的 status/ledger/partial JSON 是本轮工作状态；已发布 E90 v1 完全不变。

输入：`T/code_e90/artifacts/e90_p1_20261114_01/ckpt_05000.pt`，SHA256 `45164cfb86483709892d5c59a55fb830484cb5923eb7c018579595c826abbc49`。train residual stats 完全沿用 E88，dev 固定原人工短指令，排除指定 ep142。

执行顺序：

1. `prepare.py`：保持旧48窗，新parent沿用固定分位候选/最多8窗；先估大小。
2. `supplement_prepare.py`：审计缺失的冻结parent；从原始同合同时钟/视频补回1个demo6失败parent，其他不造数据。
3. `extract.py`：六parent全1134帧严格bitwise重放成功后扩展26parent。
4. `extract_supp.py`：同合同补第27parent，覆盖合计5440相机帧。
5. `cpu_eval.py`：CPU fp32、4Euler、N1，dev211窗和固定train64最多8窗/parent。首个收尾断言把新+100记录计入旧记录数，失败保留。
6. `cpu_validate.py`：按旧availability配对原145记录；CPU模型MSE最大差4.77e-7，copy/linear差0，新旧缓存重叠帧差0。完整推理结果无需重跑。最终 CPU 状态以 `cpu_validated_status.json` 为准。
7. `gpu_diag.py`：全dev、train64的4/8/16Euler × 前1/8/16样本均值；CPU生成逐窗固定噪声后拷GPU，float32；附joint query、FM参考与固定head子空间oracle下界。每窗检查共享显存和累计GPUh≤2。
8. `linear_03.py`：CPU训练parent内PCA/ridge闭式参照，内层训练parent验证选alpha，无dev选择。最初缺非必要threadpoolctl直接退出，随后移除依赖；linear_02等待CPU旧断言失败，均0GPUh。
9. `aux_cos.py`、`train_language.py`：CPU额外检查独立因果短余弦头及原训练instruction对照，train文本绝不进入dev。
10. `early_report.py`、`linear_report.py`、`grid_report.py dev/train64`、`extra_report.py`、最终FM/结论汇总脚本：逐节追加报告；history_section.md包含完整指定来源检索审计。

长任务均 `setsid nohup env PYTHONDONTWRITEBYTECODE=1 ... > results/.../独立日志 2>&1 < /dev/null &`。原 pipeline.py 止于CPU收尾断言，保留 `pipeline_status.json`；恢复阶段单独启动，最终会话状态使用 `E93_FINAL_STATUS.json`。不要将旧失败状态误当最终交付状态。

GPU账本：`gpu_ledger_extract.json`、`gpu_ledger_extract_supp.json`、`gpu_ledger_diag.json` 进程墙钟求和（含加载/I/O/失败，保守≥CUDA占用）；初始估算峰值8GiB，准入free≥12GiB。无需util0、未用MPS/改他人进程；峰值/余量超限或预算超限停止。仅使用本机已有SigLIP2，无VLM生成标注。

主要结果：coverage_final.json、dev/train64_v1_summary.json、dev/train64_grid_summary.json、dev/train64_decomposition.json、linear_summary.json、fm_diagnostic.json、world_head_subspace.json、history_search_hits.json。模型生成历史mask指标仅按审计代理记录，未作本轮测试真值/门槛/选型。

覆盖边界：51个冻结parent全审计，27个具有211个同合同固定窗，24个因metadata-only/必要时钟缺失/太短无法产生原合同有效窗；不宣称51个都有有效缓存。没有改变冻结划分、补模型语言、写训练缓存、移动帧或标签。

最终补充：exposure.py纯CPU精确重放主训练采样索引，状态匹配；clean_metrics.py移除非v1诊断JSON内未使用的内部linear占位字段，所有真实model/copy/skill/cosine数值不变，见metric_contract.json。只有dev/train64_v1中的linear字段代表真实因果曝光时间外推。
最终状态：0.585266GPUh，66,966,524B dev特征，神经训练更新0，受保护284文件变化0。
