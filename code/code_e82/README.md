# E82-M Piper：统一 O/A/W 潜空间 flow

仅 Piper 穿扎带；E79 架构 + E81 训练/评估合同。新代码不导入旧工具，不实现天机 adapter 或旧 P0/R0（扩展点见 `robot` 校验及 `ManifestDataset`/source adapter）。**本轮没有启动正式训练。**

## 结构

- `e82/core.py`：6×256 / 8 heads / FFN1024 / Pre-LN，共享 QKV/FFN；O只读O，A/W双向。eval逐层O K/V缓存，训练不detach。
- `vision.py,image.py`：本地原始 DINOv2-B，在线末2block Q/V rank8，E*固定768D；5个独立crop（顶、双腕全图、双腕原生ROI），有效像素加权pool。未知ROI严格missing。
- `model.py`：O187/A50/W273；task UNK=0；camera、age、missing、crop、空间/真实时距编码；q/dq/geo预留/history四槽。
- `data.py,source.py`：E81 manifest最少字段接口；E82-D实际 capture+NPZ adapter；READY状态与sha256 gate；只读原始MP4/HDF5，原next-feedback标签不重移帧；真实Δt因果dq。
- `objective.py`：P1无A、状态屏蔽、视觉/LoRA冻结；P3模式30/30/30/10，最后10%另算一次action_only LA；LA时权4/2/1、LW相机/腕全图ROI分组、端点.5/.3/.2，λ前500步升至.1。
- `randomness.py`：按seed/stage/step/slot/用途分流；R/P共享P3流，W不推进A流。
- `train.py,prepare.py`：S0、固定stat_contract、P1→P3、R直接P3、新阶段优化器、bf16+AdamW、累积有效32、恢复checkpoint与随机状态、3k/6k/9k网格、500步梯度诊断。
- `evaluation.py,eval_cli.py`：K8/10/25/32/50，每维与每臂关节rad、夹爪mm；四组共同P3训练尺度；先窗口→parent，再parent等权；10k配对parent/session bootstrap。拒绝模型标注评估真值。
- `runtime.py,diagnostics.py`：卡数×墙钟台账、预算/非有限停止、独立真值开发止损、latency P50/P95/P99与333ms miss Wilson区间。
- `profile.py`：只用一张当时空闲卡，100个microbatch=1 profile更新（P1 30、warm20、joint50），无正式checkpoint；按32倍计算与原位解码估算有效batch32 GPUh。10次/模式时延仅诊断，不能宣称1000次验收。
- `scripts/launch_formal.sh`：生成待用的 `setsid nohup` 串行完整两臂链；`chain_status.json`/各stage status与paired ledger。
- `artifacts/`：本轮CPU测试、参数表、数据等待/集成状态与profile证据。fixture为合成测试，不可当正式权重/统计。

## 环境与测试

服务器 T=`/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu`：

```bash
cd /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e82
PY=/mnt/workspace/dengjianfeng/venvs/e29/bin/python
export PYTHONDONTWRITEBYTECODE=1
bash scripts/run_contracts.sh
$PY -m compileall -q e82 tests
# CPU合成阶段衔接与resume；新路径，不覆盖
setsid nohup bash scripts/fixture_smoke.sh artifacts/fixture_smoke_NEW > artifacts/fixture_smoke_NEW.log 2>&1 < /dev/null &
```

torch/timm/pytest/numpy/cv2/h5py均使用现有环境；不安装、不下载。DINO默认本地路径由 `configs/piper_v1.json` 指定，缺失立即报错。

## READY接口与数据接入

`READY_parent_split.json` 和 `READY_data.json` 需有效发布状态且files中每个文件sha256一致。`ManifestDataset` 为显式窗口JSONL接口；当前E82-D使用 `DataPackageDataset` 读取 `READY_data.entrypoints.captures（当前为 manifest/captures_final.jsonl）` 与其 `numeric_cache`、`window_cache` NPZ。只允许train parent，P3正式训练额外要求 `train_ready_p3=true`；profile可读候选，所有候选性质写入结果。ROI无人工确认则missing；不偷用E77候选框、不从未来选框。

`StatContract`：动作/q/dq/history只拟合已锁定P3训练parent；世界只拟合允许P1训练parent，按camera/full/ROI组共享768D统计，floor=组正std中位数×.1，epsilon1e-6；两臂全阶段共用。prepare拒绝未审P3池，明确保留阻塞。

```bash
$PY -m e82.prepare counts --config configs/piper_v1.json --output artifacts/counts_NEW.json
# 下列操作用于后续正式数据准备；本轮只创建了counts，未执行正式S0/stat GPU准备。
$PY -m e82.prepare s0 --config configs/piper_v1.json --output /NEW_OWNED_PATH/S0.pt
$PY -m e82.prepare stats --config configs/piper_v1.json --device cuda:0 --output /NEW_OWNED_PATH/stat_contract.json
# profile内部选择memory<512MiB且util<5%的单卡，没有空闲卡返回75
$PY -m e82.profile --config configs/piper_v1.json --output artifacts/profile_NEW
```

profile使用真实训练小窗和临时identity尺度，仅测算力，临时参数不保存；不会绕过正式统计/质量gate。原始动作始终是next-frame follower反馈轨迹模仿，不能解释成发送命令或干预动力学。

## 正式启动脚本（本轮禁止执行）

配方：seed8101；P：同一S0→P1 5000×32→P3 warm1000×32→joint9000×32；R：同一S0→相同P3。Piper P2=0。5%warmup/cosine；联合固定候选3000/6000/9000；开发action_only K8选一次、平分选早、分组护栏；测试不用于选型。主比较action_only对action_only。

```bash
# 全部gate就绪后，由后续正式训练会话调用；NEW_RUN_DIR必须不存在
bash scripts/launch_formal.sh \
  /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e82_piper_NEW \
  configs/piper_v1.json /NEW_OWNED_PATH/stat_contract.json \
  /NEW_OWNED_PATH/S0.pt artifacts/profile_NEW/profile.json IDLE_GPU_INDEX
```

启动前要求：有效READY/人工P3质量许可、同版训练统计、同S0与encoder hash、真实GPU profile预测含4 GPUh准备≤48。正式两臂共享ledger硬停44 GPUh，准备桶另留4；不自动缩短一臂。脚本不自动重试失败。恢复同阶段使用原run、原config/stat/input，`python -m e82.train ... --resume`；严格核对代码/数据/配置hash、加载optimizer/RNG，阶段衔接则重建optimizer。静态冻结视觉权重不复制进新checkpoint，仅保存自有模型/LoRA并引用原始encoder hash。

开发止损可通过 `--monitor /OWNED_PATH/dev_monitor.json` 提供独立真值配对记录；`stoploss_monitor` 实现对齐开发点连续退化、raw限位异常、连续世界梯度主导且动作恶化。没有开发真值时不凭LW/模型标注自动选型或放行。正式网格不自动读取测试。

## 评估

```bash
# R/P预测npz必须有相同window_id/原始truth；parent/session/split/truth_source必填
$PY -m e82.eval_cli --R R_predictions.npz --P P_predictions.npz \
  --stats stat_contract.json --output /NEW_OWNED_PATH/paired_metrics.json --draws 10000
```

NPZ字段：prediction/truth为[N,50,14]；window_id、parent_component、session_id、truth_source、split为无pickle字符串数组。truth_source只允许human/recorded_feedback；不使用E82-L模型标注作评估或选型。session<10报告区间不稳定；历史集保持exploratory标识。

缓存不生成永久在线LoRA特征；固定目标如以后缓存，key须含source/frame/camera/ROI/encoder/pool/split/stats hash。当前原位读取、仅4个数值capture内存LRU；不复制大视频，不清旧数据，不触碰他人进程。

独立评估执行者可先运行 `python -m e82.predict --config ... --manifest LOCKED_EVAL_WINDOWS.jsonl --stats ... --checkpoint ... --split dev --mode action_only --device cuda:0 --ledger PAIRED_RUN/paired_budget.jsonl --output NEW_predictions.npz`。显式评估窗口必须 `allow_eval=true`，包含独立 recorded_feedback/human truth_source、原位数值/图像引用及已锁定原始配对；本轮未读取冻结test，不自动从未审P3候选制造eval清单。推理噪声按固定四seed和canonical window_id派生，不随batch切块变化。

资源审计修正：早期CPU检查点读取了CUDA RNG，可能触发CUDA上下文初始化；最终代码CPU路径禁止CUDA RNG（有回归测试），CUDA路径只保存实际单卡状态。未运行CUDA模型计算。早期潜在上下文按相关CPU进程全部墙钟×8卡作保守上界另记，真实profile桶已降为0.4 GPUh，以保持本会话总上界<1 GPUh。

最终验证（2026-11-12 BRT）：29项测试全部通过；2,476个READY文件哈希验证与真实生产尺寸DINO CPU前向通过，缓存误差0。真实GPU profile因8卡均占用未运行，48 GPUh与P99仍未实测；P3_ready=0、腕ROI未审，正式训练未启动。见 T/notes/e82_M_model_pipeline_20261112.md。
