# E104-P 外部预训练再 Piper 微调

开始：2026-11-15T12:53:31.962316-03:00 BRT
状态：COMPLETE（有效run03，终态及审计见后文）

## 预注册
A2 固定起点；F2 Piper 10000步（5000为F0）；F1外部5000+同序Piper5000。seed90，恒定lr3e-4，AdamW状态连续。外部ABC:yam=1:1固定。仅固定终点，不选点。

## 数据与代码审计
见后文追加的完整审计。旧目录/缓存只读；永久排除demo3 episode142。

## 训练合同与评估锁
已在训练前锁定；主27parent，附48窗；bootstrap2000 seed90；用户原判据不变。

## 资源与命令
GPU含失败≤3GPUh，新增≤10GB；共享GPU准入峰值1.5倍，OOM/紧张立即停止；setsid nohup分离。

## 结果、未完成与待决
见后文追加的完整审计。

## CPU合同核验与结果前锁定
{"status": "passed", "initial_optimizer_steps": [1542, 2000], "external_frozen_model_and_optimizer_unchanged": true, "no_robot_action_projection": true, "no_traj_equivalence_max_abs": 2.1457672119140625e-06, "P3_pool": 34777, "P1_pool": 107057, "external_windows": 29429, "expanded_windows": 211, "expanded_parents": 27, "original48": 48, "CUDA_initialized": false, "brt": "2026-11-15T13:00:08.533671-03:00"}
{"brt": "2026-11-15T13:00:08.538772-03:00", "initial_checkpoint": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a2_s90_20261115_03/ckpt_02000.pt", "initial_sha": "45e4118ffec7ad6c13508c4839a324deecb2fe87a3a02294ae064181f6c4eb4d", "plan_sha": "cb57feeaf3abf4e0901abdc2f8a3b20688c7ece588575d9b5f7c510635dd71c5", "seed": 90, "lr": 0.0003, "optimizer": "restore A2 AdamW; never reset including external->Piper; frozen grads=None skip momentum and decay", "F2": "Piper10000; save5000=F0 and10000=F2", "F1": "external5000 then Piper5000; save external end=F1-pre", "external": "batch32 ABC16 yam16; task round robin, parent equal; fixed generic E103-D instruction; no_traj physically deletes all robot/state/A tokens; world MSE+.1cos+.1adapter; no action/grip/chain loss; action/robot/state/proprio/grip/delta projections frozen", "Piper": "same A2 P3 batch32 + independent Piper P1 batch32 weight.1 from E103-D; P3 pool34777 P1pool107057 unchanged; parent equal demo3 cap.45; P3 seed90 P1seed10390; paired sampler prefixes and per-step RNG; horizons/global schedule2000+Piper step; C2 chain .1 predicted reentry .8", "recipe_disclosure": "A2 source P3-only; per explicit current P1+P3 request added paired full Piper P1 auxiliary .1 matching E103-D, disclosed before training; no substitution/reduction of Piper", "evaluation": "one inherited E102-X evaluator, checkpoint own A2 deterministic actions for pred_N1; all4 methods/all4 horizons; fixed human short language; main expanded27, original48 appendix; parent bootstrap2000 seed90", "criterion": "F1-F0 AND F1-F2: at least TWO SAME no_traj horizons among25/50/100 with lower CI>0; both comparisons all gt_action25/50/100 lower CI>-.01; joint L1 at8/50 and event F1 at5/10 point relative degradation <=5%; F1>F0 only: gain explainable by extra steps", "GPU_seconds_cap": 10800, "disk_bytes_cap": 10000000000, "peak_mib": 6144, "checkpoints": {"A2": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a2_s90_20261115_03/ckpt_02000.pt", "F1-pre": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f1_s90_20261115_01/ckpt_pre_05000.pt", "F0": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f2_s90_20261115_01/ckpt_05000.pt", "F1": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f1_s90_20261115_01/ckpt_05000.pt", "F2": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f2_s90_20261115_01/ckpt_10000.pt"}}

## 管线停止
```
Traceback (most recent call last):
  File "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e104p/e104p_pipeline.py", line 22, in <module>
    try:main()
        ^^^^^^
  File "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e104p/e104p_pipeline.py", line 15, in main
    assert not (CONTROL/'PIPELINE.json').exists();shared('PREPARING','新路径检查通过，报告骨架已建；预算3GPUh/10GB；预注册CPU核验后锁定，尚无本轮评分。');state('preflight');run('e104p_preflight.py',[],'preflight.log')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

```
必需步骤未全部完成，不作外部有效结论。失败计入GPUh，旧输出保留。

## 结果前连续优化器审计修订
run01开始后发现旧base.calibrate会执行额外reentry Adam更新，不符合本轮固定步数与连续AdamW。只停止本会话进程组1849104；无dev评分。旧源快照与run01保留。旧GPU含30s余量共76.456343秒。run02仅CPU启动因路径修订未完成退出（0 GPU秒，日志保留）。有效run03从同A2重启，只检查训练heldout adapter门限，无额外校准或优化器；AdamW全程连续。其余配方/判据不变。

## CPU合同核验与结果前锁定
{"status": "passed", "initial_optimizer_steps": [1542, 2000], "external_frozen_model_and_optimizer_unchanged": true, "no_robot_action_projection": true, "no_traj_equivalence_max_abs": 2.1457672119140625e-06, "P3_pool": 34777, "P1_pool": 107057, "external_windows": 29429, "expanded_windows": 211, "expanded_parents": 27, "original48": 48, "CUDA_initialized": false, "brt": "2026-11-15T13:03:06.806608-03:00"}
{"brt": "2026-11-15T13:03:06.811448-03:00", "initial_checkpoint": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a2_s90_20261115_03/ckpt_02000.pt", "initial_sha": "45e4118ffec7ad6c13508c4839a324deecb2fe87a3a02294ae064181f6c4eb4d", "plan_sha": "cb57feeaf3abf4e0901abdc2f8a3b20688c7ece588575d9b5f7c510635dd71c5", "seed": 90, "lr": 0.0003, "optimizer": "restore A2 AdamW; never reset including external->Piper; frozen grads=None skip momentum and decay", "F2": "Piper10000; save5000=F0 and10000=F2", "F1": "external5000 then Piper5000; save external end=F1-pre", "external": "batch32 ABC16 yam16; task round robin, parent equal; fixed generic E103-D instruction; no_traj physically deletes all robot/state/A tokens; world MSE+.1cos+.1adapter; no action/grip/chain loss; action/robot/state/proprio/grip/delta projections frozen", "adapter_policy": "No extra calibration optimizer or updates; check train-heldout thresholds only; continuous restored AdamW", "Piper": "same A2 P3 batch32 + independent Piper P1 batch32 weight.1 from E103-D; P3 pool34777 P1pool107057 unchanged; parent equal demo3 cap.45; P3 seed90 P1seed10390; paired sampler prefixes and per-step RNG; horizons/global schedule2000+Piper step; C2 chain .1 predicted reentry .8", "recipe_disclosure": "A2 source P3-only; per explicit current P1+P3 request added paired full Piper P1 auxiliary .1 matching E103-D, disclosed before training; no substitution/reduction of Piper", "evaluation": "one inherited E102-X evaluator, checkpoint own A2 deterministic actions for pred_N1; all4 methods/all4 horizons; fixed human short language; main expanded27, original48 appendix; parent bootstrap2000 seed90", "criterion": "F1-F0 AND F1-F2: at least TWO SAME no_traj horizons among25/50/100 with lower CI>0; both comparisons all gt_action25/50/100 lower CI>-.01; joint L1 at8/50 and event F1 at5/10 point relative degradation <=5%; F1>F0 only: gain explainable by extra steps", "GPU_seconds_cap": 10800, "disk_bytes_cap": 10000000000, "peak_mib": 6144, "checkpoints": {"A2": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a2_s90_20261115_03/ckpt_02000.pt", "F1-pre": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f1_s90_20261115_03/ckpt_pre_05000.pt", "F0": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f2_s90_20261115_03/ckpt_05000.pt", "F1": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f1_s90_20261115_03/ckpt_05000.pt", "F2": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f2_s90_20261115_03/ckpt_10000.pt"}}

## 数据、实现与合同说明
ABC缓存390 episode /28801窗，yam43 episode /628窗；总29429外部窗。原缓存约8.85GB仅只读mmap复用，本轮新增特征0字节。ABC约9.9983小时，yam标称0.24418小时；本轮选择样本配额1:1而非小时比例。外部每步ABC16+yam16，保留E103-D任务轮转、parent均匀、合法窗均匀方式和固定通用短指令。相机沿用top/left/right至Piper high/left_wrist/right_wrist映射；沿用原时间闸门、特征标准化、空间池化和源排除，不声称跨机器人几何等价。
Piper P3原池34777窗/145parent、P1原池107057窗/242parent完整保留，每个Piper优化步各采32窗，P1世界辅助权重0.1。A2旧源码仅P3，按本轮明确P1+P3要求添加E103-D同配方P1，已在训练前披露并提供澄清选项；截至锁定无回复，按此假设执行。不得把本轮F2解释为严格逐位重放旧A2轨迹；F1与F0为本轮同Piper数据严格配对。
P3四模式各概率0.25；Piper时域和模式日程取2000+local，C2链权重0.1，预测重入率0.8；现有夹爪训练阈值、两态开度、因果本体历史、确定性delta头不变。恢复A2原AdamW，初始参数优化器步数为1542或2000（因条件路径部分参数没有每步梯度），不将其强行统一或重置。外部无动作阶段世界loss=标准化MSE+0.1cos+0.1adapter；无动作/夹爪/链损失；冻结action/state/robot/proprio/delta/grip投影及头，grad=None同时跳过Adam动量及权重衰减。
所有Piper阶段sampler种子相同（P3=90、P1=10390），P3/P1样本序列各有滚动SHA；每步使用独立确定性Torch随机流防止不同adapter门控影响后续样本噪声。外部完成后只重置Piper采样随机流，不重置模型或优化器状态；外部训练不会推进Piper采样器。F2前5000与F1微调5000还检查模式计数及样本/时域联合哈希。
评估pred为每个checkpoint自身的A2确定性动作，沿用E102-X脚本；oracle仅作诊断，不作选点。四时域+10/+25/+50/+100；事件一一匹配、无事件parent指标为0、重叠窗按暴露时长、真值反馈运动起点代理等原合同不变。冻结集合不加模型文本。
项目AGENTS旧天机54D/仿真说明由用户本轮明确Piper14D/不仿真要求覆盖。无上位机操作、无新VLM、无下载、无仿真、无平移动作或标签。E98-C目录与进程未触碰。
有效命令：setsid（start_new_session=True）+nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e104p_pipeline.py；PID1884687，控制results/e104p_control_20261115_03；新训练run均为s90_20261115_03。

## 仅调度并行化（模型/配方锁不变）
{"brt": "2026-11-15T13:10:00.875489-03:00", "change": "Only parallel orchestration; no change to training/eval/data/loss/seed/optimizer/checkpoints/decision", "serial_parent_stopped": 1884687, "existing_F2_pid": 1890924, "F2_gpu": {"index": 5, "uuid": "GPU-5e231c95-8cec-b63f-a071-4acf439637ed", "total": 97871.0, "free": 97206.0, "util": 0.0}, "F1_gpu": {"index": 7, "uuid": "GPU-0345da1d-abf6-b973-f49d-c1023ae9a97d", "total": 97871.0, "free": 96530.0, "util": 0.0}, "orchestrator_sha": "43fd9493db248ce70621d0b96d6440a0170c998fb5147c7e31f75abb8c218483", "global_GPU_seconds_stop": 10400, "includes_all_prior_failures": true}

### 评估解释边界
关节真值为原始next-frame follower反馈，不是发送命令；gt_action条件使用原50帧反馈动作与训练两态夹爪、51..100终端保持。pred条件使用每个checkpoint的A2确定性delta+两态夹爪动作，亦保持终端到100；oracle使用未来实测状态及实测阈值两态，是诊断项。事件沿用反馈运动起点代理及原训练阈值；不声称发送命令真值、策略成功率或实机验证。+100短episode的3窗无目标，继续排除，不补造帧、不移标签。

## F1 外部阶段完成
{"counts": {"abc": 80000, "yam": 80000}, "frozen_unchanged": true, "checkpoint": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f1_s90_20261115_03/ckpt_pre_05000.pt", "elapsed_s": 566.2584554892965}

## F2 Piper 5000步
{"status": "midpoint_saved", "step": 5000, "P3_samples_sha256": "d28d1a5e8a77648897f0d2f186df8fbc8e90448ab40c7b8bcfdd9ab4b09aa502", "P1_samples_sha256": "8ccbcf02a09c32240b99005e6e1c0d2a4bd0d46b981caa6a46b293dd6b4219dc", "P3_flow_sha256": "6cb84bda7ce80f396e1918e069b684de58de245a3be989a14cfccbfae798d4e2", "modes": {"independent": 1236, "clean": 1195, "sync": 1332, "action_only": 1237}, "P3_draws": 160000, "P1_draws": 160000, "P3_pool": 34777, "P1_pool": 107057, "elapsed_s": 1756.0432589701377, "peak_mib": 3192.8134765625, "adapter_initial": true, "last": {"step": 5000, "global_step": 7000, "loss": 0.7822158336639404, "p1_loss": 1.2752652168273926, "horizons": [10, 25, 50, 100, 31], "grip_bce": 0.05631203576922417, "delta_mse": 0.303477942943573, "mode": "clean", "action": 0.5160271525382996, "world": 1.0881773233413696, "cosine": 0.8846686482429504, "adapter": 0.06267794221639633, "alignment": 1.116133689880371, "chain": 0.9209630489349365, "chain_weight": 0.1, "chain_supervised": 0.9209630489349365, "chain_gap": 0.0, "predicted_reentry": false, "chain_k": 25}, "brt": "2026-11-15T13:32:41.709300-03:00"}

## F1 Piper 5000步
{"status": "completed", "step": 5000, "P3_samples_sha256": "d28d1a5e8a77648897f0d2f186df8fbc8e90448ab40c7b8bcfdd9ab4b09aa502", "P1_samples_sha256": "8ccbcf02a09c32240b99005e6e1c0d2a4bd0d46b981caa6a46b293dd6b4219dc", "P3_flow_sha256": "6cb84bda7ce80f396e1918e069b684de58de245a3be989a14cfccbfae798d4e2", "modes": {"independent": 1236, "clean": 1195, "sync": 1332, "action_only": 1237}, "P3_draws": 160000, "P1_draws": 160000, "P3_pool": 34777, "P1_pool": 107057, "elapsed_s": 2255.1042080479674, "peak_mib": 3188.6455078125, "adapter_initial": true, "last": {"step": 5000, "global_step": 7000, "loss": 0.7891397476196289, "p1_loss": 1.2835150957107544, "horizons": [10, 25, 50, 100, 31], "grip_bce": 0.06514160335063934, "delta_mse": 0.29742559790611267, "mode": "clean", "action": 0.5169749855995178, "world": 1.1412113904953003, "cosine": 0.8959575891494751, "adapter": 0.05319295823574066, "alignment": 1.0929324626922607, "chain": 0.9271238446235657, "chain_weight": 0.1, "chain_supervised": 0.9271238446235657, "chain_gap": 0.0, "predicted_reentry": false, "chain_k": 25}, "brt": "2026-11-15T13:47:55.691335-03:00"}

## F2 Piper 10000步
{"status": "completed", "step": 10000, "P3_samples_sha256": "85783787ca7549ae772075bffcb2dea4898b422525c145d1507b629bff1bdaff", "P1_samples_sha256": "1eed8d48a4c7ec41ac739bf358c673b757daefb9a533f6a1f28a3874c7bf0b7f", "P3_flow_sha256": "e0a353e824f5c6f1f35c1104aeab6636ed6d069fa609c5d28880ba9ff03efd7f", "modes": {"independent": 2523, "clean": 2444, "sync": 2563, "action_only": 2470}, "P3_draws": 320000, "P1_draws": 320000, "P3_pool": 34777, "P1_pool": 107057, "elapsed_s": 3550.756659972947, "peak_mib": 3192.8134765625, "adapter_initial": true, "last": {"step": 10000, "global_step": 12000, "loss": 0.6930510401725769, "p1_loss": 1.3119345903396606, "horizons": [10, 25, 50, 100, 48], "grip_bce": 0.030784940347075462, "delta_mse": 0.30034109950065613, "mode": "action_only", "action": 0.552814781665802, "world": -0.0, "cosine": -0.0, "adapter": -0.0, "alignment": 0.904279887676239, "chain": -0.0, "chain_weight": 0.0}, "brt": "2026-11-15T14:02:36.422506-03:00"}

## 同Piper数据逐样本核验
{"P3_samples_sha256": true, "P1_samples_sha256": true, "P3_flow_sha256": true, "modes": true, "P3_draws": true, "P1_draws": true}

## 固定终点评估结果
主集211窗/27parent；+100为208窗/26parent。附表为原48窗/6parent。所有汇总parent等权，配对bootstrap2000 seed90。pred_N1为每个checkpoint的A2确定性动作，参数化及脚本一致。状态/事件为原训练拟合合同和反馈运动起点代理；49个窗口真事件、48个去重物理事件。无模型语言、无新阈值、无选点。

### all27
|组|h|GT / pred / no_traj / oracle skill %|
|---|---:|---|
|A2|10|12.420/10.802/4.279/12.432|
|A2|25|21.067/18.756/6.550/21.055|
|A2|50|28.313/23.293/8.880/28.313|
|A2|100|41.637/37.510/16.198/43.619|
|F1-pre|10|-14.363/-14.568/-11.215/-14.372|
|F1-pre|25|-12.377/-12.617/-8.406/-12.336|
|F1-pre|50|-6.087/-6.983/-3.776/-6.054|
|F1-pre|100|-0.710/-1.403/-4.574/-0.477|
|F0|10|13.163/11.457/9.431/13.168|
|F0|25|21.600/18.884/15.449/21.588|
|F0|50|27.965/23.124/20.671/27.977|
|F0|100|42.265/38.199/36.661/44.198|
|F1|10|12.123/10.832/8.441/12.117|
|F1|25|19.921/17.915/13.251/19.913|
|F1|50|25.542/22.030/18.722/25.535|
|F1|100|40.859/37.908/35.892/42.598|
|F2|10|13.669/12.334/10.345/13.660|
|F2|25|22.675/20.543/16.431/22.657|
|F2|50|29.297/24.479/22.213/29.313|
|F2|100|43.259/38.857/37.893/44.720|

|组|前8/H50关节L1|相对复制改善% 前8/H50|状态准确率|事件F1 ±5/±10|假切换/分钟 ±5/±10|
|---|---|---|---|---|---|
|A2|0.009193/0.046324|14.22/22.34|0.9402|0.1785/0.2612|10.222/9.389|
|F1-pre|0.010056/0.050088|6.17/16.03|0.9445|0.0914/0.2259|11.333/10.167|
|F0|0.008331/0.044513|22.27/25.38|0.9393|0.1218/0.2560|9.333/7.833|
|F1|0.008298/0.043891|22.57/26.42|0.9408|0.1580/0.2761|8.333/7.167|
|F2|0.008085/0.042852|24.56/28.16|0.9362|0.1233/0.2503|10.000/8.500|

### original48
|组|h|GT / pred / no_traj / oracle skill %|
|---|---:|---|
|A2|10|11.034/10.264/3.505/11.114|
|A2|25|19.741/17.933/6.223/19.776|
|A2|50|27.607/23.370/9.096/27.619|
|A2|100|42.725/40.412/17.742/44.663|
|F1-pre|10|-14.596/-15.224/-11.099/-14.717|
|F1-pre|25|-12.492/-12.867/-7.736/-12.517|
|F1-pre|50|-4.845/-5.256/-2.271/-4.808|
|F1-pre|100|3.833/3.774/0.037/4.248|
|F0|10|12.073/11.237/8.032/12.134|
|F0|25|20.574/18.558/13.046/20.611|
|F0|50|27.148/22.392/18.355/27.190|
|F0|100|42.679/40.795/37.422/44.439|
|F1|10|10.751/10.329/7.131/10.780|
|F1|25|18.147/16.795/11.140/18.193|
|F1|50|23.799/20.617/16.440/23.813|
|F1|100|40.359/39.403/37.263/41.875|
|F2|10|12.447/11.668/9.078/12.483|
|F2|25|21.649/20.275/14.528/21.680|
|F2|50|28.214/23.255/20.216/28.260|
|F2|100|45.218/41.970/40.950/46.654|

|组|前8/H50关节L1|相对复制改善% 前8/H50|状态准确率|事件F1 ±5/±10|假切换/分钟 ±5/±10|
|---|---|---|---|---|---|
|A2|0.007999/0.039907|20.81/33.45|0.9587|0.3000/0.4667|9.000/7.500|
|F1-pre|0.008987/0.043709|11.03/27.11|0.9387|0.0833/0.4111|10.500/7.500|
|F0|0.007602/0.042022|24.74/29.92|0.9594|0.1037/0.5741|11.250/6.000|
|F1|0.007590/0.040936|24.86/31.73|0.9571|0.1500/0.5704|10.500/6.000|
|F2|0.007135/0.037915|29.36/36.77|0.9577|0.1407/0.4778|9.750/6.000|

## 配对差（主27parent；skill百分点pp，95%CI）
|比较|h|no_traj差 [CI]|gt_action差 [CI]|
|---|---|---|---|
|F1-F0|10|-0.989 [-1.448,-0.480]|-1.040 [-1.409,-0.576]|
|F1-F0|25|-2.198 [-3.029,-1.451]|-1.679 [-2.263,-0.903]|
|F1-F0|50|-1.949 [-2.576,-1.316]|-2.423 [-3.006,-1.760]|
|F1-F0|100|-0.769 [-1.430,-0.091]|-1.405 [-2.233,-0.587]|
|F1-F2|10|-1.904 [-2.370,-1.408]|-1.547 [-2.163,-0.673]|
|F1-F2|25|-3.180 [-4.120,-2.271]|-2.755 [-3.440,-1.763]|
|F1-F2|50|-3.491 [-4.482,-2.492]|-3.755 [-4.353,-3.060]|
|F1-F2|100|-2.001 [-3.385,-0.541]|-2.400 [-3.638,-1.269]|
|F1-pre-A2|10|-15.494 [-18.170,-13.385]|-26.783 [-29.502,-24.530]|
|F1-pre-A2|25|-14.956 [-17.313,-12.998]|-33.443 [-35.567,-31.447]|
|F1-pre-A2|50|-12.655 [-15.826,-10.110]|-34.400 [-36.958,-32.063]|
|F1-pre-A2|100|-20.772 [-25.587,-16.989]|-42.347 [-45.624,-39.458]|
|F0-A2|10|5.151 [4.097,6.204]|0.743 [0.390,1.072]|
|F0-A2|25|8.899 [6.998,11.114]|0.533 [0.110,0.881]|
|F0-A2|50|11.791 [9.583,14.487]|-0.348 [-0.646,-0.029]|
|F0-A2|100|20.463 [16.855,24.450]|0.627 [-0.301,1.662]|
|F2-F0|10|0.915 [0.594,1.255]|0.506 [-0.025,0.926]|
|F2-F0|25|0.982 [0.478,1.445]|1.076 [0.647,1.500]|
|F2-F0|50|1.542 [0.939,2.145]|1.332 [0.944,1.788]|
|F2-F0|100|1.232 [-0.120,2.461]|0.995 [0.246,1.728]|
|F2-A2|10|6.066 [4.940,7.117]|1.249 [0.490,1.818]|
|F2-A2|25|9.881 [8.033,11.857]|1.609 [0.984,2.131]|
|F2-A2|50|13.333 [10.969,15.980]|0.984 [0.565,1.435]|
|F2-A2|100|21.695 [18.243,25.304]|1.622 [0.870,2.393]|

### 动作/夹爪非劣诊断
|比较|L1相对劣化% 前8/H50|事件F1相对劣化% ±5/±10|
|---|---|---|
|F1-F0|-0.39/-1.40|-29.73/-7.88|
|F1-F2|2.64/2.42|-28.12/-10.34|
|F1-pre-A2|9.38/8.13|48.81/13.50|
|F0-A2|-9.38/-3.91|31.75/2.00|
|F2-F0|-2.95/-3.73|-1.25/2.23|
|F2-A2|-12.06/-7.49|30.90/4.19|

## 预注册判定
{"verdict": "未满足外部预训练有效判据", "effective": false, "same_positive_horizons": [], "checks": {"F1-F0": {"positive_no_traj": [], "gt_noninferior": false, "joint_noninferior": true, "gripper_noninferior": true}, "F1-F2": {"positive_no_traj": [], "gt_noninferior": false, "joint_noninferior": true, "gripper_noninferior": true}}, "bootstrap": 2000, "seed": 90, "main": "27parent; h10026parent", "endpoint_selection": false}

关节复制改善绝对CI、所有方法skill CI、所有动作/事件完整指标见SUMMARY.json；所有配对差含原48附表见PAIRED.json。F2−F0、F2−A2仅描述多训练收益，不选点。F1-pre−A2用于直接迁移和遗忘诊断。单seed，不估计seed变异；仅冻结dev结论，不声称实机成功。

## 资源与最终状态
{"status": "COMPLETE", "brt": "2026-11-15T14:09:45.579692-03:00", "GPUh_upper": 1.7811127590949931, "new_bytes": 1131372330, "segments": [{"phase": "eval_F2", "pid": 2760524, "gpu": {"index": 7, "uuid": "GPU-0345da1d-abf6-b973-f49d-c1023ae9a97d", "total": 97871.0, "free": 41246.0, "util": 100.0}, "started_brt": "2026-11-15T14:08:26.922226-03:00", "status": "closed", "elapsed_s": 71.31000094488263, "peak_estimate_mib": 6144, "finished_brt": "2026-11-15T14:09:37.936926-03:00"}, {"phase": "eval_A2", "pid": 2694132, "gpu": {"index": 1, "uuid": "GPU-3e39ad77-5859-2f52-e3f8-a2ae3bdc8cd1", "total": 97871.0, "free": 60572.0, "util": 0.0}, "started_brt": "2026-11-15T14:03:08.350876-03:00", "status": "closed", "elapsed_s": 59.9487827019766, "peak_estimate_mib": 6144, "finished_brt": "2026-11-15T14:04:08.070135-03:00"}, {"phase": "train_F2", "pid": 1890924, "gpu": {"index": 5, "uuid": "GPU-5e231c95-8cec-b63f-a071-4acf439637ed", "total": 97871.0, "free": 97206.0, "util": 0.0}, "started_brt": "2026-11-15T13:03:25.779937-03:00", "status": "closed", "elapsed_s": 3550.779076660052, "peak_estimate_mib": 6144, "finished_brt": "2026-11-15T14:02:36.438455-03:00"}, {"phase": "eval_F1-pre", "pid": 2711269, "gpu": {"index": 1, "uuid": "GPU-3e39ad77-5859-2f52-e3f8-a2ae3bdc8cd1", "total": 97871.0, "free": 60572.0, "util": 0.0}, "started_brt": "2026-11-15T14:04:38.139078-03:00", "status": "closed", "elapsed_s": 48.20125333592296, "peak_estimate_mib": 6144, "finished_brt": "2026-11-15T14:05:26.101799-03:00"}, {"phase": "eval_F1", "pid": 2738604, "gpu": {"index": 5, "uuid": "GPU-5e231c95-8cec-b63f-a071-4acf439637ed", "total": 97871.0, "free": 45802.0, "util": 100.0}, "started_brt": "2026-11-15T14:06:40.274911-03:00", "status": "closed", "elapsed_s": 92.82433524401858, "peak_estimate_mib": 6144, "finished_brt": "2026-11-15T14:08:12.963955-03:00"}, {"phase": "train_F1", "pid": 1963599, "gpu": {"index": 7, "uuid": "GPU-0345da1d-abf6-b973-f49d-c1023ae9a97d", "total": 97871.0, "free": 96530.0, "util": 0.0}, "started_brt": "2026-11-15T13:10:20.834573-03:00", "status": "closed", "elapsed_s": 2255.1370527590625, "peak_estimate_mib": 6144, "finished_brt": "2026-11-15T13:47:55.715515-03:00"}, {"phase": "eval_F0", "pid": 2727268, "gpu": {"index": 1, "uuid": "GPU-3e39ad77-5859-2f52-e3f8-a2ae3bdc8cd1", "total": 97871.0, "free": 60572.0, "util": 0.0}, "started_brt": "2026-11-15T14:05:39.763071-03:00", "status": "closed", "elapsed_s": 47.34908809605986, "peak_estimate_mib": 6144, "finished_brt": "2026-11-15T14:06:26.900119-03:00"}], "checkpoint_cache_old_code_readonly": true}
未完成必需项：无。需要用户决定：无。外部有效性严格按预注册判定；不部署。

## 关节L1与相对复制改善补充CI
仅既定指标的CPU汇总；parent bootstrap2000 seed90，无新推理、阈值或判据。
|集|组|前缀|L1 rad [95%CI]|相对复制改善% [95%CI]|
|---|---|---|---|---|
|all27|A2|8|0.009193 [0.007822464259843419, 0.010933500031244705]|14.22 [6.99, 19.32]|
|all27|A2|50|0.046324 [0.040830208259741346, 0.05316943822154354]|22.34 [16.08, 27.66]|
|original48|A2|8|0.007999 [0.00733993061243786, 0.00856946563221249]|20.81 [11.17, 28.38]|
|original48|A2|50|0.039907 [0.037483368312577016, 0.041971611550858746]|33.45 [20.64, 40.77]|
|all27|F1-pre|8|0.010056 [0.008631539821210059, 0.011895608915978606]|6.17 [-2.98, 12.91]|
|all27|F1-pre|50|0.050088 [0.043960102820889406, 0.0578756517849562]|16.03 [10.05, 21.13]|
|original48|F1-pre|8|0.008987 [0.008272218128796442, 0.009562981518481846]|11.03 [-4.45, 22.89]|
|original48|F1-pre|50|0.043709 [0.03935524295467682, 0.048157245968321945]|27.11 [12.16, 36.13]|
|all27|F0|8|0.008331 [0.00707941764073985, 0.009864091440163373]|22.27 [16.45, 26.33]|
|all27|F0|50|0.044513 [0.03927240201240767, 0.050465870941684025]|25.38 [19.2, 30.85]|
|original48|F0|8|0.007602 [0.006670686932718834, 0.0084400565101027]|24.74 [18.59, 29.74]|
|original48|F0|50|0.042022 [0.03759925198817452, 0.04614700965703378]|29.92 [16.6, 37.6]|
|all27|F1|8|0.008298 [0.007062275151062491, 0.009817429538476634]|22.57 [15.74, 27.34]|
|all27|F1|50|0.043891 [0.039215913622981276, 0.04947507213013365]|26.42 [19.61, 32.22]|
|original48|F1|8|0.007590 [0.0064247443949195975, 0.008577644810884522]|24.86 [18.66, 30.82]|
|original48|F1|50|0.040936 [0.03530397780084766, 0.04625969046770075]|31.73 [21.53, 39.09]|
|all27|F2|8|0.008085 [0.006765117232905911, 0.009681575002932614]|24.56 [18.85, 28.93]|
|all27|F2|50|0.042852 [0.03740892278605338, 0.049442976971234086]|28.16 [22.2, 33.52]|
|original48|F2|8|0.007135 [0.006394101906506775, 0.007882484383799236]|29.36 [18.35, 38.31]|
|original48|F2|50|0.037915 [0.035157016134529484, 0.04055663981621617]|36.77 [21.36, 44.89]|

## 最终CPU审计
{"status": "passed", "brt": "2026-11-15T14:11:20.009871-03:00", "checks": {"locked_code_unchanged": true, "locked_inputs_unchanged": true, "E102X_windows_captures_gripper_unchanged": true, "old_A2_core_matches_initial_copy": true, "all_GPU_segments_closed": true, "paired_Piper_stream": true, "external_frozen_parameters_AND_AdamW_state_exact": true, "continuous_optimizer_step_counts": true, "A2_replays_E102X_exactly": true}, "optimizer_steps": {"action_head.weight": {"A2": 2000, "F1-pre": 2000, "F0": 7000, "F1": 7000, "F2": 12000}, "delta_head.weight": {"A2": 2000, "F1-pre": 2000, "F0": 7000, "F1": 7000, "F2": 12000}, "grip_head.weight": {"A2": 2000, "F1-pre": 2000, "F0": 7000, "F1": 7000, "F2": 12000}, "mean_head.0.weight": {"A2": 1542, "F1-pre": 6542, "F0": 6542, "F1": 11542, "F2": 11542}, "robot_proj.weight": {"A2": 1542, "F1-pre": 1542, "F0": 5305, "F1": 5305, "F2": 9072}}, "A2_replay": {"world_max_abs": 0.0, "action_max_abs": 0.0, "events_equal": true}, "GPUh_upper": 1.7811127590949931, "new_bytes": 1131382289, "new_bytes_upper_including_report_temporary_and_local_artifacts": 1141893746, "audit_sha256": "6bc07efe15f5575bbcfb46e11796d8a8815cc8808faa53e890f577e04416a2a8", "audit": "CPU only; no additional inference/training/checkpoint selection"}
所有锁定代码/输入SHA未变；原A2在27parent的动作与全部四时域世界MSE逐位复现E102-X；外部阶段冻结权重和AdamW状态逐位不变；关键头步数验证连续优化器未重置。仅CPU，无新增推理、训练或选点。有效主调度命令nohup e29/bin/python -u e104p_parallel.py（start_new_session=True），PID1963577；训练代码与初始锁一致。


## 最终中文结论与交付
完成：2026-11-15T14:14:21.687587-03:00 BRT；有效run03。未满足外部预训练有效判据。本设定下外部预训练不仅未带来正增益，Piper微调后世界skill仍显著低于匹配对照。
F1-F0 no_traj：+25: -2.198pp [-3.029,-1.451]；+50: -1.949pp [-2.576,-1.316]；+100: -0.769pp [-1.430,-0.091]。
F1-F2 no_traj：+25: -3.180pp [-4.120,-2.271]；+50: -3.491pp [-4.482,-2.492]；+100: -2.001pp [-3.385,-0.541]。
F1-pre-A2 no_traj：+25: -14.956pp [-17.313,-12.998]；+50: -12.655pp [-15.826,-10.110]；+100: -20.772pp [-25.587,-16.989]。
F2-F0 no_traj：+25: +0.982pp [+0.478,+1.445]；+50: +1.542pp [+0.939,+2.145]；+100: +1.232pp [-0.120,+2.461]。
F2-A2 no_traj：+25: +9.881pp [+8.033,+11.857]；+50: +13.333pp [+10.969,+15.980]；+100: +21.695pp [+18.243,+25.304]。
F1−F0及F1−F2的no_traj三时域95%CI均完全小于0；gt_action三时域非劣性也均未通过（要求CI下界>−1pp）。F1相对两对照的关节前8/H50 L1和夹爪±5/±10事件F1均满足5%非劣性。无需用“增益可由多训练步数解释”描述本结果，因为F1连同Piper数据对照F0也没有增益。
F1-pre出现明显Piper直接迁移遗忘（无动作skill在+25/+50/+100降为−8.406/−3.776/−4.574%），微调后恢复大部但未追平F0/F2；动作头与机器人投影本身未更新并不能避免共享骨干遗忘。
训练更久仅作描述：F2−F0 no_traj在+25/+50 CI下界>0，+100 CI跨0；F2−A2三时域均提升。动作并非所有指标同向改善：F2前8/H50关节L1=0.008085/0.042852 rad（A2=0.009193/0.046324），事件F1±5/±10=0.12334/0.25026（A2=0.17848/0.26120；F0=0.12181/0.25597）。不把世界头训练收益推广为整策略夹爪事件全面改善。
资源：含失败和每段余量GPUh上界1.781113；代码/results实测约1.132GB，连同报告、临时脚本、本地副本和共享日志增量保守计≤1.15GB，低于10GB；新增特征0字节。全部7个有效GPU段已关闭，旧失败段计入PRIOR_GPU_USAGE.json。
审计：A2动作指标与全部世界MSE重放差0、事件相同；冻结参数及其AdamW状态逐位不变；动作头A2/F1-pre/F0/F1/F2步数2000/2000/7000/7000/12000，世界mean头1542/6542/6542/11542/11542。Piper样本哈希、时域联合哈希、模式计数、采样总数完全一致。所有原锁代码/输入SHA未变，五组实际evaluator源码SHA完全相同。
环境：{"brt": "2026-11-15T14:14:21.687587-03:00", "evaluator_identical": true, "evaluator_hashes": {"A2": "82651a7a2db82c57424dda5c246091f44454712cb1cd34dda60aa05b8dfd5a35", "F1-pre": "82651a7a2db82c57424dda5c246091f44454712cb1cd34dda60aa05b8dfd5a35", "F0": "82651a7a2db82c57424dda5c246091f44454712cb1cd34dda60aa05b8dfd5a35", "F1": "82651a7a2db82c57424dda5c246091f44454712cb1cd34dda60aa05b8dfd5a35", "F2": "82651a7a2db82c57424dda5c246091f44454712cb1cd34dda60aa05b8dfd5a35"}, "python": "3.11.16", "torch": "2.11.0+cu130", "cuda_build": "13.0", "CPU_only": true}
交付目录：T/code_e104p；T/results/e104p_control_20261115_03；T/results/e104p_f1_s90_20261115_03；T/results/e104p_f2_s90_20261115_03。详表SUMMARY.json、配对PAIRED.json、关节复制改善RELATIVE_COPY.json、判定DECISION.json、审计FINAL_AUDIT.json。CPU审计命令：nohup e29/bin/python -u e104p_audit.py（CUDA_VISIBLE_DEVICES为空）。
未完成必需项：无。需要用户决定：无。本轮不自动更换默认A2基线；未做实机部署。
