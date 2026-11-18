# E88-W v3.1（仅 Piper 工程验证）

新实现 `e88/`，复制的 `e86/` 保留原模型行为，仅用作读取和对照。不要执行旧 e86 的写入入口（它的 C 常量仍指 code_e86）；所有本轮写入使用 `e88.common.OUT`。原 code_e86、code_e82、rev4/V1/V3、E83、特征缓存只读；episode142永久拒绝。无下载、无LoRA、无正式训练入口。

解释器 `/mnt/workspace/dengjianfeng/venvs/e29/bin/python`；cwd=`T/code_e88`，`PYTHONPATH=.`。只使用新的唯一 run ID：

```sh
python -m pytest -q tests
python -m e88.train NEW_UNIQUE_ID --recipe A
python -m e88.replay NEW_UNIQUE_ID P1
python -m e88.replay NEW_UNIQUE_ID P3_short
python -m e88.profile NEW_UNIQUE_ID
```

train 固定为 P1 160、P3 none/short/long 各192更新，batch8，不提供5000/10000正式启动选项。新GPU调用都进入同一个9000秒台账，每段≤3500秒；要求准入util=0、空闲显存≥8192MiB×1.5，监控共享显存压力、实测峰值和预算；不操作他人进程。全部停止段均保留且计入预算。smoke_01为P1指标掩码修正前，smoke_02为满维噪声旁路修正前；最终使用smoke_03。

## 模型合同

- 同一6×256/8头/FFN1024共享主干；O仅读O。A可读O/A/W，短W可读O/A/全部W，长W只读O/长W；额外禁止长W读A，阻断间接短W噪声泄露。
- 10/25/50每相机2×4目标；100/200/300每相机global目标；固定六时域81个W token，每batch额外在相邻时域区间抽整数源帧偏移（不插值真值），最多105 W token。h/300连续正弦嵌入，不是离散查表。
- y=(z_future-z_current)/scale；scale为P1允许训练parent均衡统计的每camera/token RMS变化幅度，floor=.05；随机h在同分辨率内插值scale（51..99沿用100尺度）。统计样本、refs、SHA写contract。固定编码无可训练目标投影。
- 世界FM采用`v=(clean_residual_head-w)/(1-tau)`满维768噪声旁路（分母floor=.01），clean残差头零初始化精确复制当前编码。避免768→256输入投影丢失的噪声坐标无法被速度头抵消；原始e86模型保持不变。
- 每时域FM分别按有效token均值，再有效时域均值；余弦仅短时域。确定性余弦和阶段头从只读O的clean W query预测，不读带噪未来。action_only物理删除W，同时关闭adapter/phase/chain。
- action维度和单位保持原合同：12关节当前q残差，夹爪signed absolute米，不以发送命令解释next-frame反馈。新复制基线保留raw负夹爪，同时报告旧q基线。

## 适配器与链

`reentry`是独立AttentionPool，输入固定2×4×768，输出每camera8×256（O当前token，尚未加入camera/age/kind）。同一真实帧蒸馏到`current(raw).detach()`，目标侧stop-grad。建议门槛：独立于拟合64窗的16个训练窗，归一化MSE≤0.1且余弦≥0.95。每16更新复查，未过门槛不执行主训练链损失。门槛仅依真实图像特征，不依模型审核/语言标签。

链25+25→50、50+50→100、100+100→200：第一跳stop-grad；预测端点重入adapter；历史移到a+k并只读真实因果可得帧，缺缓存整时刻mask；A和state不进入链。第二跳监督真实+2k端点，加直接+2k预测stop-grad一致性；跨细/粗的链按camera平均。100首跳的global预测显式广播到2×4重入，无法恢复细空间信息，须单独观察其误差。

默认收敛门`p1_converged=False`关闭主链；只有外部明确声明收敛，或显式`engineering_chain_probe=True`工程探针才开启，后者不代表收敛。链最大权重.1；本次工程烟雾P1边界128、ramp128，P3从阶段起点ramp128，并每4更新抽一条链、至多2窗、2 Euler步。真实/预测重入按.2→.8计划采样。**这不是P1已收敛的证明**；正式实验必须先明确P1收敛，再启ramp。`chain_weight`、`loss`及`chain_loss`参数允许独立设置warmup/ramp/max/步数/混合；本入口只运行锁定烟雾配置。

## 语言、阶段与评估

none=删除文本条件及align；short=同一人工短任务指令；long=现有E83长任务指令+已达到cutoff的过程叙述。仅训练parent有模型语言，dev仅既有人工指令。E83 narration的cutoff必须存在且≤anchor；当前阶段还要求anchor位于该段start/end内。未知mask，不根据未来阶段编号或全片进度补标签。阶段编号是原叙述段序号，不声称统一语义ontology；进度只在已因果放行的当前段内计算。

世界指标只做固定训练窗工程诊断（每层最多4窗，含长时域可用条件筛选），不作dev真值/阈值或checkpoint选型。报告每时域model/copy/linear归一化残差MSE，skill=1-model/copy（另报linear skill）；线性基线仅用当前及最近有效历史真实编码，按每路真实曝光时间外推。dev动作沿用6parent/48窗，parent等权，固定阶段末。无语言效果结论。

## 配方

默认A：sync/independent/action_only/clean=30/30/30/10，前两者25%概率加干净动作W辅助，主辅W平均。B：25/25/25/25、aux=0，可用`--recipe B`选择，当前未启动B对照。两者期望clean暴露均25%，但W主更新比例70%→75%，世界/adapter/phase/chain平均暴露约+7.14%；W前向85→75次/100更新，约-11.76%，clean额外A前向10→25次，两者总前向一阶均约125/100更新，新增链会使B稍贵。正式口径待用户锁定。

## 证据

`artifacts/cpu_audit_01/`含全池可用性/缓存帧需求/残差统计/阶段覆盖/夹爪过拟合；`contracts_*.log`保留全部轮次；新smoke contract绑定源hash、采样、固定终点和RNG/optimizer。`gpu_ledger.json`累计失败和成功。最终报告 `T/notes/e88_W_v31_20261114.md`。
