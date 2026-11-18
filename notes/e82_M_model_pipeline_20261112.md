# E82-M Piper 统一模型与流水线

- BRT 日期：2026-11-12；开始：10:19 BRT；最终审计：2026-11-12T13:40:37.566534-03:00。
- 范围：仅 Piper；新建 code_e82，不修改旧工具，不启动正式训练。
- 用户本轮 E79/E81 显式合同优先于根 AGENTS.md 的旧 54D/冻结视觉/仿真方向。

## 1. 来源与精确实现合同
已在下方对应进展与§7–10完成记录。

## 2. 参数量与 token 表
已在下方对应进展与§7–10完成记录。

## 3. 合同测试
已在下方对应进展与§7–10完成记录。

## 4. 数据入口与真实数据验证
已在下方对应进展与§7–10完成记录。

## 5. Profile、延迟与预算
已在下方对应进展与§7–10完成记录。

## 6. 命令、文件与未完成项
已完成代码交付；正式训练未执行，真实GPU profile资源阻塞，见§9–10。

### 进展 A：来源与资源（10:21 BRT）
- 已读 E81 §1、§2.3–2.5、§3、§4、§5.1 及 E79 §4；E77/E80 定位引用。
- 原始权重已存在：/mnt/workspace/dengjianfeng/models/dinov2_official_20261031/dinov2_vitb14_pretrain.pth，346,378,731 B。不下载，不用任何任务 LoRA 初始化。
- Python：/mnt/workspace/dengjianfeng/venvs/e29/bin/python；torch 2.11.0+cu130 / timm 1.0.29 / pytest 9.1.1。
- 首次 GPU 快照所有卡显存均被占用，未做CUDA模型计算（早期RNG上下文风险见§9.1）；/mnt 98%，约2.8T可用。
- 启动只读 READY watcher，每600秒检查，最多4小时；code_e82/artifacts/data_wait*.json* 记录。不自动启动训练。

### 进展 B：模型/合同（2026-11-12 10:36 BRT）

精确参数（原始 DINOv2-B 本地权重 CPU 实例化实测）：

|项目|参数|
|---|---:|
|共享 6×256 core（含末层 norm）|4,739,072|
|在线 DINOv2-B + LoRA|86,628,864|
|末2层 Q/V rank8 LoRA|49,152|
|冻结 E*|86,579,712|
|P3 总可训练|5,618,446|
|总参数（含在线/目标两塔）|178,777,870|

精确 token：顶视 4×6+global=25；左腕全图16+global1+原生ROI16=33；右腕同33。每时刻91，O=2×91+4状态+1任务=187；A=50；W=3×91=273。joint=510，action_only=237，P1=460（4状态槽保留预算但屏蔽，A长度严格0）。camera/age/missing/crop 坐标编码并入槽位，不额外增加 token；缺图的 missing 状态汇入始终存在的任务槽，图像patch严格屏蔽。

测试 artifacts/contracts_v3.txt：**23 passed, 1 warning, 20.33s**，CPU模型计算，资源审计偏差见§9.1。包含：O只读O及改未来不改O；W物理删除且不调用W输入投影；A/W双向影响；LA/LW分别到共享core/LoRA；E*零梯度；缺相机/状态；缓存与非缓存；joint/action_only均10 Euler×4固定噪声；同seed；因果真实Δt；signed absolute夹爪；R/P P3流；parent等权与mm；精确断点恢复；READY失效/篡改拒绝；P1全部A头保持S0；独立真值止损。

其中真实本地 DINO 集成测试2项，采用28×28小图仅验证参数/梯度图（不冒充生产分辨率速度）。其余合成合同使用同规格6×256核心和显式 fixture 视觉塔；fixture 权重不允许正式训练。真实 DINO 的 LA、LW 分别到 LoRA/core、E*无梯度均通过。

### 进展 C：流水线范围
- config schema 固定 Piper 14D/H50/30Hz，Tianji/R0仅保留扩展接口，未实现未宣称通过。
- S0；P1 5k×32；P3 1k warm-in+9k joint×32；R直接P3；阶段重建AdamW，5%warmup/cosine，LoRA3e-6，合并clip1。
- P3模式30/30/30/10，额外10% action_only LA；每动作样本一次主要LA；τ/噪声/增强/采样独立流，P3不含R/P臂名。
- 联合λ前500更新升至.1。梯度累积使用有效batch内LW监督样本数作分母，不随microbatch改变有效权重。
- 按每parent等权；P1 task均衡接口，失败/恢复池审核不足时不伪造50/50池。
- 评估K=8/10/25/32/50，左右关节rad、夹爪mm及四组共同训练尺度；parent等权、R/P同窗口配对、parent/session两级bootstrap（默认10k），session<10标不稳定。
- 正式链式脚本已生成但未执行；准备桶4 GPUh、两臂训练运行台账硬停44 GPUh，总48 GPUh。profile单卡桶.4 GPUh；共享卡必须memory<512MiB且util<5%。

## 7. 最终数据接入与生产尺寸 CPU 证据

READY watcher：10:21:09 BRT开始，10:31、10:41、10:51按600秒检查；10:51:09发现READY_data，等待30分钟，未触发4小时超时。发布时间10:44:06，状态为 `ready_P1_CPU_and_P3_candidate_pending_quality`：这是分阶段发布，不将P3 pending误判为整个包失效，也不将它误判为P3正式放行。

- 入口：严格读取 READY_data.entrypoints；当前 `manifest/captures_final.jsonl`、video_routes、numeric/window_cache_v2。两级READY及**2476个发布文件sha256均校验通过**，首次校验4.742秒。
- parent READY sha256：`1b6669bf82693924a6ddddfaf0fa3225cd12ed02db97e434f145003247e1db13`。
- data READY sha256：`18b5d27d047f670bc2d2af09dbf68ac7051fde4749c633028c9a3ec2ce8e9844`。
- P1：109,029窗口；244个有有效窗parent；可用覆盖6.37245h（物理候选去重6.40997h，245个候选parent）。
- P3候选：33,888窗口；144个有有效窗parent；可用覆盖2.60497h（候选2.61142h、150 parent）。**P3_ready=0**，`train_ready_p3=false`。正式dataset/stat/preflight均拒绝；没有以成功字段替代逐段质量审核。
- 未读冻结test图像，未为dev/test生成模型标注，未用E82-L模型结果当评估真值。

生产尺寸真实CPU前向：`artifacts/real_cpu_forward_v1.json`。仅允许train parent的小窗 `piper_030c4dc6ede63c47d4f90aef:8`；原始本地DINO、顶252×448、双腕224×224，独立5crop接口。O[1,187,256]；A[1,50,14]；W[1,3,91,768]；有效O=122、W监督/结构有效=177。**缓存max_abs=0；P1 A不存在；输出有限；cuda_initialized=false**。总CPU流程19.929秒（含哈希/读视频/加载/多次前向，不能当GPU时延）。该数值smoke没有优化器、没有正式checkpoint、没有最终统计拟合。

固定腕ROI当前未人工确认，严格missing，保留全腕图；每时刻59个有效视觉token，容量槽位仍91。未来目标缺失只参与loss mask，不借目标可用性改变O。结构性未知ROI同时从W query/key硬屏蔽；更改这些masked W值不能改变A。不能声称已验证锁孔/尖端的可见性，也不能把候选框当已批准ROI。geo槽缺标定屏蔽，未制造FK。

## 8. 最终合同与流水线验证

最终：`artifacts/contracts_release_final.txt`：**29 passed, 1 warning, 21.38s**；全部CPU，含2项真实DINO集成。唯一warning为测试张量转换提示，不是测试失败；完整输出保留。`compileall`和各shell `bash -n`通过。

|必须合同|结果/证据|
|---|---|
|O只读O；改未来不改O|精确一致；对未来W的O梯度为0|
|W硬删除无隐性条件|不调用W投影；改变未使用τW不改A；结构missing W同样不影响A|
|LA/LW分别到core/LoRA|fixture和真实原始DINO均非零梯度；A/W hidden不detach|
|E*梯度0|target全冻结、eval/no_grad；部署policy不实例化E*|
|缺模态/缺状态|全部视觉/状态mask时仅task槽仍可运行；未知任务与越界ID归UNK|
|缓存/非缓存|合成joint/action_only容差通过；真实生产尺寸max_abs=0|
|joint/action_only|两条均10 Euler×4固定噪声；按canonical window派生，不随batch切块变化|
|同seed/恢复|P3 R/P流相同、W不推进A；optimizer/RNG恢复后下一更新与参数精确一致|
|P1|A模块与state头保持S0；A token为0；视觉/LoRA零梯度，仅LW|
|流水线与统计|全部4模式含额外LA、有效batch分母、因果真实Δt、signed grip、parent等权/mm/配对簇bootstrap、独立真值止损、READY篡改拒绝|

`artifacts/fixture_smoke_final/status.json`：合成有效batch32的P1→P3_warm→P3_joint→resume全部完成。`fixture_profile100_v1/profile.json`：100个CPU fixture更新的profile流程已走通；**此文件仅验证程序，不是用户要求的真实GPU profile，也不拿其中外推值判断48 GPUh**。

正式链式脚本已交付：`scripts/launch_formal.sh` → `formal_worker.sh`，setsid nohup串行 P:P1→warm→joint→R:warm→joint，每阶段状态、checkpoint、共享paired ledger；本轮未执行。正式初始化/恢复校验config/encoder/data/stat/code hash；CPU路径不读CUDA RNG，GPU路径只记录实际单卡RNG。冻结视觉基础/目标参数不重复写checkpoint，引用原始encoder hash。

## 9. GPU profile 阻塞与预算结论

真实GPU profile尝试：`artifacts/real_gpu_profile_20261112_1054/status.json`，实际10:53:14 BRT返回 `blocked_no_idle_gpu`。本轮起始/10:40/10:53及13:37复查，8卡均有明显显存占用；13:37最少仍约21GiB。0号卡虽util=0，仍约89GiB占用，**不能当空闲卡使用**。未触碰任何他人进程，没有运行CUDA模型训练/推理/真实100更新profile。

因此：**真实P1/P3 GPUh估算未完成，48 GPUh能否装下未决；P99≤250ms、333ms miss验收未测。** 不用CPU时间或fixture推算冒充GPU实测。实现已提供100更新真实profile（P1 30、warm20、joint50；microbatch1，按32倍算力+原位解码保守外推有效batch32）；真实两类小窗、100更新、两个模式时延10调用/模式；10调用仍不能宣称E81≥1000调用时延验收。

预算公式（秒/有效batch32更新，包含真实I/O与训练开销）：

`总GPUh = 4准备 + 5000*t_P1/3600 + 2*(1000*t_warm+9000*t_joint)/3600`。

P1阶段=`1.38889*t_P1`，每臂warm=`0.27778*t_warm`，每臂joint=`2.5*t_joint` GPUh。若各阶段均同速，仅作条件场景：

|有效更新秒数（非实测）|P1 GPUh|单臂warm+joint GPUh|含4h准备两臂总GPUh|48内？|
|---:|---:|---:|---:|---|
|1|1.389|2.778|10.944|条件成立时是|
|5|6.944|13.889|38.722|条件成立时是|
|7|9.722|19.444|52.611|否|

等速时必须≤**6.336秒/有效更新**才在44h训练桶内；这不是本机实测承诺。正式launcher要求真实profile、完整P3质量许可与最终stat_contract，当前不能正式放行。

续跑（空闲卡出现后，仍只profile，不启动正式训练）：

```bash
cd /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e82
setsid nohup bash scripts/run_bounded_profile.sh artifacts/real_gpu_profile_NEW \
  > artifacts/real_gpu_profile_NEW.log 2>&1 < /dev/null &
```

程序自动选择memory<512MiB且util<5%的单卡；无卡返回75。当前profile桶收紧为**0.4 GPUh**，外层timeout1450秒且只终止本脚本子进程。没有预留/占用卡等待，不终止他人进程。

### 9.1 必须披露的资源审计偏差

早期CPU测试保存checkpoint时曾调用 `torch.cuda.get_rng_state_all()`，它可能触发CUDA上下文初始化，违反了CPU测试不应接触非空闲GPU的预期。没有CUDA模型计算，但**不能把早期CPU ledger=0解释为已证明完全没有CUDA上下文**。已修复为CPU不读CUDA RNG、CUDA仅读实际单卡；增加强制禁止CUDA RNG/_lazy_init的回归测试，并在真实CPU前向确认cuda_initialized=false。

为不掩盖潜在占卡，`artifacts/gpu_accounting.json`将所有相关早期CPU测试153.16秒＋smoke23.176秒＋30秒额外余量，极保守按8卡全程上下文占用计上界**0.458525 GPUh**，这不是实测计费用量。即使后续用满0.4h真实profile桶，总保守上界约**0.858525 GPUh<1**。早期真实上下文是否/何时分配无法事后精确还原；保留全部原始测试/账本，未删除痕迹。

## 10. 交付、未完成与需安排

- 已交付 `T/code_e82/`：源码、配置schema、README运行/测试/评估/恢复命令、合成/真实CPU证据、待用正式脚本。代码及自有测试artifact约802MiB；不复制大权重/录像，不写同事目录；没有正式run。
- 已完成真实数据入口接线，最终统计拟合与正式S0生产包尚未生成：P3_ready=0，不能把候选统计锁成正式尺度。P1/P3统计接口和stage gate保留。
- 未完成：空闲单卡上的真实100更新profile、真实GPUh估算、≥1000次全pipeline时延验收；ROI/像素可见性与P3逐段质量许可；独立冻结eval窗口/新确认集与真实性能评估。天机adapter和P0 R0本轮按用户要求不做，`interfaces.py`保留协议扩展位。
- 需用户/资源负责人安排：一张确实空闲H20的短时窗口；不需要批准正式训练，本轮仍禁止启动。P3/ROI人工复核由E82-L/人工流程继续，不用模型标签顶替。
- 后续正式训练前若profile超48h，需要两臂同时重注册配方或另安排预算，不能单臂缩短补齐。

状态：**代码/全部合同/真实CPU接入完成；真实GPU profile资源阻塞；未启动正式训练。**
