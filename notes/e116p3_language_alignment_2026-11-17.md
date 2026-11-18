# E116-P3：语言-图像对齐探针（完整策略组装方案 P3）

- 日期：2026-11-17（BRT，UTC-3）；状态：完成。
- 目标：纯推理评估 SigLIP2 图文联合空间能否识别 Piper 阶段和因果谓词；通过后才考虑阶段/子目标文字条件。本轮未训练。
- 代码：`code_e116p3/`；结果：`results/e116p3_20261117_01/`；模型：本地 `/mnt/workspace/dengjianfeng/models/siglip2`。
- 范围：语言-图像对齐；语言-动作对齐不在本期范围。
- 预注册：阶段 top-1 的 parent bootstrap 95% CI 下界须 >0.20（10 类随机基线 0.10 上方至少 0.10）；且谓词宏 AUROC 的 CI 下界 >0.60。P073 固定弃权。
- 数据规则：dev 阶段/谓词参考标签由模型生成，仅作开发参考；不得作为测试真值、阈值来源或 checkpoint 选型依据。永久排除 demo3 episode142 / `piper_4dc107335389f5711a0f65e0`。
- 资源约束：GPU ≤0.5 GPUh；输出/缓存 ≤2GB；不下载数据集或大权重。全部旧结果、标签和特征仅读取。

## 1. 先验材料与评估锁定

P1 已显示把阶段后验条件输入动作头未带来可测动作收益（阶段信息不进入动作头）。P2 显示预测 token 的任务进展探针能区分对/错动作，但不检验语言。P3 只检验文本-图像对齐，不扩展到动作条件。

本轮锁定：20 个 dev parent，共 1,867 个阶段评价帧（每 30 帧抽一帧）；逐相机独立提取联合空间图像特征，主结果为三相机 logit 均值，另报 cam_high。主阶段文本为起始+结束描述英文版，其他五种文本组/语言并列为敏感性。谓词用 question_en 与 primary_camera，中文/相机融合为敏感性。文本在检查 dev 前写入 `text_lock.json` 并固定。谓词评估仅因果 yes/no（约 2,533 条、20 parents），P073 弃权。父级 bootstrap 2,000 次：阶段 seed 90，谓词 seed 10201。CF H0 监督对照用冻结 H0_full.pkl/final_aux.pkl 作用于 E102-S b 特征，逐帧 emission argmax、不用 HMM。

## 2. 锁定文本（dev 评估前）

主阶段文本固定为 **起始+结束描述，英文**；相机融合固定为三相机 logit 均值。选择理由是该组完整描述阶段可见关系而非仅标签名；本文不根据 dev 调整。另并列报告阶段名/起始描述 × 中英全部敏感性。英文翻译一次锁定。谓词主评估固定用英文 question_en 与 primary_camera；中文为敏感性。P073 固定弃权。完整锁定见 `results/e116p3_20261117_01/text_lock.json`。

|阶段|中文名|English name|起始中文|Start description|结束中文|End description|
|---|---|---|---|---|---|---|
|S01|接近待取扎带|Approach the selected zip tie|空右夹爪向容器口或筐中竖立带身运动|An empty right gripper moves toward an upright strap in the bin or basket.|目标带身位于两指间，锁头在指尖上方且未随爪抬起|The selected strap is between the two fingers, with its locking head above the fingertips and not lifted with the gripper.|
|S02|夹取并提离料区|Lift the selected zip tie from the bin|目标带身已位于张开右爪两指之间|The selected strap lies between the open fingers of the right gripper.|所持目标随右爪离开料区，相邻带身仍留在源容器/筐内|The held target moves with the right gripper away from the bin while neighboring straps remain inside.|
|S03|锁头交接|Hand off the locking head|右爪持锁头附近带身向左爪开口移动|The right hand moves a held strap near the locking head toward the open left gripper.|左爪支撑锁头，右爪离开原夹持段，目标未下落|The left gripper supports the locking head after the right gripper leaves its original grip, and the target does not fall.|
|S04|沿带身获取自由端|Reach for the free tail|左手持锁头，空右爪向悬垂带身下端移动|The left hand holds the locking head while the empty right gripper moves toward the hanging lower strap.|自由端与右爪共同运动，尖端露出且未被指尖完全包住|The free end moves with the right gripper, with the tip exposed beyond the fingers.|
|S05|弯回并对准锁孔|Bend the tail back and align it with the locking head|右手持自由端离开下方直垂位置，向左手锁头回弯|The right hand moves the held free end from its hanging position back toward the locking head in the left hand.|尖端与入口在腕视中对齐，近邻时序和辅助视角支持深度接近|The tip is aligned with the opening in the wrist view, with nearby timing and the auxiliary view supporting depth alignment.|
|S06|送入并确认穿出|Feed the tip through and confirm it exits|尖端在入口附近开始向孔内推进|The tip begins to advance into the opening near the locking head.|出口侧出现可辨识的新增带段，左侧锁头仍被支撑|A new visible segment appears on the exit side while the locking head remains supported by the left hand.|
|S07|换抓穿出段并拉出成环|Regrasp the emerged segment and pull a loop|出口段已露出，右爪移到锁头出口侧|A segment has emerged from the exit and the right gripper moves to the exit side of the locking head.|拉动后出口带段仍穿过锁头且环体保留，右爪可卸载|After pulling, the exit segment remains through the locking head and a loop remains; the right gripper can unload.|
|S08|降低并释放|Lower and release the finished tie|所持环体向桌面或筐内降低|The held loop is lowered toward the tabletop or into the basket.|新放下的目标由接收面支撑，两爪张开且不再牵动目标|The receiving surface supports the newly placed target, both grippers open, and neither pulls the target.|
|S09|撤离并决定下一轮|Withdraw and decide whether to start another tie|目标已释放，双臂离开落点|The target has been released and both arms move away from the drop point.|两爪脱离成品，按可见新料选择下一轮或进入批次等待|Both grippers are clear of the finished tie, and a visible next tie is selected or the batch enters a wait.|
|S10|失败恢复与回退|Recover from a failed grasp or insertion|可见抓空、滑脱或已建立连接消失，或同根扎带的退出再送入|A missed grasp, slip, lost connection, or withdrawal and reinsertion of the same tie is visible.|恢复到可重抓/可对准关系，或放弃目标并释放到可见接收面|The grasp or alignment relation is restored for another attempt, or the target is abandoned and released onto a visible receiving surface.|

谓词问句按规格原文锁定，文件含 21 项中英文本及 primary_camera；P073 固定弃权。


## 3. 联合空间小测试

- 沿用 `code_e86/e86/vision.py` 的 E86 预处理：RGB、224、保留宽高比双线性缩放、中心边缘填充、无 crop、mean/std=0.5。
- 本地 `SiglipModel` 的 `get_image_features` / `get_text_features`；图文特征先 L2 归一化，再用模型 logit_scale=112.6689、logit_bias=-16.7717 计算 sigmoid 相似度。文本 tokenizer 固定 `padding=max_length,max_length=64`，不截断。
- 同一训练 parent 图像重复提取的图像嵌入最大绝对差为 0；文本 102 条、联合维度 768；相似度样本范围 4.04e-6 至 0.1651；有限值检查通过。MHA fastpath 禁用。
- GPU 准入：GPU 4 初始空闲 85,253 MiB，预估峰值 10,240 MiB，超过 1.5×准入要求（15,360 MiB）。模型前向累计 71.65 秒，即 0.0199 GPUh；实测峰值 1,778,686,464 bytes（1,696 MiB）。没有其他 GPU 任务或失败 GPU 段。

## 4. 评估结果（单次冻结 dev；模型生成标签仅作开发参考）

### 阶段零样本与监督对照
主结果 start+end 英文、三相机 logit 均值：top-1=0.0364 [0.0304,0.0412]；top-3=0.1971 [0.1770,0.2169]；macro-F1=0.0170 [0.0132,0.0207]。10 类随机 top-1=0.10；预注册通过门槛为 CI 下界 >0.20。
cam_high top-1=0.0552 [0.0453,0.0655]。CF H0 同帧逐帧 emission argmax（无 HMM）top-1=0.6213 [0.5968,0.6477]，macro-F1=0.5592；零样本−CF H0 top-1 差=-0.5849 [-0.6128,-0.5599]。

|文本组|语言|融合|Top-1|Top-3|Macro-F1|
|---|---|---|---:|---:|---:|
|name|en|high|0.075|0.333|0.039|
|name|en|mean|0.049|0.297|0.014|
|name|zh|high|0.129|0.305|0.027|
|name|zh|mean|0.114|0.262|0.029|
|start|en|high|0.089|0.284|0.052|
|start|en|mean|0.061|0.317|0.043|
|start|zh|high|0.094|0.209|0.060|
|start|zh|mean|0.037|0.123|0.030|
|start_plus_end|en|high|0.055|0.220|0.045|
|start_plus_end|en|mean|0.036|0.197|0.017|
|start_plus_end|zh|high|0.076|0.271|0.055|
|start_plus_end|zh|mean|0.036|0.208|0.016|

主结果混淆矩阵（行=参考、列=预测，S01–S10）：

```text
2 0 0 0 0 0 0 145 0 1
2 2 0 5 0 0 0 80 0 2
1 32 5 56 0 0 0 134 0 2
0 6 2 5 6 0 0 263 0 23
0 28 3 7 0 0 0 256 0 0
0 147 22 4 0 0 0 194 0 0
5 91 38 9 0 0 0 115 0 0
0 0 0 0 0 0 0 54 0 1
0 0 0 0 0 0 0 97 0 0
0 3 6 1 0 0 0 12 0 0
```

人工 4 片段以人工 marker 区间对照同一抽帧：帧加权 top-1=0.094，片段 macro-F1 均值=0.036，共 521 帧。

### 谓词零样本与对照

|方法|记录数|parent|可估谓词|Macro AUROC [95%CI]|
|---|---:|---:|---:|---:|
|SigLIP2_text_primary|2533|20|19|0.520 [0.486,0.584]|
|SigLIP2_text_cam_mean|2533|20|19|0.469 [0.443,0.529]|
|SigLIP2_text_cam_max|2533|20|19|0.459 [0.435,0.518]|
|SigLIP2_text_zh_primary|2533|20|19|0.459 [0.418,0.544]|
|CF_H0|2533|20|19|0.908 [0.876,0.928]|
|VLM_zero_shot|2533|20|19|0.629 [0.580,0.661]|
|VLM_multiframe_V3|2533|20|19|0.683 [0.618,0.716]|
|SigLIP2_linear_b|2533|20|19|0.795 [0.755,0.812]|

|谓词|英文 primary|三相机均值|三相机最大|中文 primary|CF H0|
|---|---:|---:|---:|---:|---:|
|P011|NA|NA|NA|NA|NA|
|P021|0.693|0.480|0.466|0.453|0.831|
|P022|0.285|0.265|0.205|0.421|0.913|
|P031|0.730|0.784|0.738|0.521|0.974|
|P032|0.503|0.501|0.488|0.454|0.977|
|P041|0.686|0.726|0.776|0.516|0.933|
|P042|0.657|0.707|0.696|0.326|0.928|
|P051|0.447|0.294|0.343|0.519|0.917|
|P052|0.649|0.293|0.289|0.553|0.953|
|P061|0.560|0.486|0.453|0.423|0.948|
|P062|0.701|0.588|0.538|0.621|0.957|
|P071|0.372|0.649|0.593|0.767|0.908|
|P072|0.660|0.350|0.340|0.635|0.860|
|P081|0.649|0.576|0.672|0.262|0.958|
|P082|0.595|0.502|0.583|0.504|0.982|
|P091|0.472|0.435|0.343|0.204|0.907|
|P092|0.583|0.582|0.567|0.654|0.723|
|P101|0.333|0.324|0.370|0.421|0.711|
|P102|0.305|0.377|0.260|0.467|0.875|
|P103|0.000|0.000|0.000|0.000|1.000|

配对差=本轮英文 primary 文本−基线（仅共同可用标注行；seed10201 parent bootstrap）：

|Comparator|共同记录|primary AUROC|对照 AUROC|配对差 [95%CI]|旧结果参考|
|---|---:|---:|---:|---:|---:|
|VLM_zero_shot|2533|0.520|0.629|-0.109 [-0.153,-0.021]|0.629|
|VLM_multiframe_V3|2533|0.520|0.683|-0.163 [-0.198,-0.079]|0.683|
|SigLIP2_linear_b|2533|0.520|0.795|-0.275 [-0.311,-0.194]|0.795|
|CF_H0|2533|0.520|0.908|-0.388 [-0.424,-0.314]|0.908|

P073 固定弃权并排除；宏 AUROC 对有 yes/no 的其余谓词取平均。

### 时间一致性

|阶段|parent-stage AUROC 可估数|平均 AUROC|
|---|---:|---:|
|S01|16|0.220|
|S02|16|0.332|
|S03|17|0.684|
|S04|18|0.420|
|S05|18|0.605|
|S06|17|0.307|
|S07|16|0.655|
|S08|14|0.634|
|S09|14|0.104|
|S10|6|0.370|

边界相对帧当前阶段文本平均 logit：-90: -2.982, -60: -2.995, -30: -3.034, 0: -3.050, 30: -2.855, 60: -2.895, 90: -2.929。三个人工片段曲线见 `results/e116p3_20261117_01/stage_similarity_curves.svg`。


## 5. 资源、判定、局限与未完成项

### 资源与复现

- GPU：GPU 4；累计 71.65 秒 / 0.0199 GPUh（预算 0.5 GPUh）。
- 新结果目录：9,106,289 bytes（约 8.69 MiB）；新代码目录约 121 KiB。为运行本地模型接口，在既有用户 venv 补装 Transformers/tokenizers 等轻量运行库，合计约 135 MiB；未下载模型权重或数据。新增落盘合计低于 150 MiB，低于 2GB。
- 主要命令：`setsid nohup .../code_e116p3/eval.py`（一次冻结 dev 图像推理）；`setsid nohup .../code_e116p3/metrics.py`（只读冻结分数、标签与 b 特征的 CPU 统计）。详细状态、GPU ledger、锁定文本和分数保存在本 run 目录。

### 预注册判定

**不通过。** 主阶段 top-1=0.0364，95% CI [0.0304, 0.0412]，其下界低于预注册 0.20；谓词英文 primary 宏 AUROC=0.5199，95% CI [0.4862, 0.5842]，其下界低于 0.60。相同阶段样本的 CF H0 top-1=0.6213 [0.5968, 0.6477]，零样本−CF H0 差为 −0.5849 [−0.6128, −0.5599]。人工 4 片段的抽帧 top-1=0.0940，支持阶段对齐弱的结论，但样本较少。

**该 SigLIP2 的文本-图像空间对本任务专业描述对齐很弱，语言不能直接用作阶段/子目标条件，需要微调或换用语言监督。** 本轮不执行以下方向，仅列作候选：

1. 用训练 parent 上的阶段/谓词图文对对 SigLIP2 做任务特定对比微调，之后用全新冻结评估集验证。
2. 换用带时间上下文的视觉语言编码器或面向操作场景的已训练模型，再重复相同预注册探针。
3. 以可见对象关系和阶段变化为目标构建人工核验的图文监督集，提升专业描述质量后再训练/比较。

### 局限与未完成项

- dev 阶段和谓词标签均由模型生成，不是人工真值；不能外推真实任务成功率。人工真值仅 4 个片段，按每 30 帧抽样，共 521 帧。
- 为控制缓存，保存的相似度概率为 float16；极低相似度可能量化/并列，需在后续全新评估中用 float32 分数复核。未根据该风险重跑或改选文本。
- P011 与 P073 无可估 AUROC（P073 按预设弃权）；其他少数谓词样本也较少。单一 SigLIP2 checkpoint、单一描述翻译。
- 阶段零样本与谓词零样本均未过门槛；没有把阶段/子目标文字接入策略。语言-动作对齐未评估，也不属于本期范围。
- 下一步需用户决定是否开展语言监督微调、替换编码器或先补人工核验图文集；本轮没有未完成的实验命令。
