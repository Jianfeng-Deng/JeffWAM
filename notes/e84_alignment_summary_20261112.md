# E84 对齐验证：前置核验 NO-GO

ID `e84p_20261112_153338`；BRT 2026-11-12T16:37:44.053580-03:00。**状态：完整E84未完成，因必需的合格像素真值缺失，在正式模型实验前停止。** 依据用户“缺什么就停下来列出并汇报”，已停止本会话的等待/自动启动进程；没有真实图像模型推理、正式探针、LoRA或WAM训练，GPU **0 GPUh**。不将合成测试、固定HSV或粗投影包装成正式对齐成绩。

## 三种对齐×三个层级

|对齐类型|冻结零样本|冻结特征探针|LoRA|
|---|---|---|---|
|语言↔图像（7对象/左右）|未执行：精确像素真值未批准|未执行|未执行|
|语言↔像素运动|未执行：仅有粗投影；FK自身base运动可另作受限实验|未执行|未执行|
|语言↔动作反事实|该项需训练预测器|未执行：等待完整实验/受限范围选择|未执行|

因此没有真实top1/top3、pointing/AUROC、方向角误差、跟随率、混淆矩阵或模型差值置信区间；不编造随机/打乱对照差距。均匀随机理论值仅为24类top1=4.17%、top3=12.5%，8向=12.5%，3幅度=33.3%，不能代替实验结果。

## 已完成的工作

1. 读取必需文档和revision3数据/时间合同；READY_split就绪后才读取允许训练域缓存。F划分195/49候选parent，SHA锁定；独立核验244候选parent全部来自E82 train，dev/test图像和数值未读。
2. 核验本地 `google/siglip2-base-patch16-224`，revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`，375,187,970参数；双塔strict加载无missing/unexpected/mismatch。视觉224、patch16、196×768；文本256000词表、64token，fast tokenizer离线可用；固定分辨率，**非NaFlex**。
3. 建立可恢复报告/progress、脚本、样本清单与预算保护。正式全量清单 {'probe-train': 11376, 'probe-val': 10854}，CPU回退清单 {'probe-train': 1541, 'probe-val': 387}，有效parent {'probe-train': 193, 'probe-val': 49}；整父排除诊断数据，另隔离一个已知左腕视频无法解码的capture。只预选，未抽取模型特征。
4. 通过合成双塔前向、逐patch MAP等价优化、指标tie处理、parent bootstrap、历史维度/仿射、线性/对比/条件位移学习路径、LoRA双塔梯度与合并、单线程视频解码测试。对比头738144参数；双塔末2层Q/V rank8共98304参数。**这些全是实现验证，不是任务成绩。**
5. 接收并逐入口核验F最终回执；发布后F增加诊断资格门槛时，旧SHA拦截生效，重新接收新SHA `95e48c2130c9a65a4d8fc121f9b7b61c04610ec0d3cfaa901310a1d682f9efa9` 并保留快照；没有基于旧回执运行模型。已实际复看H01–H06/P01–P06共12张原图/overlay，保存逐项观察，未新增精确标签。

## 阻塞的具体证据

F回执：`precise_pixel_ground_truth_approved=false`。30帧投影抽查，30个可见估点，误差中位 **37.44px**、P90 **71.72px**、最大 **116.64px**，估点不确定15–25px。旧demo4投影仅粗诊断；demo3/demo6没有合格投影，demo6沿用demo4标定NO-GO，右腕手眼未开放为夹爪中心GT。

HSV在三路各10帧中，含误检帧顶/左腕/右腕分别 **8/10、10/10、6/10**，顶视还有5/10帧漏掉可见橙色。颜色成员是固定阈值真值，但不是目标扎带实例、锁头或尖端真值；这也不是人口precision/recall。F尝试5832张、成功5824张，不能把数量当作语义真值已合格。

P实际复看也见多根橙带与预存环共同进入mask、腕部小结构误选、淡细带身不完整覆盖；部分投影点位于底边/不可见夹爪，两臂重叠时点接近。**缺少的不是一个下载文件，而是可验收的左右/夹爪/锁头/尖端像素依据。**

FK自身可用：435404合格运动行、547capture/243parent（诊断712行只审计）。末端按URDF几何中心，各臂base_link独立，+X前/+Y机器人左/+Z上，无独立共同世界安装变换。它可支持“自身基座方向/幅度”和语言条件位移的受限实验；不能补成精确像素或物体身份GT。

## patch级文本图的方法与局限

SigLIP2 MAP头用可学习单probe对全部196视觉token作注意力，再residual MLP得到全局向量；注意力本身不依赖文本。拟用同一MAP头逐patch单独运行，再与归一化文本向量余弦；singleton注意力=1，等价于V/out投影+residual MLP，并以原始patch余弦作对照。该优化与字面逐patch实现合成误差0，合并LoRA的完整双塔前向最大差4.77e-7。

逐patch头偏离全局池化训练分布，token已含全图上下文，14×14容易丢细尖端/锁孔，分数不是分割概率。即使未来零样本失败，也必须区分“没有原生时间/patch文本投影”和“视觉信息不可学”。

本地缺preprocessor_config.json，通过[官方配置文档](https://huggingface.co/google/siglip2-base-patch16-224/blob/main/preprocessor_config.json)核实bilinear、224、mean/std .5，未下载模型/数据包；fast tokenizer不依赖缺失的tokenizer.model。本轮代码显式全视野保比例edge pad并存仿射，区别于官方直接方形resize，后续须披露分布差异。

## E83与FK文本的信息量

固定E83主文件快照（A仍可能续写）：107条、106parent；A/B交集 `{'pc_3dbd3a4e134442feb0df': ['E83B_022', 'E83_A_b7bad467f23b9d013e75658d']}`。训练应按parent去重/等权，不能把跨tier记录当独立示范。

|字段|记录|唯一完整文本|parent|文本熵bit|超过64token|
|---|---:|---:|---:|---:|---:|
|A/task_instruction_en|13|13|13|3.700|13|
|B/task_instruction_en|94|8|94|2.578|94|
|B/instruction_variant_1_en|94|1|94|-0.000|0|
|all/task_instruction_en|107|21|106|3.248|107|
|all/scene_description_en|107|107|106|6.741|107|
|A/process_narration_en|93|93|13|6.539|14|

|FK字段|行数|唯一文本|parent|文本熵bit|
|---|---:|---:|---:|---:|
|all/text_en|435404|18393|243|8.119|
|all/text_en_without_numeric|435404|471|243|5.616|

B完整任务指令仅8种、最短改写1种，主要表达单任务流程；13条B记录未实证完整双臂流程，规范补全不是动作事件。完整英文指令全部超过64token，直接截断会丢失大部分后续步骤。FK精确cm数字会增加唯一数，去数字版和8向×3幅度覆盖更有意义。完整text↔parent对应见不可变语言快照和fk_information.json。

建议任务层继续短任务指令，局部方向/幅度语言优先来自FK；E83分句/按过程段作训练辅助，先处理64token分块与parent去重。生成语言不作测试真值/阈值/选checkpoint；未来过程字幕不作部署当前输入。

## 阶段一建议与需要决定的事

**当前建议：暂不投入全量阶段一，先补合格像素真值。** 现有证据无法可靠在“冻结即可 / 需要LoRA / 需要更强VLM”中三选一；尤其不能把坏投影/不可靠实例mask的低分当作SigLIP2失败。没有更换/下载VLM的依据；文本塔继续按用户原方案先冻结。

待用户选择：补独立左右夹爪/对象实例与锁头/尖端像素标注并重验demo4标定后再跑完整E84；或明确接受先完成FK运动、语言条件动作反事实和严格标注的HSV/粗投影受限诊断。后者仍不能验收完整阶段一。

若继续，预注册的流程/参数/对照均已准备：固定parent分集；实验2用已观测O→W窗口而非当前去预测未来，机器人系与画面系分报；实验3只起点/因果历史/状态/未来FK句，换8方向、幅度固定，比较无语言/打乱语言并排除<1mm方向。固定epoch、train-only归一化、不用val挑checkpoint，parent bootstrap2000次。LoRA只可由合格FK方向失败且探针可学触发，不由未批准像素proxy触发。

## 资源、断点与局限

已记录13次GPU轮询，首末相隔60.0分钟，均无显存<512MiB且利用率<5%的卡；尚未到2h CPU回退时点即因真值门槛停止。GPU 0 GPUh；正式特征0帧、探针0次、LoRA0次。共享R派生 0.748GB，P代码 0.19MB，明显低于30GB；未覆盖旧结果、未改他人源文件、未操作他人进程。只把自生成审阅拼图取回用于实际查看，未从外网下载权重/原始数据包。

样本量只是已锁定计划，不是已训练/评估量；标定可信范围不足、单任务相关性强、同parent时间相关、demo6各层val只有2parent。后续即使完成也只是E82 train内探针留出，不是dev/test或新独立实机确认集。

主报告 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e84_P_e84p_20261112_153338.md`；进度 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84p_20261112_153338/progress.json`；代码 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84p_20261112_153338`；F回执 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84p_20261112_153338/READY_fk_gt_accepted_95e48c2130c9a65a.json`；实际目视记录 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84p_20261112_153338/visual_gt_second_look.json`；E83快照 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84p_20261112_153338/e83_language_information_ea93c3d64559dbb3.json`；完整FK文本映射 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84p_20261112_153338/fk_information.json`。自有等待进程停止记录 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84p_20261112_153338/stopped_own_waiters.json`。
