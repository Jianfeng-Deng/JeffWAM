# E56 动作选择 / 候选评分文献调研

日期：2026-11-07

## 范围与结论

依据简报：天机双臂将瓶子放入箱，无腕部相机，93 条示教；run E 是微调 DINOv2-B 加 flow 动作头、使用 1 秒关节历史并可采样；E33-B 是动作条件 latent predictor（h16/h32）。按简报，MuJoCo 只用于最终验证，最终还要真机验证。本次只调研，不写代码、不跑实验、不用 GPU。

简报列出的 W/runs/e53_head_diag/REPORT.md、W/runs/e55_v2/REPORT.md、W/runs/e55_v2b/REPORT.md 在服务器项目树中均未找到，因此以下现有实验结论来自简报转述，原始报告、指标定义和统计不确定性均未核实。尤其“oracle 0.53/0.74/0.96、scorer 0.49/0.52/0.64”只按简报原样记录，不把它们解释为成功率或 AUC。

判断：眼下不应再训练一个只看示教阶段标签的分类头，也不该先扩大世界模型。先用 E48 候选与 E49 标签查清同一状态下的候选是否真的有不同结果、现有预测是否能分辨；再决定是否值得做小型 fail-gated 选择器。打分信号优先采用短时段的任务几何进度、安全约束和 E33-B 的相对预测变化；动作似然只约束候选是否像策略会出的动作，不作为成功概率。

## 1. 仅示教时有哪些评分信号

| 方法 | 数据与评分 | 仅示教可行性、成本和本任务风险 |
|---|---|---|
| 目标图像/latent 距离 | DINO-WM 在预训练 DINOv2 patch 特征上学习离线动作轨迹的未来特征，用目标图像特征作规划目标；V-JEPA 2-AC 在预训练视频模型上加动作条件预测器。[1][2] | 可用示教动作训练动力学，再以瓶子进箱的目标观测作伪目标；不是“零数据”，也不等于通用模型可直接替代本机动力学。目标距离容易偏爱外观相似但未接触、瓶子倾倒或姿态错误的状态。适合检查 E33-B，不足以单独当终局 verifier。模型训练和在线规划通常需 GPU；V-JEPA2 依赖大规模外部预训练及额外机器人视频。 |
| 学习进度/价值 | TCN 从视频时间邻近和多视角对应学时序嵌入；VIP 用人类视频预训练 goal-conditioned value，LIV 将语言/图像目标与无动作视频结合，Rank2Reward 以成功视频帧排序学增量进度，GVL 用 VLM 在上下文中估计进度。[4–8] | Rank2Reward/TCN 可用成功示教的时间次序，但依赖“沿演示单调前进”；简报已指出最近邻分数不单调，B1 退化为阶段检测器。VIP/LIV/GVL 借用了外部视频/模型或语义先验，输入小幅几何误差时可能过粗；不直接拿外部权重部署符合偏好。小型时间基线 CPU 可评估，学习排序模型通常需 GPU 训练。 |
| 示教能量/动作密度 | IBC 学习状态—动作能量，让策略选择示教支持的动作；diffusion/flow 策略给候选分布或采样支持度。[9][10] | 只用示教可训练，适合作为“是否偏离策略分布”的正则/候选筛选；高密度只说明像示教，不表示在当前状态有进展。run E 的 flow 概率也不能直接解释为成功概率，需单独校准。训练及多候选推理通常需 GPU。 |
| IRL、对比能量 | IRL 从专家行为推奖励；正反序、错配目标、打乱时间可构造对比负例。[11] | 只有成功示教时奖励不可唯一识别，且许多奖励都可解释同一轨迹。合成负例不是实际失败：时间反转/错配容易学到阶段或相机线索。可作为排序预训练，不足以给动作 outcome 定标。 |
| 离线 RL | PlaNet/Dreamer 类用潜在动力学规划并预测奖励；IQL 用数据回报训练价值并改进策略。[3][12] | 核心都需要奖励/回报监督或环境交互。93 条全部成功的示教若没有结果差异，回报几乎常数，不能凭算法名称“挖出失败价值”；加上人工/自动目标奖励后才可测，但奖励误设会被利用。CPU 可做小模型诊断，训练与规划通常需 GPU。 |

所有距离/进度信号都应与简单几何参照比较。对固定相机可先用 SAM3 得到瓶/箱区域，再计算瓶中心到箱口、瓶轴倾角、箱内投影/遮挡等量；手部采用正运动学与相机几何可用的部分。保留全局场景特征，显式记录 mask 置信度/缺失，不重复简报中表现更差的 hand ROI + bottle-mask 池化。相机标定和单目深度精度未核实，不能把 2D 距离当 3D 接触真值。

## 2. 采样后重排 / 验证器

V-GPS 以离线 RL 学得的价值函数重排通用策略提案，在多个预训练策略及真实机器人任务报告改善；其支撑依赖规模更大的跨任务数据和价值监督，与 93 条单任务示教差别很大。[13] RoboMonkey 对一个 VLA 采多个动作并加扰动，再用 VLM verifier 选优，报告 OOD 绝对提升 25 个百分点、ID 平均提升 8 个百分点；代价是合成偏好数据和大型 VLM，收益不应直接外推到 run E。[14] GPC 将生成策略与动作条件视频动力学模型放入在线规划，在 push 仿真和真实机器人上胜过 BC；模型还用了随机探索数据，计算与数据条件高于当前简报。[15]

候选选择的评估至少分四层：同一初始状态分支下候选结果是否有差异（selectable headroom）；评分对真实分支 outcome 的 pairwise accuracy/AUROC 与排序相关；best-of-N top-1 成功率相对单次策略、随机选和 oracle 的配对差；校准、拒绝率、延迟及安全违规。只报“选出的轨迹更好”会有赢家偏差。简报中 jitter 0.05/0.10/0.20 的 oracle/scorer 数字显示小扰动可能不可辨；指标定义未核实，需先重算。

2026 年新预印本对 4 个 WAM、2 个基准和超过 10 万次分支 rollout 做可选择空间审计，指出单次执行的 best-of-N 会混入后续随机性并高估收益；结构化扰动产生了更多可选空间，fail-gated selector 在其测试中提高闭环成功。[16] 这支持先审计再训练，但其仿真和模型结果不是天机双臂证据。另一篇 WISE 预印本用动作有效性、动作—预测未来一致性和任务进度对 WAM 的四个候选重排，在 RoboLab/Cosmos 3 Edge 上报 6%→24%，在 LIBERO/DreamZero 上 75.1%→78.3%；属于特定大模型、基准和推理开销的结果。[17]

## 3. 反事实负例从哪里来

- 合成扰动：围绕 run E 动作块作小幅、有限的关节/末端位姿偏移，先作为“候选生成”而非失败标签。用 E33-B 预测筛除安全/明显无效动作，再用已验证结果训练排序。扰动幅度必须覆盖模型可分辨范围；简报已显示最小 jitter 可能低于分辨率。
- 时间反转/随机打乱/错误目标配对：成本低，可学阶段、顺序或目标一致性；简报的 shuffled 仍被 demo 击败 0.71，且 demo 距离非单调，不能当物理失败负例。
- DAgger：在当前策略访问的状态让专家补动作标签，可纠正分布偏移，但需要专家参与每轮；DART 证明示教加噪声可提高模仿鲁棒性，并不提供动作成功/失败真值。[18][19]
- 自主 rollout 与恢复轨迹：能暴露真实错误分布；必须记录同一任务成功/失败和恢复动作，且有安全监督。最有辨别力的是同一初始状态、多个候选的分支执行及统一终局检查。
- 少量真机自评：用固定相机 + SAM3/几何规则给“瓶在箱内且直立”的终局标签，人工只抽查检测器错例；这无需持续逐帧标注。标签可靠性应先由盲审小样本核验。若 E49 已有候选级终局结果，则优先复用；否则 E49 的具体语义未核实，不能假定它是 outcome 数据。

## 4. latent 分辨率不足时怎么处理

文献中可行的共同策略是缩短预测跨度并频繁重规划、把模型用作粗筛而把精细接触留给低层策略、增加几何/关键点或动作—状态一致性判据。视频生成模型给更细的空间输出但算力与误差传播成本高；对象中心 slot/关键点需要训练/可靠检测，不能把它等同于简单 ROI pooling。结构化潜动作能提高候选差异，但现有 E33-B 的 h16/h32 若在小动作下预测排序接近，单纯扩大 latent 或长 rollout 不保证改善。

建议先测 h16/h32 在同一帧、候选动作对上的预测距离随实际几何变化是否有稳定斜率。若无，采用固定短动作块、在接触/进入箱口前后重规划；把 E33-B score 当粗筛，结合瓶—箱相对几何的短时增量与 FK 手部距离。评分可用“目标几何改善 + latent 预测改善 − 姿态误差 − 安全惩罚”，但以硬安全门控优先；若预测差小于验证出的分辨阈值，就回退 run E 原动作，不强行排名。

## 5. 2024–2026 具身世界模型与真机证据

DINO-WM（ICML 2025）展示预训练视觉特征上的离线动作条件规划；V-JEPA 2-AC 用少于 62 小时 DROID 无标注机器人视频后，报告在两处实验室做零样本 Franka 目标图像抓放。[1][2] 它说明“预训练视频动力学 + 少量机器人动作”可上真机，但不证明从 93 条天机示教直接得到收益。Cosmos Policy（ICLR 2026）把视频生成模型微调为 visuomotor policy/规划器，报告 LIBERO 98.5%、RoboCasa 67.1% 及双臂真机结果，并与同机器人示教训练的强基线比较；规模、数据和训练成本与当前设置不匹配。[20] DreamZero（2026 arXiv）把未来视频和动作联合建模，使用 14B 视频扩散模型，在 AgiBot/Franka 真机报告对新任务和环境的泛化优势及 7Hz 闭环；不是轻量候选打分器。[21]

在“未来视频是否真的由该动作导致”上，2026 年动态一致性预印本发现 action-state consistency 可区分若干基准上的成功/失败 rollout，并用候选未来共识做无奖励选择；背景静止也可能造成虚假的高一致性。[22] WAM 中模型内价值头、视频未来、外部 verifier 是不同部件，模型能生成逼真视频不等于动作可执行。整体证据表明大型预训练模型已有真机可行性，但没有同机同数据设置可证明它优于 run E；支持的是先验值得借鉴、外部权重不值得直接替换。

## 推荐与可证伪的最小实验（仅方案，不在本次执行）

**1. 低成本优先：E48/E49 可选择性与可排序性审计。** 输入：E48 候选包；核实 E49 的生成过程、标签粒度和是否为同一初始状态每候选的真实终局 outcome；E33-B h16/h32 预测、run E 原始动作、SAM3 几何。对已有候选分支离线计算几何增量、latent 增量、策略支持度、最近邻分数，并与标签配对。指标：候选 outcome 不同的比例（headroom）、pairwise AUROC/准确率、按 episode bootstrap 的 top-1 lift、oracle regret、tie rate。成功：E49 确为候选级 outcome，某评分相对随机/原策略的配对 top-1 lift 95% CI 下界大于 0，且 AUROC 点估计至少 0.65；失败：候选几乎同结果、AUROC≤0.55 或几何/latent 不胜简单基线，就停止训练 verifier。E49 若只是示教阶段标签，本实验只能判可排序性未识别，不能报告成功。

**2. 低成本、可解释：短时域 fail-gated 几何 + latent 重排。** 仅在方案1发现 headroom 后，固定相同初始状态和预算，对 run E 基线与少量候选比较；评分用安全硬门控、瓶—箱局部几何增量、E33-B 未来差分与不确定性，候选无显著分差则回退原策略。先离线；按既定流程只在最终验证阶段使用 MuJoCo，最后真机验证。指标是配对任务成功率、top-1 regret、碰撞/掉瓶/超限率、每次决策延迟。成功须超过原策略且安全指标不恶化；若 only-latent 低于几何基线或提升置信区间含零，删除 latent 评分。实际样本量按可用 E48/E49 分支数决定，不预设简报未提供的候选数。

**3. 数据成本中等：自动终局标签的小规模分支数据，再训练相对排序头。** 在固定相机下自动检查“瓶完整进入箱内且直立”，人工只抽查失败/置信度低的终局；同一初始状态对安全的小幅结构化动作分支记录终局，不做连续帧人工标注。数据划分按 episode，避免同轨迹泄漏。成功判据预注册为留出集 AUROC≥0.65 且比“进度阶段/动作密度”基线高至少 0.05，之后仅小规模真机比较闭环成功率；若标签错率高、分支 outcome 无差异或留出指标未过门槛，停止训练 verifier。该方案借鉴测试时验证，但不是重做 B1：目标是同状态候选的相对终局结果，而非示教事件类别。

已试/已否定的方向：B1 阶段分类头失败；E33_sub 子目标生成不重推；demo 最近邻未来评分非单调；手 ROI + bottle-mask pooling 劣于全局池化且缺失多。只有当第一项审计证实有真实可选空间且找到独立于阶段标签的任务结果信号，才有理由重开评分器训练。

## 参考文献

1. Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto. “DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning.” ICML 2025, PMLR 267. [论文](https://proceedings.mlr.press/v267/zhou25t.html)
2. Mido Assran et al. “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.” arXiv:2506.09985, 2025. [论文](https://arxiv.org/abs/2506.09985)
3. Danijar Hafner et al. “Learning Latent Dynamics for Planning from Pixels.” ICML 2019, PMLR 97. [论文](https://proceedings.mlr.press/v97/hafner19a.html)
4. Pierre Sermanet et al. “Time-Contrastive Networks: Self-Supervised Learning from Video.” ICRA 2018. [论文](https://arxiv.org/abs/1704.06888)
5. Yecheng Jason Ma et al. “VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training.” ICLR 2023. [论文](https://arxiv.org/abs/2210.00030)
6. Yecheng Jason Ma et al. “LIV: Language-Image Representations and Rewards for Robotic Control.” ICML 2023, PMLR 202. [论文](https://proceedings.mlr.press/v202/ma23b.html)
7. Yecheng Jason Ma et al. “Vision Language Models are In-Context Value Learners.” ICLR 2025. [论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/54854cf15a24fff9f5134a8641136fe4-Abstract-Conference.html)
8. Daniel Yang, Davin Tjia, Jacob Berg, Dima Damen, Pulkit Agrawal, Abhishek Gupta. “Rank2Reward: Learning Shaped Reward Functions from Passive Video.” ICRA 2024. [论文](https://arxiv.org/abs/2404.14735)
9. Pete Florence et al. “Implicit Behavioral Cloning.” CoRL 2021, PMLR 164 (2022). [论文](https://proceedings.mlr.press/v164/florence22a.html)
10. Cheng Chi et al. “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.” RSS 2023 / arXiv:2303.04137. [论文](https://arxiv.org/abs/2303.04137)
11. Andrew Y. Ng, Stuart Russell. “Algorithms for Inverse Reinforcement Learning.” ICML 2000, pp. 663–670. [论文 PDF](https://people.eecs.berkeley.edu/~russell/papers/ml00-irl.pdf)
12. Ilya Kostrikov, Ashvin Nair, Sergey Levine. “Offline Reinforcement Learning with Implicit Q-Learning.” ICLR 2022 / arXiv:2110.06169. [论文](https://arxiv.org/abs/2110.06169)
13. Mitsuhiko Nakamoto et al. “Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance.” CoRL 2024. [论文](https://openreview.net/forum?id=4c8e74f714242d258c2592502393c160e7bbe18d)
14. Jacky Kwok et al. “RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models.” CoRL 2025, PMLR 305:3200–3217. [论文](https://proceedings.mlr.press/v305/kwok25a.html)
15. Han Qi, Haocheng Yin, Yilun Du, Heng Yang. “Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling.” arXiv:2502.00622, 2025（最新版本标题；会议出处未核实）. [论文](https://arxiv.org/abs/2502.00622)
16. Haojia Gao et al. “How Much Can Best-of-N Help a World Action Model? Measuring Selectable Headroom and Realizing It in Closed Loop.” arXiv:2610.37196, 2026. [论文](https://arxiv.org/abs/2610.37196)
17. Cheikh Ahmed, Mahdi Mostajabdaveh, Milad Yazdani, Zirui Zhou. “Seeing Through the Illusion: Test-Time Scaling for World Action Models.” arXiv:2611.03585, 2026. [论文](https://arxiv.org/abs/2611.03585)
18. Stéphane Ross, Geoffrey Gordon, J. Andrew Bagnell. “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.” AISTATS 2011, PMLR 15. [论文](https://proceedings.mlr.press/v15/ross11a.html)
19. Michael Laskey et al. “DART: Noise Injection for Robust Imitation Learning.” CoRL 2017, PMLR 78. [论文](https://proceedings.mlr.press/v78/laskey17a.html)
20. Moo Jin Kim et al. “Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning.” ICLR 2026 / arXiv:2601.16163. [论文](https://arxiv.org/abs/2601.16163)
21. Seonghyeon Ye et al. “World Action Models are Zero-shot Policies.” arXiv:2602.15922, 2026. [论文](https://arxiv.org/abs/2602.15922)
22. Bo-Kai Ruan, Teng-Fang Hsiao, Ling Lo, Hong-Han Shuai. “Is the Future Compatible? Diagnosing Dynamic Consistency in World Action Models.” arXiv:2605.07514, 2026. [论文](https://arxiv.org/abs/2605.07514)

注：对论文在基准/平台上的收益只复述原论文摘要或出版页面的口径；不同任务、机器人、数据、成功判据不能横向当作本任务预期。
