# E50 候选动作生成器（P2）方案调研

检索日期：2026-11-05。范围：只读调研并撰写本文；没有改训练/推理代码，没有启动训练、仿真、真机或 GPU 作业。  
背景依据：PROJECT_DIRECTION.md、E44 文献调查、E47 P0–P1 设计合同、tools/e33_model.py（并只读追到其 FF.FlowPolicy 定义以确认采样实现）。年份以首次公开年份为主，另注正式发表年份；链接指向论文/作者页/项目或作者代码库。可行性、算力与推荐均为本项目条件下的工程判断。

## 结论摘要

- **最便宜的首个基线是现有 flow 权重的多噪声采样**：同一 context 用多个独立 noise 产生候选，不均值化；每个候选沿用 E47 的命令反归一化、从最新实测 q 锚定，并为自己重算 FK。它不需要新的训练，能直接回答“换噪声是否能提供可用覆盖”，但 E36 已显示当前分布的 16 样本 hindsight 最优误差上限只比 hold 好约 18–19%、均值约 9%、样本间标准差约 0.047 rad。该上限是对单条记录动作的 imitation oracle，不是物理任务成功上限。
- **如果换噪声没有足够可选空间，优先试全 chunk 级的 mode-latent / mixture / K-query 提议头**，再考虑从头改扩散头。整段共享一个模式变量比逐步独立抽模式更不容易出现动作跳变。训练只有 93 条示教时，模式覆盖和“多样但都可行”比增加模型大小更关键。
- **best-of-N、CEM/MPPI/iCEM 是搜索/选择器，不会自动增加生成分布的支撑集**。先单独报告生成器的候选覆盖与质量，再测评分器能否预报真实后果。若在现有评测样本上调参，必须把结果标成开发诊断，不能再称未见测试。
- E47 已定义 P2/P3 张量接口，但目前是设计合同：共享当前 context，候选为 B×K×32×54，forecast 对每候选给 h={4,8,16,32} 的 latent/mask 和逐步 geometry。P1 仍是 K=1，不含经过验收的候选评分器。按 AGENTS.md/PROJECT_DIRECTION.md，**MuJoCo 是主线第一个动作条件后果模型**；E47 latent forecast 可作后果特征/研究接口，不能据此宣称替代了 MuJoCo，也不能恢复已退役的旧 latent planner。

## 0. 当前接口与比较尺度

E33 的 E33.flow 是 FF.FlowPolicy(256, True)，动作维 54；输入噪声形状为 B×32×54。FlowPolicy 把 noisy action chunk 投影到 256 维，叠加 32 个 action position embedding 与 flow-time embedding，使用 4 层、8 头 TransformerDecoder cross-attend memory，再经 LayerNorm/Linear 预测 54 维速度。sample 从标准高斯噪声起步，用 10 步显式 Euler 积分。E33 infer 从 torch.randn 抽一次噪声，采出的归一化残差再按 a_mean/a_std 还原，并加到最新 q。E33 的 B/C 还可将 hold-future memory 送给 flow；E47 P1 合同则规定仅用当前 z、最新实测 q/dq/FK，固定噪声 K=1。两种 memory 不能混为同一个已验证模型。

32×54 每个候选是 1,728 维动作序列。若候选数 K、每个 flow 有 10 次速度网络调用、每次有 M 次 world-model rollout，朴素计算量随 K×10×M 增长；批处理能提高吞吐，但不消除所有算力/显存成本，必须实测 p50/p95。E47 可以共享 context 编码，forecast_candidates 对候选分块；相同 noise、候选次序和分块应有确定性，不能让分块改变结果。

## 1. 候选动作生成方案

### 1.1 现有 flow 头：多 noise / 多温度 / 候选分层

- **核心机制与出处**：Flow Matching Policy 在学习的向量场下，将噪声沿 ODE 积分到动作分布。π₀（2024）和 FlowPolicy（2024/AAAI 2025）展示 flow 动作策略；本地最直接的证据是 E33 代码本身。对冻结的 E33 checkpoint 改变初始噪声，可生成一组 chunk；温度缩放或将噪声分成小幅局部扰动，可诊断噪声敏感方向。
- **本地可行性**：最高，无需重训，先测 K={1,2,4,8,16}。使用 nested common seeds，使 K=2 的样本是 K=4/8/16 的子集。可报告按 arm A/B、左右手分组的幅度和终点 FK 分散度；不把较大的全维方差直接算成好候选。
- **算力**：单候选约 10 次 action velocity 前向；flow 样本 K 倍，随后每候选加 E47 F/decoder 和 verifier 成本。独立候选可批处理。
- **接口**：天然匹配 propose(ctx, noise[B,K,32,54], steps=10)；对每个候选先还原到绝对 q，再做每偏移动作标准化、FK(命令自身)和合法性过滤；forecast_candidates 接收原始动作 chunk 与各自 FK。
- **风险**：现有方差小，可能重复采样或只增加不相关的关节噪声；噪声坐标各向同性不代表任务模式。采样优于记录动作的 imitation oracle 可能只因单一 demo 没覆盖等价成功动作，不能当成坏策略或好策略的终判。

### 1.2 Diffusion Policy

- **核心机制与出处**：条件去噪器逐步修正整段动作；反复从噪声采样自然支持多模态序列和 receding-horizon 控制。Diffusion Policy (2023, RSS 2023；后刊 IJRR 2025) 是代表作，官方代码开放。
- **本地可行性**：54 维、32 步可表达，但要将 E33 flow policy 换成新训练头；93 条数据可以作目标域小规模拟合/对照，不能直接外推论文多任务或预训练效果。比 flow 增加采样步骤，训练与调参成本中高；若动作由 E47 共享编码器提供条件，结构接口相容。
- **算力**：每候选需要若干次去噪网络前向，K 候选并行仍近似按 K×反向步数增加；典型推理比 E33 的 10 步 flow 更贵，具体步数取决于 DDPM/DDIM/Scheduler 配置。
- **接口**：输出[B,K,32,54] 后接相同绝对 q 锚点、FK、forecast、过滤。候选不在像素域生成，世界模型可完全复用。
- **风险**：生成质量需要新训练；采样步数可能违背 15Hz 闭环时限；多模态能力是表达能力，不保证 93 条数据覆盖真实抓取/手指/接触替代动作；世界模型评分可能利用模型漏洞。

### 1.3 ACT / CVAE chunk 生成

- **核心机制与出处**：ACT (2023, CoRL 2023) 将 chunk 视为 CVAE decoder 输出，隐变量 z 编码动作风格。ALOHA 官方说明中，测试时原配方移除 CVAE encoder、把 z 置为先验均值 0；要做 best-of-K 必须显式从先验采样多个 z，不能把默认单样本推理误写成多样本。ALOHA 报告每任务 50 条示教的双臂基准，但任务、动作维度和相机条件均不同。
- **本地可行性**：较高，结构简单，适合作为小模型对照；相较扩散/flow 推理便宜。chunk 可为 32×54、沿用最新 q；训练/解码可复用 E47 当前 context。采样 prior 需要标定 temperature 与 mode consistency。
- **算力**：候选生成通常每个 z 一次 decoder 前向；K 倍解码开销，低于迭代扩散/flow。CVAE encoder 仅训练期用示教 chunk。
- **接口**：E47 context → z_k → action decoder → K 个绝对/残差 chunk → per-candidate FK → 同一 forecast。
- **风险**：隐变量可能被 decoder 忽略或仅造成细微抖动；ACT 的确定 prior-mean 默认推理不提供多样性；其 50 条/任务结果不是天机 54D 灵巧手的保证；对唯一成功演示拟合只能学“记录动作风格”，不会自行发现新接触模式。

### 1.4 离散动作模式：BeT / VQ-BeT / FAST

- **核心机制与出处**：BeT (2022, NeurIPS 2022) 先聚类连续动作，再预测离散 mode 并回归 residual；VQ-BeT (2024, ICML 2024) 用层级/Residual Vector Quantization 表示 latent action；FAST (2025) 对 action chunk 作 DCT 频域压缩，再离散 tokenization，针对高频灵巧动作，FAST+ tokenizer 用了约 1M 条真实轨迹训练。
- **本地可行性**：BeT/VQ-BeT 可将模式选择显式化，值得与 ACT/mixture 作为不同归纳偏置对照；FAST 需要 tokenizer 适配，FAST+ 的数据规模远超本地，不宜把其预训练优势记为可从 93 条复制。分量很多的 1,728D chunk 不能简单每个关节每帧独立分箱；建议整 chunk 或低秩/分段动作 token。
- **算力**：离散模式一次 categorical sampling + residual decoder，推理便宜；训练另需聚类/codebook 或 VQ-VAE 预训练，codebook/search 成本低于迭代生成，但数据准备和 codebook 诊断为中等。
- **接口**：先抽 K 个 code / token sequence，再解码 chunk；保持 mode ID 供 scorer 诊断，之后通过同一 E47 forecast。
- **风险**：93 条轨迹下 codebook 死码、模式只编码时间/数据频率、量化抖动与精度损失；per-step token 独立采样会造成动作跳变。FAST+ 公开资产不是小样本训练证据。

### 1.5 Mixture / multi-modal chunk head

- **核心机制与出处**：Mixture Density Network 直接预测 p(a_chunk|ctx)=Σ_m π_m(ctx)N(μ_m,Σ_m)，抽 categorical mode m 再采连续 chunk。BAKU (2024, NeurIPS 2024) 比较过多种 action head，包括 GMM、BeT、VQ-BeT 与 diffusion，且报告 30 个任务每任务平均 17 个示教；这是不同多任务架构的示例，不是天机同条件比较。
- **本地可行性**：中高，增量小于重做整套 score/diffusion。推荐共享一个 mode 对整段 32×54 生效，输出 per-step mean + 对角/低秩 covariance；先 M=2–4，避免 1,728 维完整协方差。较适合检验数据中是否存在可分离的有限风格。
- **算力**：一轮 head 前向即可得到 M 分布；从 K 个模式抽样开销很小；多分量前向增量近似随 M 线性，但远低于 rollout。
- **接口**：每个 proposal 有 mode ID、μ/σ 与 noise，decoder 变换成[B,K,32,54]；后续 E47 几何与预测接口不变。
- **风险**：仅成功轨迹时，最大似然可能集中在已见示范模式；per-joint 独立高斯可能给出不协调手型，分量方差会用过大值掩盖误差。采用序列级 mode，按 54 维组和 FK 检查协方差，并报告 component utilization。

### 1.6 多假设 / K-query proposal head

- **核心机制与出处**：共享 context，输出 K 个候选 query 或候选 latent，再用 winner-takes-all 或 mixture likelihood 对目标 chunk 训练。Multiple Choice Learning (2012) 提出对一组结构化预测做 best-of-set 学习；和基于采样的 DP/flow 不同，网络一次前向直接产 K 个候选。
- **本地可行性**：中等，可挂在 flow latent 或小型 action decoder 上，GPU 推理便宜；需要新训练，不建议同时改变 E47 世界模型。可先用冻结 P1/基线 context，单独测 head 的学习成本。
- **算力**：K 个 query 带来约 K 个小动作输出；共享 context 编码，预测动作 head 的成本近似随 K 增长，但无 K 条独立 diffusion solver。
- **接口**：输出 K×32×54，保留候选索引；统一送入 forecast/score，不做候选均值。
- **风险**：min-over-K 只把动作梯度给当前 winner，容易坏死模式或重复候选；单一专家目标会使多条候选都围绕同一 target。需要 mode 负载监控/配对覆盖证据，不能把 diversity loss 变成无约束关节排斥。

## 2. 采样期引导、优化与重排

| 方法 | 核心机制 / 一手来源 | 本地适配、开销与主要风险 |
|---|---|---|
| Best-of-N + verifier / world-model rerank | 一次产 K 个候选，对每个预测后果并按目标代价/成功概率排序。DINO-WM (2024/ICML 2025) 以目标 DINO patch feature 对预测 latent 评分；近期 Best-of-N WAM 研究 (arXiv 2026-10，尚未同行评审) 指出“一次执行一个候选取最好”的 hindsight 估计会混入 rollout 随机性，并建议重复 continuation、common random numbers 与 fail-gated selector。 | 最直接匹配 E47 的 forecast_candidates→score→select。E47 forecast latent/mask 是计划接口，P1 未实现/验收；现行首要物体结果 oracle 仍是 MuJoCo。以 MuJoCo 实际候选结果校准 scorer；用预测风险/不确定性作为保守门控。成本约 K 次 forecast，context 可共享；风险是 oracle 可选上限虚高、单一 demo 误差冒充任务分数、模型 exploitation。 |
| 分类器/价值头引导 | 训练 classifier p(success|ctx,forecast) 或价值函数 V(forecast,goal)，对扩散/flow 的生成路径提供任务梯度，或只在最终候选排序。DynaGuide (2025, NeurIPS 2025) 用独立 dynamics latent 与期望/不希望图像的距离，在 diffusion denoising 中加梯度引导；展示了在既有策略上引导稀有行为。 | 本地第一版应优先做末端 rerank，不做每去噪步的梯度引导。只有 MuJoCo 配对候选产生实际结果标签后，非 VLM 小型分类/排序头才有监督依据。Flow 不是 diffusion：把 DynaGuide 梯度注入 10 步 Euler 的公式未经本地验证。求导须穿过 world model/生成过程，耗时和 exploit 风险高。 |
| CEM / iCEM / MPPI | 将候选动作序列作为搜索变量，rollout latent dynamics，按 goal/reward/约束选 elite，再更新分布。DINO-WM 直接在动作序列上做 CEM；iCEM (2020/CoRL 2020) 加 temporally-correlated sampling 和 warm-start/memory 提高采样效率；MPPI (2017) 用轨迹代价权重更新动作序列。 | “在世界模型隐空间优化”应理解为：搜索变量仍是可执行动作/低维 action latent，E47 F 对它们推进 z；不要在 z 空间直接选一个不可解码的 latent 当动作。54×32 直接独立高斯维度太高；先在 action flow chunk 周围优化 6–16 维时间基、arm/hand synergies 或阶段 mode。每轮 N 条 × I 轮 × 多个 forecast horizon，开销远高于纯 best-of-N；trust region 与 MuJoCo 验证是必要条件。 |
| 局部梯度优化 / action guidance | 对 differentiable F 的目标代价对动作求梯度，更新候选；或者如 DynaGuide 把梯度注入生成器。 | 适合连续、平滑、局部可微的低维变量。接触切换/离散抓取、遮挡和 F 的未校准 Jacobian 会让方向错误；很可能找到模型缺陷而非真实可行动作。E47 文档已指出 flow encoder 梯度治理，但那不是动作优化正确性的证据。 |
| 独立 learned verifier | 学 Q(ctx,action) / P(success|ctx,forecast)，为候选打分，而不是复用生成器的训练损失；可用真实/模拟器成功失败 pair 做 pairwise ranking。 | 可能在低成本下区分 task outcome，但 93 条成功示教没有失败/近错覆盖，单独用 demo action RMSE 会让“更像示教”被误当“更成功”。先用 MuJoCo 出真实反事实标签，报告 score 对物理 outcome 的 AUC/pairwise accuracy；输出需校准并支持 reject/hold。WorldGym 是评估框架而非训练此头的证据，且其论文使用的 VLM 奖励不适用于本项目约束。 |

**搜索约束**：候选得分至少含任务目标进展、关节/碰撞可行性和不确定性风险；固定 flow proposal prior 的偏离惩罚或可信域，限制优化不能无限远离动作数据。只对通过合法性过滤的 proposal 排序，但保留全部原始候选 ID/分母与拒绝原因；全候选拒绝时按 E47 合同回退 hold。

## 3. 候选生成器与世界模型如何训练/耦合

| 工作 / 年份 | 核心机制与训练关系 | 天机可行性与接口 / 计算 / 风险 |
|---|---|---|
| V-JEPA 2-AC (2025) | 大规模无动作视频 JEPA 预训练后，以机器人动作条件化 predictor；控制时将 action-conditioned latent rollout 与目标视觉 latent 做 MPC。官方 V-JEPA 2 repo 含 AC 模型与训练配置。 | 是“预测后果实际参加规划”的高质量范例，但 1.2B 规模基座与 DROID 机器人后训练数据不等于 93 条从头可学；本地不宜重新预训练。接口概念上接 E47 forecast，compute 很高，视觉 goal 的任务距离不保证抓稳/入箱。 |
| DINO-WM (2024，ICML 2025) | 冻结 DINOv2 patch encoder，离线轨迹监督动作条件未来 patch，测试时用 CEM 优化 action sequence 对齐 goal patch；可选像素 decoder。是模型学习与在线动作优化相分离。 | 与 E47 的 patch latent 预测/候选 rollout 最接近，且搜索可以复用已训练 F。论文数据涵盖 PointMaze、Push-T、绳/颗粒等，实验数据量从千至万级；不是 93 条成功双臂手数据。Goal feature 易忽略接触，得用 MuJoCo 结果验证。 |
| Ctrl-World (2025 预印本；ICLR 2026) | 将 SVD 初始化的多视角视频 diffusion fine-tune 成动作条件、姿态条件 world model，允许 policy-in-the-loop imagination，用于比较/改进通用 policy。它训练的是 generative WM，不是与本地 flow proposal 一同以 best-of-K loss 学出来的候选生成器。 | 论文使用 DROID 约 95k 轨迹、SVD 级算力、多视角/逐帧 action/pose 历史；对单顶视小数据不具成本可行性。可借 action conditioning、失败回放/可控性诊断，不能搬用视频逼真作为打分信号。 |
| WorldGym (2025 预印本；ICLR 2026) | 动作条件生成模型作为 policy evaluator 环境，重复 rollout 再用视觉语言奖励汇总策略成功率。定位是评估完整策略，不是 joint training 候选生成器或局部候选 verifier。 | 只借“策略级 rollout 和统计”，不能依赖 VLM scorer；顶视双臂接触细节和物体 shape 幻觉是风险。对本地应改用 MuJoCo 可计算结果做 oracle calibration。 |
| DreamerV3 (2023 预印本；Nature 2025) | RSSM 学 latent dynamics、reward/continue 和 observation head；actor/critic 在 imagined latent trajectories 上同时优化。候选由学到的 actor 分布产生，world model 训练和控制目标共同迭代。 | 方法需要环境 reward 和真实/离线交互覆盖；93 条专家成功路径不足以替代在线 RL 的奖励/状态分布。15Hz 54D grasp 里 model bias 会放大。原则可借“候选策略要直接因果改善预测回报”，工程不建议照搬 RL loop。 |
| TD-MPC2 (2023 预印本；ICLR 2024) | encoder、implicit latent dynamics、reward/value 和 actor 联合训练；推理时以短 horizon trajectory optimization（常用 CEM）在 latent world model 中局部搜索。 | 其公开训练基于百余任务、数亿 transition，主干是 model-based RL；不能据大规模结果推断单任务 93 episodes。可借低维搜索、actor warm-start 和 model/value 联合，但本地要先有真实物理 outcome 与成本预算。 |
| LAWM / latent-action planning (2025) | Latent Action Pretraining Through World Modeling：从有/无 action labels 的视频学 compact latent action 与未来 latent 预测，再适配 BAKU/DP 等策略。它主要是表征/预训练路线，并不自动意味着部署时运行一个 WAM 对每个候选评分。Latent Diffusion Planning (2025, ICML 2025) 分开 latent trajectory planner 与 IDM，利用无动作/次优数据在视觉 latent 中规划后译成 motor action。Hierarchical Planning with Latent World Models / HWM (2026, CoRL 2026) 将 primitive action chunk 编码为 macro latent action，分时间尺度用 CEM 做层次计划。 | latent/macro action 可能显著降低 1,728D 的搜索维度，适合未来 P2/P3 的高层 proposal；然而本地固定 15Hz 手指接触需要低层 54D 可行解码器，当前 93 条示教也没提供充分高层模式监督。LAWM 需视频先验，LDP/HWM 多模块和长预测增加成本；只把高层变量与 E47 F/MuJoCo 解码后果接通才算有候选筛选。 |

**类别澄清**：联合 loss、预训练、预测未来辅助动作头、推理时用未来选动作是不同级别。P2 要声明是哪一级。若 candidates 变化不改变 P3 排名或真实 outcome，world model 就没有参与动作选择。WorldGym 只提供 policy evaluation 概念，Dreamer/TD-MPC2 则要在线奖励/交互，均不等价于当前离线候选生成实验。

## 4. 多样性度量与训练目标

### 4.1 候选集合怎么量

1. **imitation oracle 质量**：每个 chunk 计算归一化动作误差；汇报 K 候选 mean、median、min（best-of-K）相对 K=1、hold、示教动作的改善；按时间 h=4/8/16/32 和 arm A/B、左右手分组报告。best-of-K 的单条 demo 最小误差只叫 imitation oracle，不能叫 task-success upper bound。
2. **多样性/重复度**：在 fit-only 统计归一化动作空间测候选两两距离、nearest-neighbor distance、unique ratio；另报各 4 个动作组的标准差/协方差有效秩、末端 FK/掌点轨迹距离、手闭合 synergy 差异。标准差须相对训练 σ 归一；只报 raw std 会把单位和自然关节范围混在一起。候选被限位过滤后再报可行集合的有效多样性。
3. **物理可选上限**：在可复位的 MuJoCo 初始状态，对同一状态执行 K 候选，记录成功率、碰撞/掉落/限位率、最优实际 outcome、regret to MuJoCo oracle。真实示教 future 与换动作 future 不能配对当作新标签；没有真实分支/模拟器反事实就不具 outcome oracle。
4. **评分器可区分性**：用固定起点的候选 pair，比 forecast score 的排序与 MuJoCo outcome 差异，报告 pairwise accuracy/AUC、Kendall/Spearman、top-1 success、calibration/Brier、模型选择相对随机的配对增益和置信区间。若候选 outcome 几乎相同，就应先停止加大 scorer。
5. **多样性维持质量**：按 candidate ID 分层展示 mode frequency、无效比例、重复候选比例；端到端必须同时给“覆盖质量”和“筛选可用”，不能用多样性单指标晋级。

### 4.2 训练损失选项

- **winner-takes-all / minimum expected error**：对 K 个输出只回传误差最小者 L_WTA=min_k ℓ(a_k,a*)。它直接优化 best-of-K 模仿误差，与“多候选中找到一个像 target”的目标一致，单个 target 每步动作成本低。但无 mode balancing 时非 winner heads 收不到梯度，易死掉/重复；一条 demo 不会告诉模型哪些不同动作同样成功。
- **多假设 softmin / matching**：先将一个 chunk 统一分配给获胜 mode，再对该 mode 回归；或用温度软化 softmin 让多个近邻 head 暂时学到相似样本后逐渐分化。全 chunk assignment 比对每帧/每关节独立取 min 更能保时序连贯性。
- **Mixture likelihood**：对整段动作计算 log Σ_m π_m N(a*;μ_m,Σ_m)，同时学模式概率与条件分布，通常稳定使用 likelihood，但最大似然可能让所有样本集中在频繁模式；需看 component use 与 heldout log-likelihood。
- **多样性/repulsion 项**：在同一状态的候选对间加 margin/hinge repulsion，或对多假设 head 作负相关/DPP 风格分散；建议只在通过初步合法性筛选的候选上，使用低权重、按动作组和 FK task-space 距离，且保留候选 quality cap。对原始 54 维全量排斥会推高关节角、制造限位/自碰撞和不连续手型。
- **推荐组合**：chunk-level M=4 的 mixture likelihood 或 K=4 proposal + 全 chunk WTA，加 mode-use 诊断；等存在同起点 MuJoCo 成功/失败反事实后，才增加基于 outcome 的 pairwise rank/score objective。先冻结 E47 world model，不把候选策略与 F 联合端到端改动，以免混淆“候选质量”与“模型评分偏差”。

## 5. URDF / FK / 手部几何约束

| 层级 | 建议机制 | 本地接口与注意事项 |
|---|---|---|
| joint position | 用 URDF hard limits 与更保守 safety/soft limits 检查 54 个关节每一帧；按 E47 joint_order A7+B7+左手20+右手20、单位 rad。可以投影/clip，但需报告投影幅度；整段大幅 clip 表示 proposal 原本不可信。 | E47 命令反归一化公式为 q_abs=q_latest+a*sigma_a[offset]+mu_a[offset]，所有动作按最新实测 q 重新锚定；不能将预测关节回灌为观测。换成 abs q 后检查 limits，再由 normalize_commands 重算 a。 |
| 速度/平滑 | 15Hz 对 q[j]-q[j-1] 检查 velocity/acceleration/jerk 与最大变化；含初始 command 与 measured q_t 的第一步差。值需匹配硬件/控制器限幅，不能拿示教统计 σ 代替安全限制。可在 action latent trajectory basis 上优化平滑。 | 如 URDF 未含加速度/jerk，MoveIt 配置可独立给 motion planning limits；需要由硬件/驱动规格审定。重规划时只执行 k 步并用新实测 q 重新锚定。 |
| FK / workspace | 每个绝对 command 用已校验 URDF/MDH FK 算双臂 EEF、掌框和中间路径；硬过滤越界/奇异姿态/超过 workspace，或对 end-effector continuity 加软罚。 | 现有 tools/e46_project.py 与 adopted calibration/hash 是唯一本地几何起点。E47 的 geometry G=66 含两侧掌点uv、低分辨率 ROI、rot6d、20D hand_q 和 valid；它不含 fingertip depth，不能声称 3D 接触有效或手指 collision-free。 |
| collision / hand geometry | 用 URDF/SRDF collision geometry 计算 arm-arm、arm-table/object、hand-object 与 swept chunk 碰撞；多指 joint limits + 拟合的 grasp synergy（如 open/close 与几组 finger synergy）可作低维候选变量，但最终仍用逐手指 joint FK/碰撞筛。 | MoveIt 2 planning scene 或 cuRobo 有 URDF/碰撞/自碰撞路径；cuRobo 的 differentiable collision costs 可用于后续优化。需准确的天机碰撞 mesh、物体/桌面模型和允许接触矩阵；只有 joint limits/hand geometry 不能判断“抓住”或“瓶在手里”。 |
| invalid/recovery | 输出 valid[B,K] 和 rejection reasons；每个 K 固定入账。全无效时回退 hold；score 不能靠删除难候选改变分母。可选将轻微越界投影后重新 FK，否则 reject。 | 这与 E47 forecast_candidates/score/select 合同相符；对未来真机约束应另作工程验收，不把候选过滤称为安全认证。 |

建议分两级用约束：proposal 训练时输入 FK/geometry 及动作组编码；采样后做 hard feasibility/filter，再用 MuJoCo 验证物体后果。Hard filters 应先于学习评分，避免 scorer 用高目标收益掩盖硬限位/自碰撞。

## 6. 排序后的最小验证实验（仅供下一阶段注册）

以下 GPU 小时是**预算估算，不是实测**，未从 P1 既有 8.0h 预算挪用；P1 checkpoint/forecast interface 还未验收，只有准入后才开实验。MuJoCo candidate outcome rollouts 预期为 CPU，不计 GPU 时；forecast/learned scorer 前向和小头训练计 GPU 时。所有实验固定已有 episode 起点，candidate order/noise 与 seed 可复现；先在开发窗调试，再用锁定协议做确认。由于 E46/E47 文档记录 eval 曾被历史使用，eval 结果应注明是冻结开发验收，不称全新独立确认。

| 排名 | 实验设计 | 必报指标 / 停止条件 | 预计 GPU 时 |
|---|---|---|---:|
| 1：flow 多样本可选上限 | 同一已训练 E33 flow checkpoint、同 context，K=1/2/4/8/16，nested noise，温度先固定 1；与 best-of-K 以及 hold 同口径比较。同步按 MuJoCo 可复位起点做 K 候选实际 rollout；首轮不训练不引入 CEM。 | imitation best-of-K RMSE（全 chunk 与 4 组、h4/8/16/32）、mean/median、相对 hold 改善；pairwise normalized q/FK 距离、unique/invalid rate；MuJoCo actual outcome best-of-K/headroom。若 K=8→16 无显著 headroom 且候选有效多样性不升，停止扩大 K/调温度。 | 0.15–0.25 |
| 2：全 chunk mode proposal 小试 | 只在 E47 context 和原 flow/BC action objective 上添加 M=4 chunk-level mixture 或 K=4 WTA query head；freeze E47 world model 和视觉 encoder，两个配对 seed、相同训练 episode/有效 batch/固定短预算。用 K=1 flow 作为计算匹配基线。若 head 坍塌则只记录，不做同时训练 scorer。 | candidate mean/min imitation error、候选 group/FK diversity、component utilization、合法率、与实验 1 同口径 MuJoCo best-of-K。目标不是提升训练 loss，而是相同计算下增加可行 headroom；若 only diversity 上升但 MuJoCo outcome 不升，停该生成器。 | 0.8–1.2（两种子 pilot，约各 0.4–0.6） |
| 3：小维度 iCEM + E47 forecast 排序审计 | 从 flow proposal 开始，以 6–16 维动作基/synergy/whole-chunk mode 为 search variable，iCEM warm-start，trust region；最多 2–3 轮，proposal ≤32/轮。每候选用 E47 forecast score，同时在同初态 MuJoCo 得到 outcome；对照 flow best-of-N、随机选和 MuJoCo oracle。第一个版本冻结 scorer，只测它是否区分。 | score 对 MuJoCo outcomes 的 pairwise accuracy/AUC、Kendall/Spearman、top-1 success、calibration、invalid rate、oracle regret、GPU latency p50/p95。若排序置信区间覆盖随机、或优化后动作在 MuJoCo 变差/靠近限位，停止梯度/搜索扩张，先修 model/objective。 | 0.3–0.5 |

预算建议排序的理由：先确认当前 proposal 有没有 selectable headroom；再用小头增加候选空间；最后测世界模型评分/搜索能否兑现实际物理结果。世界模型 joint-training 不是第一轮：先冻结 E47 P1，避免同时引入“候选分布变化、F 动力学变化、评分器变化”三个变量。若第 3 项证明存在未用 headroom 且预测排序可靠，另注册联合训练/闭环实验；若失败，预算不应用于扩大 K 或更重生成器。

## 7. 文献与代码引用

### 候选生成
1. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (arXiv 2023; RSS 2023; IJRR 2025). [论文](https://arxiv.org/abs/2303.04137) · [作者官方代码](https://github.com/real-stanford/diffusion_policy)
2. π₀: A Vision-Language-Action Flow Model for General Robot Control (2024). [论文](https://arxiv.org/abs/2410.24164) · [openpi 代码](https://github.com/Physical-Intelligence/openpi)
3. FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching for Robot Manipulation (2024 preprint; AAAI 2025). [论文](https://arxiv.org/abs/2412.04987) · [代码](https://github.com/zql-kk/FlowPolicy)
4. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware / ACT (2023; CoRL 2023). [论文](https://arxiv.org/abs/2304.13705) · [ALOHA project](https://tonyzhaozh.github.io/aloha/) · [official ACT code](https://github.com/tonyzhaozh/act)
5. Behavior Transformers: Cloning k modes with one stone (2022; NeurIPS 2022). [论文](https://arxiv.org/abs/2206.11251) · [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2022/hash/90d17e882adbdda42349db6f50123817-Abstract-Conference.html) · [author code](https://github.com/notmahi/bet)
6. Behavior Generation with Latent Actions / VQ-BeT (2024; ICML 2024 Spotlight). [论文](https://arxiv.org/abs/2403.03181) · [official code](https://github.com/jayLEE0301/vq_bet_official)
7. FAST: Efficient Action Tokenization for Vision-Language-Action Models (2025). [论文](https://arxiv.org/abs/2501.09747) · [author project](https://www.pi.website/research/fast) · [FAST+ tokenizer and training code](https://huggingface.co/physical-intelligence/fast)
8. BAKU: An Efficient Transformer for Multi-Task Policy Learning (2024; NeurIPS 2024). [论文](https://arxiv.org/abs/2406.07539) · [project/results](https://baku-robot.github.io/) · [code](https://github.com/siddhanthaldar/BAKU)
9. Multiple Choice Learning: Learning to Produce Multiple Structured Outputs (2012; NeurIPS 2012). [paper](https://proceedings.neurips.cc/paper/2012/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf)

### Guidance / search / evaluation
10. DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance (2025; NeurIPS 2025). [paper](https://arxiv.org/abs/2506.13922) · [project and code](https://dynaguide.github.io/)
11. DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning (2024; ICML 2025). [paper](https://arxiv.org/abs/2411.04983) · [official code](https://github.com/gaoyuezhou/dino_wm)
12. Sample-efficient Cross-Entropy Method for Real-time Planning / iCEM (2020; CoRL 2020). [paper](https://proceedings.mlr.press/v155/pinneri21a/pinneri21a.pdf) · [PyTorch implementation](https://github.com/UM-ARM-Lab/pytorch_icem)
13. Information Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving / MPPI (2017 preprint). [paper](https://arxiv.org/abs/1707.02342)
14. How Much Can Best-of-N Help a World Action Model? Measuring Selectable Headroom and Realizing It in Closed Loop (2026-10-27 arXiv preprint; not yet peer reviewed at survey date). [paper](https://arxiv.org/abs/2610.37196)

### Joint or connected world-model / policy methods
15. V-JEPA 2 / V-JEPA 2-AC (2025). [paper](https://arxiv.org/abs/2506.09985) · [official code/models](https://github.com/facebookresearch/vjepa2)
16. Ctrl-World: A Controllable Generative World Model for Robot Manipulation (2025 preprint; ICLR 2026). [paper](https://arxiv.org/abs/2510.10125) · [official code](https://github.com/Robert-gyj/Ctrl-World)
17. WorldGym: World Model as an Environment for Policy Evaluation (2025 preprint; ICLR 2026). [paper](https://arxiv.org/abs/2506.00613) · [official code/project](https://github.com/world-model-eval/world-model-eval)
18. Mastering Diverse Domains through World Models / DreamerV3 (2023 preprint; Nature 2025). [paper](https://arxiv.org/abs/2301.04104) · [official code](https://github.com/danijar/dreamerv3)
19. TD-MPC2: Scalable, Robust World Models for Continuous Control (2023 preprint; ICLR 2024). [paper](https://arxiv.org/abs/2310.16828) · [official code](https://github.com/nicklashansen/tdmpc2)
20. Latent Action Pretraining Through World Modeling / LAWM (2025). [paper](https://arxiv.org/abs/2509.18428) · [official code](https://github.com/baheytharwat/lawm)
21. Latent Diffusion Planning for Imitation Learning / LDP (2025; ICML 2025). [paper](https://arxiv.org/abs/2504.16925) · [code](https://github.com/amberxie88/latent_diffusion_planning)
22. Hierarchical Planning with Latent World Models / HWM (2026; CoRL 2026). [paper](https://arxiv.org/abs/2604.03208). [author code](https://github.com/kevinghst/HWM_PLDM)（公开实现覆盖 PLDM minimal setup；README 称其他 WM/task 的完整结果见项目页）.

### Kinematics, constraints, and collision checking
23. MoveIt 2 URDF/SRDF, joint limits and collision checking tutorial. [official docs](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/urdf_srdf/urdf_srdf_tutorial.rst) · [joint-limit configuration guide](https://github.com/moveit/moveit2_tutorials/blob/main/doc/how_to_guides/moveit_configuration/moveit_configuration_tutorial.rst)
24. cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation (2023). [paper](https://arxiv.org/abs/2310.17274) · [official code/collision checking](https://github.com/NVlabs/curobo)
25. Differentiable and Learnable Robot Models (2022). [paper](https://arxiv.org/abs/2202.11217) · [official code](https://github.com/facebookresearch/differentiable-robot-model)
26. PyTorch Kinematics (software). [project](https://github.com/UM-ARM-Lab/pytorch_kinematics)

---

**交付边界**：本文提出后续可验证的 P2 方案与预算，不代表 E33/E47 已重新评测、world-model scorer 已可用、MuJoCo candidate rollout 已通过，或 GPU 性能已实测。
