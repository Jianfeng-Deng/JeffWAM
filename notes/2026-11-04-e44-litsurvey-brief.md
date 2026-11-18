# 任务 E44：文献与开源项目调研——世界模型如何真正参与动作生成/选择（sol high；只调研与分析）

## 背景（我们的现状，供你对照）
- 天机双臂+Wuji 手抓瓶放箱：93 条示教、顶视单相机、54 维关节、32 步 chunk。当前最佳 E33-B：DINOv2 编码共享潜变量 + 动作条件 predictor（JEPA 预测 +16/+32）+ flow-matching 动作头（单样本，10 步 Euler）。离线比 hold 好约 26–32%。关键缺陷（E36 实测）：predictor 学到了动作信息（真实动作 future 比 hold future 更接近真实未来 42–50%），但策略头只读 hold 条件的 future，世界模型不参与动作评估/选择；把真实动作 future 喂给策略反而变差 62–65%；best-of-16（真值选）上界只有约 18–19%，候选多样性很小；无任何 reranker 通过动作敏感性门槛。子目标生成（ID-WM 式）路线失败已终止；time-to-go 价值模型排序未过门槛（demo 优于其它动作仅 48–56%）。
- Piper 双臂穿扎带：173 条（79 遥操+94 DAgger）+136 条策略回放录像、三相机（含腕部）、14 维。E32 在 8 GPU 小时后右臂显著优于 hold、左臂不显著；动作互换测试 recorded 胜 shuffled 93–97%（世界模型对动作敏感）；gaze 视角与加大重建权重（E41）都没有改善动作；夹爪不比 hold 好；数据中 action≡下一时刻 qpos（无滞后）。
- 两个项目都还没有闭环成功率；同事验证过微调 OpenWAM（WAN2.2-5B 视频 DiT + 动作 DiT 联合 flow matching，MoT）在真机可用。

## 调研问题（每条给论文/项目名、年份、链接、核心做法、是否开源、与我们是否相关；只写确认存在的文献，找不到就说找不到）
1. 世界模型作为**验证器/评分器**选择候选动作（policy steering / reranking / MPC）：如 V-JEPA 2-AC 规划、DINO-WM、Ctrl-World、World-model-based policy evaluation（WorldGym、WorldEval 等）、FOREWARN、DynaGuide、V-GPS、RoboMonkey 等；它们用什么评分信号（目标图像距离、VLM 评判、价值函数、进度），需要多少数据，是否有"动作敏感性"问题及如何解决（反事实/扰动/DAgger 数据）。
2. **联合训练视频/潜变量世界模型与策略**并在推理时使用世界模型：UniPi、UVA（Unified Video Action）、Video Policy、PAD、VPP、GR-2/GR-3、Seer、FLARE、WorldVLA、Genie Envisioner、DreamGen、UniVLA、OpenWAM/类似 WAM 等；区分"世界模型只做辅助损失/表征" vs "推理时参与生成或选择"，以及小数据下哪个更有效的证据。
3. **进度/价值/子目标**作为评分信号：GVL（Generative Value Learning）、VLAC、LIV、Rank2Reward、ReWiND、PROGRESSOR、基于子目标/阶段标注的奖励等；它们如何避免"只学会时间进度、对动作不敏感"。
4. **动作敏感性/因果混淆**：单一路径成功示教导致世界模型/价值函数对动作不敏感的文献与对策（随机扰动数据、DAgger/rollout 失败数据、反事实增强、play data、inverse dynamics 正则）。
5. 小数据双臂灵巧操作的有效做法（几十到两百条示教）：动作 chunk、时间集成（ACT temporal ensembling）、扩散/flow 策略的多样性与模式坍缩、预训练表征/预训练 VLA 微调（π0/π0.5、OpenVLA-OFT、RDT 等）的数据效率证据。
6. 可选：评估方法——离线指标（对 hold 的误差）与闭环成功率的相关性研究。

## 输出要求
- notes/2026-11-04-e44-litsurvey.md（中文 ≤220 行）：按问题分节的表格（名称/年份/链接/做法/开源/相关性），每节末尾 3–5 条"对我们的启示"；最后给出综合判断：针对我们两个项目，最值得借鉴的 5 个具体做法（按预期收益/成本排序，写清楚要改什么、需要什么数据、怎样验证）。
- 区分"文献原文结论"与"你的推断"；不复制大段原文；不确定的年份/数字标注"未核实"。
- 只读：可联网检索、可只读浏览/克隆开源仓库到 runs/e44_20261104/（不安装、不运行训练）；不修改项目代码与既有 runs；colleague-e、colleague-c 严格只读。
