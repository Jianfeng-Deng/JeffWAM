# 任务 E37：阅读 arXiv:2611.01729 与 NTUMARS/ODE-POLICY 代码，总结并分析对我们的适用性（luna max）

## 要做
1. 读论文 arxiv.org/abs/2611.01729（HTML 版 arxiv.org/html/2611.01729 若有），以及 GitHub 仓库 NTUMARS/ODE-POLICY（只读，git clone 到 runs/e37_20261104/ODE-POLICY 或 scratch 目录，不安装进系统环境，不运行其训练）。
2. 论文总结（中文）：问题、核心方法（模型结构、动作如何生成、训练损失、推理流程/步数/ODE 求解方式、与 flow matching/diffusion policy 的区别）、实验设置与主要结果（数据集、基线、数字）、局限；仓库结构：关键文件/类/函数与论文章节的对应、默认超参、动作 chunk 与推理延迟。
3. 适用性分析（对照我们的现状，见 notes/2026-11-04-e36-mismatch-cand-brief.md "已知事实"与 tools/e33_model.py）：
   - 我们的约束：天机双臂+Wuji 手，54 维关节动作、32 步 chunk、顶视单相机、示教数据仅 93 episode、JEPA 式共享潜变量+世界模型；动作生成目前是 10 步 Euler 的 flow matching 单样本。
   - 逐条判断论文里的思想能否借鉴：能直接借用的（给出改哪个函数）、需要改造的、不适用的及原因；特别关注 (a) ODE 求解/步数/一致性蒸馏与推理加速 (b) 对候选动作生成/评分/多模态的支持 (c) 与世界模型/未来预测的结合方式 (d) 数据效率。
   - 给出最多 3 个、按优先级排序、可在 ≤2 GPU 小时内验证的实验建议（改动、对照、指标、预期风险）。
4. 说明哪些结论来自论文/代码原文、哪些是你的推断；不要复制大段原文（引用不超过一句）。

## 约束
只读外部仓库；不修改 tools/e3*、tools/e2*、既有 runs；colleague-e、colleague-c 严格只读；不启动训练。输出 notes/2026-11-04-e37-odepolicy.md（中文 ≤150 行）。
