# JeffWAM

双臂（Piper）扎带穿孔任务的 **World-Action Model（WAM）** 研究代码快照，以及全部实验报告。
A research snapshot of a bimanual (Piper) zip-tie threading **World-Action Model**: code, experiment reports and robot models.

> 状态 / Status：研究代码，**不能开箱即用**。数据、权重、特征缓存、视频不在仓库里；代码里的绝对路径指向作者的个人工作站（同事用户名、内部主机名、IP 已脱敏为占位符）。本仓库用于审阅、复现思路和审计，不是发布版框架。
> Research code, not turnkey: no data, weights, caches or videos are included; absolute paths point to the author's workstation (colleague usernames, hostnames and IPs are replaced with placeholders).

## 内容 / Layout

| 目录 | 说明 |
|---|---|
| `code/code_e*/` | 各实验的代码（`e82`→`e122`），每个目录对应一次实验或一代模型。主要模型实现：`code_e86`、`code_e88`（v3.1）、`code_e98w`、`code_e99w2`、`code_e102a`（动作头 A2） |
| `notes/` | 实验报告与每轮结论（含所有失败与否定结果）；`worklog.md` 是按时间的决策记录 |
| `docs/` | 方向分析、文献综述、实验日志（`docs/experiment_log/EXPERIMENT_LOG.md`） |
| `robot_models/marvin_m6/` | Marvin M6 双臂 URDF 与 STL 网格（第三方厂商资源，见 `NOTICE.md`）。**Piper 的正运动学链是写在代码里的常量，没有 URDF 文件** |
| `MODEL.md` | 当前模型结构说明 |

## 当前默认模型一句话

冻结的 SigLIP2 视觉 token（3 相机 × 2×4 × 768）→ O/A/W 三组 token 的共享 Transformer 主干 → 确定性动作头（50 步 Δq + 夹爪开/合）与均值世界头（预测 +10/+25/+50/+100 帧特征残差）；旁路有一个"21 个谓词 → 10 个阶段 → 在线 HMM"的阶段控制器。详见 `MODEL.md`。

## 主要结论摘要（均来自 `notes/`，数据规模与口径见各报告）

- **世界头**：训练步数越多越好，20000 步仍未饱和；外部数据（ABC、yam）预训练与混训均无收益；多峰/flow/能量分数建模样本塌缩，无收益。
- **动作头**：确定性 Δq 头优于 flow 头；因果本体历史输入至关重要；夹爪切换时刻（事件 F1）仍是弱项，因果解码规则无效。
- **阶段控制器**：谓词头 AUROC 约 0.91（因果标签训练）；在 ±90 帧容差下阶段边界召回约 0.73；历史输入未带来收益；因果阶段标签与回顾式标签同帧仅约 50% 一致。
- **阶段信息进动作头**：真实阶段也无可测收益（P1）；预测未来特征上的谓词读数可反映任务进展，且对动作敏感（P2）。
- **语言**：现成 SigLIP2 图文空间对本任务的关系性描述几乎不对齐（P3：零样本阶段 top-1 0.036）；语言条件在"强制依赖"的训练下才被使用，且只学到模板而非语义（E119–E122）；MaskCLIP 式桥能恢复颜色/物体的逐 patch 定位（E120）。
- **局限**：单任务、单演示者、dev 样本少；dev 阶段/谓词标签由模型生成，非人工；没有仿真或实机闭环验证。

## 约定与限制

- 仅 Piper 数据；`episode142` 永久排除。
- 数据与标签：模型生成标签仅作训练目标和开发参考，不作测试真值。
- 第三方：SigLIP2（Apache-2.0）在代码中通过本地权重加载，未随仓库分发；其余见 `NOTICE.md`。
- 暂未选择开源许可证（未包含 LICENSE 文件），默认保留所有权利。
