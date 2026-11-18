# E103-D 外部无动作视频世界辅助训练

权威状态：`results/e103d_control_20261115_01/PIPELINE.json`；报告：`notes/e103d_external_data_20261115.md`。

- 基线：只读 E100-C C2 `results/e100c_c2_s90_20261115_03/ckpt_05000.pt`。严格恢复模型和 AdamW 状态；seed90、lr3e-4、各5000更新，唯一终点。
- C2 原训练只有P3。当前成对方案保留每步Piper P3 batch32、原四模式与C2链；两组均增P1 batch32世界辅助项，系数.1。D0 P1全Piper；D1每批20 Piper、6 yam、6 ABC，缺源回拨Piper。P3采用全局5001..10000，链权重.1，预测重入概率.8。
- `e103d_model.py` 保持C2参数键和原Piper前向；P1只拼接O/W，物理删除state/A/R，不构造机器人轨迹，不计算动作、对齐、阶段或链损失。CPU测试用禁用投影hook与梯度断言检查。
- 原位读取ABC S1/train/RS640约10h，yam仅49 allowlist减指定6。1秒锚点、历史[-24,-18,-12,-6,0]、目标[10,25,50,100]，30Hz物理时距。因果O age≤100ms、三路跨度≤50ms、W原生近邻容差、跨窗口gap规则；无平移或图像插值。
- `e103d_plan.py` CPU清单；`e103d_audit.py`源身份与时间统计；`e103d_cache.py`旧阶段（保留审计）；`e103d_cache03.py`一次MCAP三路读取/3个CPU解码worker/同一GPU，完整解码帧数及无B帧检查；已完成receipt只读校验后复用，未完成缓存隔离保留。
- 特征完全沿用冻结SigLIP2、E84 FOV预处理和2×4 fp16池化；两个Piper训练parent、6路×128帧复现最大差0。原片只为复现读取迁移后的`dataset/piper`，旧缓存不修改。
- `e103d_train.py contracts` 完整CPU合同；`e103d_train.py D0/D1`训练；`e103d_eval.py D0/D1`固定评估；`e103d_eval.py finish`成对bootstrap/判定。`e103d_pipeline02.py`已分离执行，有效训练run后缀为_20261115_02；run01的D0仅因监控同步开销中断并保留，不参与评分，禁止原目录重跑覆盖。
- 每GPU阶段在当下空闲显存≥6000MiB×1.5时准入，2秒后台持续监控，每步检查峰值和预算，峰值上限6000MiB；GPU总墙钟≤21600s，失败和中断段计入。全部新增≤40GB。资源台账仅本会话目录。
- dev只用固定人工任务指令；ABC/yam过程语言不接入本轮，通用短句为Manipulate objects on a tabletop.。Piper训练保留原E83指令/短句fallback。模型语言不作测试真值、阈值或选点依据。
- 旧代码、旧ckpt、同事目录与dataset只读；不使用仿真/MuJoCo/retarget；永久排除piper_4dc107335389f5711a0f65e0。

不要直接执行复制来的旧launch/main脚本。它们仅保留导入和审计用途；本轮入口均为e103d_前缀。
