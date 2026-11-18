# E55-1 简报：Tianji 目标距离评分器 v2（物体中心表示 × 三种目标）

日期：2026-11-06（BRT）。执行：luna-xhigh。设计依据：T/notes/2026-11-06-e55-scorer-v2-design.md（先读，含两处修订；本任务**不做**悲观/弃权/不确定性/OOD，那是后续步骤）。T=/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu；W=/mnt/workspace/dengjianfeng/code/wam_common。

## 背景（已有结论，不要重复验证）
- E48–E53：B1 事件二分类评分头（演示窗口训练）在 8 个评估集上 G_rank≈0.47–0.52，不过；原因是只有演示窗口、没有反事实标签，学成阶段检测器。W/runs/e53_head_diag/REPORT.md 有全部细节。
- E54：E33-B 预测器对动作敏感（演示预测比 hold 更接近真实未来 99%+）；免训练的“全局池化 token 最近邻示教目标距离”demo>hold 0.77–0.81（非正式口径，脚本 W/runs/e53_head_diag/e54_nn_goal.py、e54_oracle_distance.py）。
- 本任务：把这条思路做成规范的、可复用的评分器，并按正式门槛口径评估。

## 数据与现成代码（只读复用）
- 候选包：W/runs/e48_p3a3/tianji/{fit,dev,eval}.npz（11 候选：demo、hold、5 个 shuffled、reverse、3 个 jitter；h16/h32 预测未来 token；real_future_token；masks；roi）。SHA256 不得改变，运行前后核对。
- 标签与配对：W/runs/e53_head_diag/eval8/pairs_relaxed_{fit,eval}.npz（E49 人工标签 24 集，评估集 8 集 7,18,38,56,64,66,77,88 有标签；含 window_index、target_mask（PID 瓶 mask）、hand_side、label_h16/h32、next_event_h16/h32、mask_valid 等）。标签口径为“放宽口径”（下一事件是否在窗口内；next_next_event_reached 置 False）；该修订已记录，沿用。
- 特征：W/scorer/features.py 的 extract_b1_features（手 ROI token、池化 token、数值几何）；W/scorer/crosswalk_e49_v2.py 的 get_features(bundle,pairs,cfg,h,split) 能得到“目标 mask 已换成 PID mask”的候选侧与真实未来侧特征，可直接调用或复制其中逻辑到新文件。
- 门槛：W/scorer/gates.py（compute_rank_gate 等）、controls.py；候选类型顺序见 bundle 的 candidate_type。

## 要实现的评分器（W/scorer/scorer_v2/，新文件）
打分 S(a) = − d( phi(ẑ_h(a)), g )，d 越小越好；无训练参数，不引入分类头。
### 表示 phi（两种）
- R0（对照）：全局池化 token（复现 E54）。
- R1（主）：物体中心表示：当前窗口手 ROI 4×4 token + 以目标瓶 mask 池化的 token（+ 目标邻域池化，若 B1 特征已提供），拼接后做 L2 归一化；不使用数值几何/FK 项作为 phi（只作后续步骤备用）。
### 距离 d：相对 L2 与余弦两种。
### 目标 g_k 三种，由简到繁（只能使用 fit 集数据；评估窗口的真实未来、评估集标签严禁进入目标）
- G-A 最近邻示教未来：以当前窗口的 phi（当前帧）为键，在 fit 配对中（排除同 episode）取 K 近邻（K∈{1,5,20}；可选同手侧/同下一事件约束两种），目标=这些配对的真实未来 phi 均值。
- G-B 事件原型：对每个事件 k∈{G1,G2,G3,G5} 与手侧，原型=fit 中“事件 k 在窗口内发生”的正例配对的真实未来 phi 的均值（样本少时合并手侧并注明）；窗口的目标=其因果下一事件 k 的原型。
- G-C 任务终态：目标=fit 中“t+h 落在 episode 最后 16 帧内”的窗口真实未来 phi 均值（瓶入箱终态），对所有窗口相同；可选对照：只用本窗口手侧。
## 评估（先写 config.json 预登记，再看评估集）
1. 选型只在 fit 集内做“留一 episode 交叉”：对每个变体（表示×距离×目标×K/约束）计算 demo>hold 与 demo>shuffled（5 个）的 episode 平均胜率，主变体=二者均值最高者；规则写进 config.json 后再运行评估集，评估集只报告一次（每个目标类型各报告其 fit 内最佳变体，另报告全部变体矩阵但标注“未用于选择”）。
2. 评估集（8 集）正式口径 G_rank：用 gates.py 的 compute_rank_gate（平局边界 0.02 归一分差、NaN 留分母、episode 等权、10,000 次 episode bootstrap、固定种子），h32 为主、h16 次；合格窗口=标签“下一事件在窗口内发生”（演示到达下一事件）。比较项：demo>hold、demo>shuffled、demo>reverse；另报告 demo>jitter_05/10/20（非门槛）。
3. 分层报告：按下一事件 G1/G2/G3/G5 分层的胜率（样本少的层标注样本数），以及“有运动 vs 无运动”混杂的对照：demo vs shuffled 的结果单列。
4. 预言机上限（G_action 的不依赖解码器版本）：预测未来到真实未来的距离排序，演示比 hold/shuffled/reverse 更近的比例（用正式口径与 bootstrap）。
5. 回归检查：R0+G-A(K=5) 在评估集上应复现 E54 的 demo>hold≈0.77–0.81（口径略有差异需说明）。
6. 信息性：分数与轨迹内事件进度的 Spearman（按轨迹）；评分时延（K=11 候选，≥200 轮 p95，仅报告）。
## 完成标准
- scorer_v2 代码（GoalDistanceScorer，含 fit(…) 与 score(…) 接口，输出与 gates.py 兼容的分数与 valid 矩阵）、单元测试（构造完美/随机/反向目标的预期数字；评估窗口真实未来与评估标签不进入目标的断言；NaN 与拒评处理；dev_only 标记）。
- W/runs/e55_v2/{config.json,metrics.json,variant_matrix.csv,REPORT.md,LOG_ENTRY.md}；REPORT.md ≤60 行，中文：变体矩阵摘要、主变体选择依据、评估集正式 G_rank、预言机上限、分层与混杂、回归检查、失败/未覆盖项、下一步建议（含对悲观/弃权的接口建议）。
- 不下“通过”结论：评估集 8 集刚好满足最低集数，请如实写区间与样本数；写出该评分器度量的是“动作后果与目标的距离”，不是“优于示教”。

## 红线
- 只写 W/scorer/scorer_v2/ 与 W/runs/e55_v2/；不改 T、W 下任何已有文件与 bundle；不训练分类头；不使用 GPU（CPU，nice -n 10，线程数≤4）；不碰他人和 P1 训练的进程；不改 ~/.ssh/config；时间用 BRT（宿主机 CST − 11 小时）；所有完成标准满足前不要结束。
- SSH：服务器会拒绝短时间内大量新连接。先开自己的持久连接：mkdir -p ~/.ssh/cm; ssh -fN -o BatchMode=yes -o ControlMaster=yes -o ControlPath=$HOME/.ssh/cm/e55.sock -o ControlPersist=4h dengjianfeng-101，之后所有 ssh/scp 都带 -o ControlPath=$HOME/.ssh/cm/e55.sock；被拒绝则等 60 秒重试，最多 5 次。
