# 任务 E46：天机顶视相机—机器人基座标定复核与重标定（sol high；CPU 为主，只读数据，不用腕部相机）

## 背景与矛盾（先读并厘清）
- E22（runs/e22_camera_calibration_20261101/e22_calibration_metrics.json）：投影网格 vs 手 mask IoU 均值 0.069、质心中位误差 145 px（留出 9 集 4333 窗）。E35-vis/E45 因此认为"顶视标定差"。
- E22b-fix（runs/e22b_calib_check_20261102_fix/，diagnosis.md、summary.md、fits.json）：在修正手 mask（仅手、最大连通域、图像侧判左右）与坐标后，offset-only（法兰系手点偏移）拟合，A 臂验证 中位 5.24 px / p90 14.57 px / 98.3% 在 mask 内，B 臂 4.8 / 11.6 px / 99.7%，通过 ≤12px/≥80% 门槛。即：**手点投影可能其实是够用的，145 px 来自旧的网格-手 mask 对比（含前臂、左右互换等问题）**。请核实这两组数字能否相互解释，究竟什么标定好、什么没标好。
- E38 报告：无法由现有相机/基座校准核对 FK 垂直轴，故 G4（抬起）/G6（下放）禁用；E45 要求：手中心 ROI 覆盖率≥95%、投影中位≤8px/p90≤16px 作为未来工程门槛。
- 数据：93 episode、15Hz、顶视裁剪 320×192（FK 投影 640×384 要除 2），q 54 维（A7+B7+左手20+右手20，弧度），mask：data/object_masks_clean_20261102/、runs/e22b_calib_check_20261102_fix/hand_targets/；FK：tools/e22_tianji_kinematics_20261101.py、tools/e28_hand_geometry.py；相机/基座参数 runs/e22_camera_calibration_20261101/e22_camera_params.json、2026-10-15 实测标定（见 e22b 报告）。不用腕部相机。

## 请完成
1. **矛盾厘清**：用同一批 held-out episode 同时算：旧网格-mask IoU、E22b-fix 的手点-mask 距离、以及 5 个手指指尖点/掌点投影到 mask 的距离，并按左右手、arm 姿态分层；说明 145 px 的来源（最可能：前臂并入 mask、mesh 与 mask 定义不同、左右互换、遮挡），给出一张诊断图（overlay 样例 8 窗）。
2. **要验证的几何量**（各给出指标与门槛）：(a) 手掌/五指指尖点投影中位≤8px、p90≤16px、点落在手 mask 内≥90%；(b) 手中心 ROI（以投影掌点为中心的固定大小框）覆盖手 mask 面积≥95%；(c) **垂直轴/高度**：能否用顶视图确认 FK 的竖直方向与尺度（例如利用抬手/放下阶段的图像尺度变化、手 mask 面积变化、已知 box/桌面平面，或左右手与桌面高度约束）；若顶视无法确认，明确说明"高度 proxy 不可用"并建议是否需要其他几何约束（如已知物体尺寸）；(d) 随时间/episode 的稳定性（外参是否漂移，按 episode 报告）。
3. **重标定方案与实现**（若 1 显示需要）：用 fit 的 84 集（不用 9 个留出）自监督拟合：手点偏移（法兰系）+ 可选的相机外参小修正（6 DoF 小扰动，带正则，限制范围）+ 可选的时间偏移（measured q 与图像相差 0–2 帧）；目标函数＝投影点到手 mask 距离（及指尖点到手 mask 边缘）的稳健损失；在 9 集 held-out 上验证；对比"仅偏移"与"偏移+外参"；输出新的参数文件到 runs/e46_recalib_20261104/（不覆盖旧参数文件），并提供 `tools/e46_project.py`（给定 q 与参数返回掌点/指尖/ROI，接口与 e28_hand_geometry 兼容）。
4. 评估对下游的影响：用新旧参数分别计算 E33 predictor 的 hand-geometry 输入（e28_hand_geometry）的差异幅度，以及 E35-vis 中 8 个窗口的叠加结果变化；不重训任何模型。
5. 结论写清：哪些可直接用于 ROI/评分器（G_rank 前置）、哪些仍不可用、需要的额外数据（如一次性在相机视野里放置已知标定板/AprilTag 做 10 分钟采集，由用户决定，不要自行下载或联系机器人）。

## 约束
CPU 线程≤8，不用 GPU；不修改既有代码/参数/runs；新代码放 tools/e46_*.py；colleague-e、colleague-c 严格只读；不连接上位机；不用腕部相机。报告 notes/2026-11-04-e46-recalib.md（中文≤150 行），图表放 runs/e46_recalib_20261104/；追加 EXPERIMENT_LOG.md 与 worklog（BRT=宿主机 CST−11h）。
