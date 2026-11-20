# Worklog

- **2026-11-03 05:41 BRT — E-report:** Read consolidated-report brief and saved experiment evidence; wrote `runs/report_20261103/ANALYSIS_REPORT.md`, CSV tables, source manifest, numeric evidence ledger and validation record. Verified 53 source hashes and report arithmetic; documented historical calibration uncertainty, E27a summary errors, E28 bridge invalidity and unmeasured statistics. Added E-report to experiment log. No trainer outputs changed and no recommended trials executed.
- **2026-11-03 05:47 BRT — Astra root-cause review:** Executed notes/2026-11-03-root-cause-astra.md; wrote runs/report_20261103/ASTRA_ROOT_CAUSE.md with seven-cause evidence table, ranked verdict, separating experiment, five alternatives and rejected approaches. Read-only H20 provenance checks; no training/evaluation or run-output changes. Updated experiment log atomically after rereading; preserved concurrent consolidated report.
- **2026-11-03 06:32 BRT — E29-S3 (luna-standard, CPU):** Built `data/e29_probe_windows_20261103/` and `runs/e29_vjepa_ac_probe_20261103/s3/REPORT.md`. 2,235 eval / 23,558 train windows; eval strata reach/grasp/lift-carry/release = 1932/43/196/64, moving/static = 350/1885. Actual cam_high squash/center loader and dummy 14-D variant self-test passed; structural checks passed for split, stride, missing-label NaNs, bin-mask coverage, and same-phase cross-episode shuffle donors. Updated E29 in the experiment log atomically; no GPU or training used.

- **2026-11-03 06:34 — E29-S2 (luna-standard, CPU):** Read the pinned V-JEPA 2-AC DROID convention and applied relative-rotation XYZ deltas. Added `tools/e29_action_map.py`, seven passing unit tests, and cached 93 episodes / 27,281 frames (default measured q, 26,909 k=4 pairs at 3.75 Hz; commanded q selectable). Mean absolute commanded-vs-measured delta error: 6.59 mm translation components, 1.72° rotation; mean xyz-delta norm 17.3 / 18.5 mm measured / commanded. k=3 commanded processing also succeeded (27,002 pairs at 5 Hz). Wrote `s2/REPORT.md` and `s2/state_action_stats.json`; E29 remains in-progress overall.

- **2026-11-03 06:51 BRT — E29-S1 (luna-standard, GPU H20 6):** Cloned V-JEPA 2 at the recorded commit; downloaded and hashed the sole approved 11.76 GB checkpoint; built the e29 venv by reusing compatible installed packages, with no new package downloads. Final-config strict predictor load and one forward passed in bf16/fp32; peak allocator memory 2.538/5.109 GiB and first-forward latency 0.687/0.644 s. Added loader, official convention/license notes and S1 results under `runs/e29_vjepa_ac_probe_20261103/s1/`. Next: other E29 sub-tasks continue; no Tianji action-faithfulness conclusion from this setup probe.

- **E29-S4 完成记录（2026-11-03T08:09:59.005197-03:00，sol-code）：** 冻结 V-JEPA 2-AC，H20 GPU7，仅训练解耦读出头，固定4000训练候选帧、train内episode验证选checkpoint。完成2235 squash窗口与350 center moving窗口，五动作变体×两只单臂×TF/AR四步、episode bootstrap1000次；主判定 部分。详见 `runs/e29_vjepa_ac_probe_20261103/s4/REPORT.md`、results.json、CSV与完整复现日志。grasp仅43/moving18，不作显著性判断；未下载或系统安装。S2报告已复制到标准目录，原件保留。

- 2026-11-03 BRT — E30 (luna-research / gpt-6-luna max): 按任务书只读扫描 /mnt/workspace/colleague-e，报告写入 runs/e30_colleague-e_scan_20261103/REPORT.md。发现 54D Optical93 joint action/state 契约与目标顺序相符；worldmodel DF job 仍 pending，单样本 loss 不是验证成绩。未运行脚本、未拷贝模型/数据。下一步：确认命令到 residual 的变换与顶视标定；沿用 MuJoCo outcome-model 方向。按用户指示停止全目录 mtime 核验，结果未核实。

- **2026-11-03 10:04 BRT — E31-B（luna-research / gpt-6-luna max）：** 完成 worldmodel-df 深度调研并写入 `runs/e31_dit_wm_survey_20261103/B_dfwm.md`（68 行，含结构/DF/动作契约/数据与训练状态/评估和样例/算力/迁移表/未核实项）；按最新可读状态记录 96/30,000 更新，明确无已找到的预测样例和动作反事实评测。对 colleague-e、colleague-c 目录只读；未运行目标脚本、未拷贝权重/数据/视频、未用 GPU。原子更新实验日志与本 worklog。

- 2026-11-03 10:04 BRT — E31-A (luna-research): 完成 OpenWAM Optical93 调研并写入 `runs/e31_dit_wm_survey_20261103/A_openwam.md`（验证曲线、源码/commit、58D 契约、部署时延与证据边界）；SSH 全程只读检查 `/mnt/workspace/colleague-e`、`/mnt/workspace/colleague-c`，无脚本/GPU/拷贝；原子更新 experiment log。

- 2026-11-03 BRT — E32-T1：完成3,078窗口索引/READY、CONVENTIONS和恢复脚本；指定OpenWAM checkpoint只读权限不足（root:colleague-c 0640，PermissionError），GPU smoke未成功，探针/PCA/cache/baseline待权限恢复；REPORT.md明确未完成，不发布教师READY。

- 2026-11-03 11:34 BRT — E32-T4（sol-code，E32-dinob24-PhaseE-v2）：代码、CPU假教师20步、H20假教师20步、精确恢复与部署验证通过；真实教师20步因T1 checkpoint读取权限等待 READY，未全部验收；正式训练未启动；报告 `runs/e32_jepa_distill_20261103/t4/REPORT.md`；第三方目录只读、未操作他人进程。

- **E32-T3 (2026-11-03 11:40 BRT):** 完成 `tools/e32_eval.py`；FK 与 OpenWAM `state_eef58` 同步核对 93 episodes/27,281 frames，位姿与手部误差为浮点精度内 0；266 eval 窗口 copy/真值自身 JSON+CSV、episode bootstrap、中心 CNN（CPU，8k train/8 epochs）、解码帧/action-swap API 与 smoke tests 均完成。检测器留出中位误差 hand L/R 7.6/9.7 px、bottle 34.0 px、box 5.5 px；详见 `runs/e32_jepa_distill_20261103/t3/REPORT.md`。OpenWAM/T4 预测未生成，故不做横评。


- **E33-A（2026-11-03 22:11 BRT）：** 完成共享256D B/C/D模型与续跑训练代码；各20步GPU4 smoke、7项结构/接口检查、bitwise参数/优化器续跑、PAUSE、fp16 ep10加载及独立部署验证均通过；正式80epoch未启动，launch_e33.sh仅交付；详见 runs/e33_20261104/t_a/REPORT.md。C首步SIGReg编码器梯度占比98.609%，供审核。
- **2026-11-03 22:19 BRT — E33-E:** 新增 E33 评测与对比工具；E32 266 窗口 batch16 回归 max abs diff 6.16e-7（容差 1e-3），B smoke 检查点端到端评测完成；报告 runs/e33_20261104/t_e/REPORT.md。B/C/D run 与 epoch 曲线未就绪，compare 已按任务书跳过并标注。

- **2026-11-04 00:46 BRT — E33_sub-设计评审定稿：** 完成 `notes/2026-11-04-e33sub-design.md`（144行）；核对 ID-WM v1、E33/E32源码及ep30原始评测；定稿联合双时刻残差flow、固定PCA/冻结基座、禁止错配生成目标监督、离线筛选准入与最多三次运行消融/预算。仅设计，不写训练代码、不启动GPU、不接入现役规划器；colleague-e/colleague-c严格只读。设计SHA256=52b47c645618bf3bf3ad2ee39c4ef5f6ab8a26d0ceb751332b7d4e7b0b19cf1c。
- **2026-11-04 04:18 BRT — E33_sub-实现：** 完成新模块及安全启动脚本；R1/R2 CPU smoke 20步和①–⑫均通过，v3 单窗口 E2E完成。ep80 3,078窗正式缓存完成，PCA32 与 PCA64 压缩准入失败，故按设计未启动正式训练；完整记录见 runs/e33sub_20261104/IMPL_REPORT.md。

- **2026-11-04 05:33 BRT — E33_sub v2（luna-max）:** 实施 full240x256-delta-v2，v1 脚本备份到 runs/e33sub_full_20261104/v1_backup/；只读复用旧 cache_ep80 数组。132 dev 表示准入通过，R1/R2 各 20 步 CPU smoke 与①–⑫及新增结构/评分验收通过；主 A 已改为四项逐项相对 hold 误差。dry-run 验证启动脚本在缺 PREPARED、B 未完成、GPU 忙时拒绝。未启动正式训练、未用 GPU；colleague-e 未访问，colleague-c 仅只读调用 Python 解释器。BRT=宿主机 CST−11h。详见 runs/e33sub_full_20261104/IMPL_REPORT.md。

- **2026-11-04 06:22 BRT — E34 实现交付：** 完成 notes/2026-11-04-e34-code.md 的代码、CPU审计与报告；报告 runs/e34_20261104/IMPL_REPORT.md（36行）。
  - CPU提取93 episodes/27,281帧；瓶子/手/箱池化映射自动一致；联系表160段×5帧，fit/dev=120/40。dev无暗瓶和箱边候选，projected-in-bin正例8；G0待人工复核，未创建标记。
  - 契约烟测15/15通过；AC/S/NOZ参数797,711/798,479/797,711；R1冻结B、零gate和同噪声复现CPU烟测通过。启动器和缓存器缺G0时均状态码2拒绝；评测延期，G1–G4未运行；无正式训练或GPU使用。
  - colleague-e未访问；colleague-c只读使用Python环境，没有写入。保护清单12,600项；结束比对发现6项差异（B_s2旧run四项、tools/e33sub_cache.py、tools/e33sub_verify.py），来源待核；E34写入范围不含这些路径，保持原状。

- **2026-11-04 07:53 BRT — E33_sub 终止（用户决定）：** R1/R2（无rec损失，全240x256残差子目标，ep60）266窗评测未通过预登记门槛：generated A=1.169/1.164 vs E33-B 1.160；oracle仅比B好1.3%（53.9 vs 54.6 mm）；R1 视觉cosine 0.960<copy 0.966，R2 0.972。用户决定停止子目标路线并视为该技术不可达；不做 rec-weight A1、Phase2、K8选择。产物保留于 runs/e33sub_full_20261104/（不删除）。

- **2026-11-04 09:49 BRT — E35-vis（CPU，只读评估与渲染）**：执行 `notes/2026-11-04-e35vis-brief.md`，报告见 `runs/e35vis_20261104/REPORT.md`。使用 E33-B ep80（SHA256 `86595f3d0389b92001da2e0fc7863a0d9833f74f08feb17e12925ea6abe28fe7`），CPU 8 线程、266 个 stride-8 窗口/9 个 episode；未训练、未用 GPU。时间审计 `action[t]` 对齐 `q[t+L]`：A/B 臂最佳 L=2 帧（约133 ms），左手 L=1，右手 L=0；臂组偏差接近零。左臂末端第一步对教师命令的位置误差为13.64 mm（hold 14.37 mm），旋转误差2.72°（hold 2.34°）；对齐实测 q[t+2] 后模型误差仍高于 hold（位置12.10/7.41 mm，旋转2.62/1.63°），因此延迟不足以解释全部误差。步32左/右末端位置误差53.4/71.4 mm，hold为81.7/98.1 mm。已生成轨迹、MuJoCo静态姿态、目标/实测曲线、阶段误差图和23秒 MP4（115帧、5 fps）。当前 CPU 固定种子扩散样本与历史 CUDA 样本不完全一致（模型动作均值最大差2.19；hold最大差约9e-6），报告注明该限制。`colleague-e` 未读取；`colleague-c` 仅只读访问机器人描述。
- **2026-11-04 10:41 BRT — E38 pilot S1 (CPU，不标注)：** 冻结20集 manifest（12/4/4）、生成 raw/overlay 联系图各 41 张与 events_rule.csv；G4/G6 因 FK 垂直轴未核验而 disabled，G7/G8 保留 unknown。CPU smoke 8/8 通过；无模型分数/权重、无 GPU、无人工标注。报告 data/e38_subgoals_20261104/v1/STEP1_REPORT.md。

- **2026-11-04 10:49 BRT — E36b 实现交付：** 新增隔离 `tools/e36b_*.py` 与 `runs/e36b_20261104/` 产物。C 复用 E33 hold memory；T 训练/推理使用当前候选 future，冻结 encoder 与按 E36 补齐的 target。CPU 烟测覆盖 detach、t=0/1、C bitwise、冻结梯度、PAUSE/续跑；全通过。H20 GPU7 只做了≤20步临时计时（37.7 s，非正式单元），测得 p95 1.674 s，据此锁定四单元相同步数上限746；预计训练1.665 GPUh，计时/评测预算0.096 GPUh。**未启动正式训练或最终评测。**哈希与细节见 `runs/e36b_20261104/IMPL_REPORT.md` 和 `source_hashes.json`。colleague-e/colleague-c 严格只读，未用腕部相机；BRT=宿主机 CST−11h。

- **2026-11-04 13:30 BRT — E46（CPU）:** 完成93集全帧顶视几何复核、84/9受限拟合验证、五指/ROI/episode分层、8窗叠加与E33输入扰动；复现旧IoU/145px并区分viewer与实测标定路径。采用offset候选；高度proxy不可用，G4/G6仍禁用。中文报告108行：notes/2026-11-04-e46-recalib.md；完整产物runs/e46_recalib_20261104/。数据与colleague-e/colleague-c严格只读，无训练/GPU/腕相机/上位机；BRT=CST−11h。

- **2026-11-04 14:24 BRT — E36b eval-fix：** 完成 `notes/2026-11-04-e36b-eval-fix.md`。原始 T 在 B/B_s2 均4/4256候选、4/266窗非有限；首错为 Euler 0 的 FK `uv` NaN（5点深度≤0.05m）。仅改 `tools/e36b_*.py`，按训练 `act_norm` q0.001/q0.999裁FK输入，并将无效几何显式标记；训练损失和默认行为不变。T_B/T_B_s2 修复后全266窗有限；原始有限262窗口径与全窗结果并列，未删窗。中文结果见 `runs/e36b_20261104/RESULTS.md`（58行）。评测预算307.3→1500秒；累计约0.559 GPUh（旧失败评测按日志估算），本轮诊断/修复评测0.049/0.076 GPUh；无重训、无腕相机，colleague-e/colleague-c只读；BRT按宿主CST−11h。

- **2026-11-04 14:39 BRT -- E47-A data implementation:** Added tools/e47_config.py, e47_data.py, e47_verify_data.py and read-only fit-stats snapshot. CPU smoke passed: 80/4/9 episodes, 2680/132/266 windows, geo66, causal dq/label alignment, slot/IDM, leakage and malformed-index sentinels; CUDA_VISIBLE_DEVICES empty, 2 threads, no training/GPU. Relevant input SHA256 start/end match. No writes to colleague-e/colleague-c; colleague-c used only read-only for model description and existing Python environment. Report: runs/e47_p0p1_20261104/IMPL_REPORT_A.md.


- **E47-C（2026-11-04 14:44 BRT）**：完成 C 所有评测实现及 CPU 烟测。动作互换按 fit 统计重锚残差；mask 12×20/native24×40、empty-union、corrected/legacy 像素损失、按全窗 MSE 聚合 PSNR、paired episode bootstrap、五门 judge 均有实现。12/12 手算/接口桩哨兵通过；见 `runs/e47_p0p1_20261104/eval_smoke.json` 和 37 行报告 `runs/e47_p0p1_20261104/IMPL_REPORT_C.md`。未真实重放 E33-B 266窗或评测 E47 checkpoint，门槛 (a)–(e) 未裁决。CPU only、无训练/GPU/权重下载；缓存定向 E47 runs/cache；colleague-e 未访问，colleague-c 只读；共同保护清单哈希无变化，tools/e36b_*.py 排除。

- **2026-11-04 14:56 BRT — E47-B 编码完成：** 按会话B任务书实现模型/训练接口/B-refit和 `tools/e47_verify_model.py`。CPU smoke 全通过，含真实 fit 窗 E47与B-refit前后向（无优化器 step），B报告 `runs/e47_p0p1_20261104/IMPL_REPORT_B.md`（20行）。总参数/可训参数：E47 188,489,284/43,246,404；B-refit 182,169,012/37,171,892。未使用GPU、未启动训练或下载权重；time profile/GPU显存与正式训练按用户约束未运行。发现A的 `E47Data.__getitem__` 在 `e47_data.py:436` 因 `raw_q` 未初始化抛错，B自有训练包装器从只读数据构建等价输入并补当前mask/IDM pair，未改A文件。官方DINO权重存在，SHA256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`。起止保护源码哈希匹配；colleague-e未访问、colleague-c仅只读解释器，烟测缓存重定向新E47 runs/cache。

- **2026-11-05 03:12 BRT — E47 集成：** 修复数据项当前四槽 mask/IDM 与 B 回退，统一 `run_eval`/Context stats+episode donor 接口，修 pixel legacy N>1 广播；三子校验及真实 E47/B-refit 前后向 smoke 通过（E47 loss 3.0324，B-refit 2.9737，无 optimizer.step）。E33-B ep80 266窗 CPU 重放复现 PSNR 与 legacy obj_h（0.02050560/0.02357990），corrected_obj_h 0.02385345/0.02802114；h32 mask、动作因历史 CUDA RNG/本机无CUDA未达容差，详见 `runs/e47_p0p1_20261104/INTEGRATION_REPORT.md`。GPU计时未跑（无CUDA/NVML telemetry），正式训练未启动；源哈希见 `source_hashes_start_integration.json` 与 `source_hashes_integration_end.json`。`colleague-e`、`colleague-c` 无写入，缓存仅写 E47 runs。

- **2026-11-05 03:23 BRT — E47 GPU 资源复核：** H20 主机 8 卡利用率 97–100%；无卡满足显存空闲≥16 GiB 且利用率≤10%，未运行两臂 time_profile 或 E33-B CUDA 重放，GPU 使用 0.000 GPUh。更新集成报告并将既有 CPU 重放的 h32 mask/action 差值占位符替换成实数；动作偏差与 CPU/CUDA RNG 不同相符，mask 归因待 CUDA 对照。未触碰 GPU 进程、未训练/安装，本轮未访问 colleague-e/colleague-c。详见 `runs/e47_p0p1_20261104/INTEGRATION_REPORT.md`、`gpu_profile.json`。

- **2026-11-05 04:29:56 BRT — E47 GPU共享计时与 E33-B CUDA 收尾：** 主卡 GPU0（选择时空闲 29,741 MiB、利用率 82%）对 E47/B-refit 各跑 20+100 步；GPU3（选择时空闲 97,352 MiB、利用率 0%）复测两臂。E47/GPU0 p50/p95=0.5084/0.5627s；B-refit/GPU0=0.1410/0.1502s；GPU3 复测 E47=0.3270/0.3682s、B-refit=0.1412/0.6103s。均标注为共享卡上界、非独占值；13,440 更新按 p95 外推为 2.101/0.561 GPUh（GPU0）和 1.374/2.278 GPUh（GPU3），与 0.442 s/update 的判定见 runs/e47_p0p1_20261104/INTEGRATION_REPORT.md。五个任务窗口保守计 0.0732 GPUh < 0.3；无 OOM、未调用 train()、未干预他人进程。E33-B ep80 的 266 窗 CUDA 重放（GPU0、device=cuda、batch=16、seed=20261103）四类 h32 mask IoU 和六项 step32 动作指标与历史一致；此前 CPU/CUDA 兼容差异由设备路径解释。产物和采样日志见 runs/e47_p0p1_20261104/。

- **2026-11-05 06:44:07 BRT - E47 P1 formal training resumed:** GPU1 selected at 0% utilization and 95.07 GiB free; existing E21 venv; no system install or foreign-process intervention. Initial seed47 pair paused at steps 46/115 with checkpoints (PAUSE source unknown). Detached setsid/nohup coordinator PID 2754000 resumed them. E47 PID 2754015 and B-refit PID 2754017 each passed at least 20 additional finite-loss updates. See runs/e47_p0p1_20261104/TRAIN_LAUNCH.md. Frozen 80 epochs, weights 4/2/1, and gates (a)-(e) unchanged; profile_pass=false preserved. The 1.65-hour per-unit cap estimates budget stop near 2026-11-05T08:10:15-03:00 BRT, not epoch-80 completion; seed48 chains only after both seed47 units complete. E47 recent s/update exceeded 0.6; its consecutive-update guard remains active.

- **2026-11-05 06:46:57 BRT - Runtime guard update:** E47 stopped at step 83 with SLOW_GUARD_PAUSED after sustained s/update >0.6; loss is finite and it will not be resumed. B-refit continues alone at step 268 with finite loss and recent s/update 0.29227240245090796. Current GPU free 29.72332763671875 GiB, utilization 48%; seed48 remains gated on both seed47 units completing. Estimated B-refit budget stop: 2026-11-05T08:11:33-03:00 BRT.

- **2026-11-05 06:54:10 BRT - B-refit 串行续跑：** E47 保持 SLOW_GUARD_PAUSED（step 90，loss 有限），未重启。B-refit 从 step 268 的有限 checkpoint 单独恢复，PID 2945220、GPU0（启动时 66%、空闲 29.06 GiB）；现已前进到 step 307，loss_finite=True，s/update=0.1593551902915351。seed48 因 E47 未完成而不启动。预计 B-refit 到预算停止约 2026-11-05T08:16:46-03:00 BRT。

- **2026-11-05 07:00:26 BRT - 最终进程纠正：** E47 在 step 83 达到 SLOW_GUARD_PAUSED（s/update 约 1.19，loss 有限）；旧项目 supervisor PID 2925950 错误地把 E47 作为串行 fallback 重启。核对命令和子进程 PID 2935565 后，仅对自有协调器发 SIGTERM，并给 E47 写 PAUSE 使其在 step 98 优雅退出；修正 supervisor 对 SLOW_GUARD_PAUSED 的硬停止处理。未触碰他人进程，E47 不再恢复。B-refit PID 2945220 单独运行于 GPU0，已到 step 445，loss_finite=True，s/update=0.13429821672616526，状态 running；seed48 因 E47 未完成而不启动。B-refit 预计预算停止 2026-11-05T08:17:05-03:00 BRT。


- **2026-11-05 07:11 BRT — E47 P1 checkpoint 修复与续跑：** `tools/e47_train.py` 原子保存完整模型/EMA/optimizer/scheduler/RNG/data_epoch/采样游标/已用GPU秒；SIGTERM/SIGINT/SIGUSR1 与慢速/PAUSE 均在安全步保存。CPU-only tiny-model smoke 捕获 SIGTERM 后恢复；loss1=1.456109642982483，连续/恢复 loss2 均精确为0.5439258217811584。原进程日志0字节且集成后训练器哈希变动，无法归因旧异常的具体符号；可复现根因与控制流断点、改动及哈希见 `runs/e47_p0p1_20261104/CKPT_FIX_REPORT.md`。
  - 现场正式续跑：E47 PID3053966/GPU4，step116，loss有限；B-refit PID2945220/GPU0，step747，loss有限；均超过50步。GPU2复核时100%利用率/约31GiB空闲；E47切到较低负载GPU4，未触碰他人进程。链式 watcher PID3058870仅在 seed47 两臂 complete 后启动 seed48。保留用户确认的1.65 GPUh硬预算；batch16实测optimizer-step时长意味着13,440步全程超预算，详见 `TRAIN_LAUNCH.md`/修复报告。


- **2026-11-05 07:27:06 BRT — E47 P1 状态收尾：** 完整检查点修复与 CPU 保存→SIGTERM→重载续跑烟测记录于 `runs/e47_p0p1_20261104/CKPT_FIX_REPORT.md`。最新 E47 seed47 在 step 136 因 s/update 均值 1.487 超过 0.442 而 `SLOW_GUARD_PAUSED`，loss 有限、checkpoint 已保存，PID 3053966 退出，GPU4；B-refit seed47 PID 2945220/GPU0 仍运行 step 1177，loss 有限。保留硬预算 1.65 GPUh/单元；B-refit 预计约 2026-11-05T08:18:29-03:00 BRT 触及预算，早于当前步速下的完整训练 ETA。nvidia-smi 复核 GPU2 为100%利用率，未碰其他人的进程；seed48 watcher PID 3058870 只在两臂均 complete 时启动。详见 `TRAIN_LAUNCH.md` 与 `EXPERIMENT_LOG.md`。

- **2026-11-05 11:07 BRT — E47 I/O 与计时口径修复：** 追加 runs/e47_p0p1_20261104/IO_FIX_REPORT.md；实测NFS mmap+FK每样本约10ms串行等待，两worker/prefetch2/pinned memory后等待约0.1ms。GPU前向/损失和反向占主耗时；本次20热身+100测量 B-refit/E47 样本更新p50/p95=0.1205/0.1389s、0.3047/0.3379s，16样本步外推仍超1.65 GPUh。状态schema2修正优化步/样本/等待/计算口径并核算日志与checkpoint开销；监督器兼容旧status；同seed CPU张量逐位一致。未启动正式训练、未写入或删除train_*_s47/latest.pt，受保护目录只读，无安装。BRT=CST−11h。


- **2026-11-05T12:40:19-03:00 BRT — E47 / B-refit 真 batch 提速评估：** 完成 batch1/4/8/16 与 fp32/bf16 GPU 粗扫；B-refit 最优 batch16 bf16 精测 p50/p95=0.2169/0.2212s，E47 batch16 bf16=0.5702/0.5869s。各臂实测用量0.0803/0.0843 GPUh，未 OOM；E47粗扫受300s臂上限影响，batch8 fp32只测16步、batch4/1 fp32未测。CPU batch4比较、Dropout/RNG审计、bf16差异、GPUh/预算表和源备份哈希见 `runs/e47_p0p1_20261104/BATCH_SPEEDUP_REPORT.md`。未调用正式训练器、未重启正式训练、checkpoint只读；未写保护目录，未安装。BRT=宿主CST−11h。

- **2026-11-05 12:48 BRT — 默认精度开关修正：** 发现旧训练器 CUDA autocast 默认开启后，已将 amp 未设置的行为改回 bf16；配置 dtype 与显式开关可覆写。修改前备份和哈希记录在 `runs/e47_p0p1_20261104/source_hashes_amp_default_fix.json`；GPU benchmark 的精度均为显式指定，本次改动未改变测量结果或估时。

- **2026-11-05 13:20:36 BRT — E47 P1 batch16/bf16 restart:** 80 epochs, 13,440 steps, effective batch 16; weights 4/2/1, gates (a)-(e), and split 80/4/9 remain unchanged. Both seeds use `micro_batch=16`, bf16. CPU/GPU smoke passed in `/tmp`; old seed47 outputs were `mv`-archived and SHA256-verified. Both arms passed 20 finite-loss steps before >1.5x slowdown triggered serial fallback; external compute PID 3697146 was observed and left untouched. See `runs/e47_p0p1_20261104/TRAIN_LAUNCH.md` for current state and ETA.


- **2026-11-05T14:04:32-03:00 — E47/B-refit GPU2 续跑及协调器安全接管：** 旧协调器 PID3476783 仅收到 SIGTERM；E47 PID3483192/GPU0 在独立 setsid 会话中继续，step 3416→3438，随后推进至 4478，loss有限、0.5767s/step。B-refit PID3850966 从 step75 在 GPU2 续跑，验证新增≥20步且 loss 有限；当前 step 1577、0.2248s/step。E47 并行时相对启动前均值约 1.00x，未超过1.3x；四次卡利用率样本为 GPU0 88/87/100/86%、GPU2 81/63/66/70%，最新 GPU0/GPU2=100%/57%。监督器备份及 SHA、/proc 精确 argv 认领修复、CPU 模拟与干跑见 `runs/e47_p0p1_20261104/TRAIN_LAUNCH.md`。协调器 PID3878891 认领原两 worker，无新 worker；chain_status=`seed47_reclaimed_monitoring`，seed48 仍只在两臂 complete 后启动。ETA（BRT）E47 2026-11-05 15:30:40 BRT、B-refit 2026-11-05 14:48:59 BRT。

- **2026-11-06 00:46:42 BRT — E47 P1 seed48 续跑：** 从 E47 step20 / B-refit step30 的 latest.pt 恢复；CPU torch.load 与 cfg_hash 校验通过。E47 PID1463762 / GPU4 到 step684（+664，0.5868s/step，1.03×）；B-refit PID1463763 / GPU3 到 step1436（+1406，0.2281s/step，1.05×）。两臂有限 loss。GPU4/GPU3 分开，E47 获当时最空 GPU4；GPU7未用。B-refit 有一次1.51×窗口且卡上有外部计算进程，复查已回到1.08×，故未迁卡；若再次持续高于1.5×，安全 checkpoint 暂停后迁到合格低占用卡或暂停等负载下降。协调器 PID1473634 通过 dry-run 后只认领、不重启；详见 `runs/e47_p0p1_20261104/TRAIN_LAUNCH.md`、`launcher/processes.json`。

- **2026-11-06T01:34:06-03:00 — E47 P1 seed47 gates:** Wrote runs/e47_p0p1_20261104/GATES_SEED47_REPORT.md and eval_seed47/analysis.json / gate_diagnostics.json plus 1,105 per-episode gate rows in episode_gate_metrics.csv. Frozen test verdicts for E47: a 14/16, b 6/16, c 9/10, d 8/12, e 24/24; val diagnostic only; B-refit partial endpoints disclose missing native 24x40 and chain rollout. Seed48 GPUs 4/3 were verified from launcher/processes.json; eval used GPU0 only, about 0.10 GPUh. Checkpoint SHA256: E47 12dea1354e24e51ef19aa0efeea448311d5dd208a750d823b25abf6764e61255, B-refit 54ca31d9347b2b10be69d717db80e91729cfe92b10604cc014450f2a3fe8599a. Logged-data evaluation only; no simulator labels, train_*_s47 writes, protected-directory writes or installs. Single seed only; seed48 must supplement.

 
- **2026-11-06 02:55:45 BRT — E47 seed47 bottle/box 只读诊断：** 完成 `runs/e47_p0p1_20261104/diag_bottle_box/DIAG_REPORT.md`，逐项检验面积与空集约定、运动/手物距离分层、损失梯度、解码与 h0、冻结线性探针、标签连续性、逐 episode 差异、阈值和动作敏感性；所有改进项仅建议、未执行。test 仍用冻结协议。GPU3/4 排除后选择 GPU0（执行前0% util、34,310 MiB free），GPU总保守用量0.04788h；未操作其他进程，未用 MuJoCo/模拟器标签，未改训练目录、协议或阈值，未安装软件。checkpoint SHA256、命令、环境和数据在报告及 `diag_bottle_box/` JSON/log。BRT=CST−11h。

- **2026-11-06T04:19:32-03:00 BRT — E47 P1 seed48/two-seed gates:** E47 seed48 test a–e = 14/16, 4/16, 9/10, 9/12, 24/24; seed47 = 14/16, 6/16, 9/10, 8/12, 24/24. protocol.json lists seeds [47, 48] but specifies no cross-seed aggregation; joint gate rule is 协议未规定. Exact seed47 evaluator rerun matches the three requested metrics and full primary JSON/aux arrays. 27 paired metrics, 78 seed-consistency rows, and both-seed bottle/box gate-b rows are preserved in eval_seed48/. Frozen diagnostic strata reproduce static-bottle negative and native-moving-bottle positive results; action sensitivity pattern mostly recurs with bottle h16 demo−hold ratio 0.590. GPU4 H20 selected at 79,697 MiB free / 0%; conservative task usage 0.23972 GPUh <=0.3; no processes touched. Checkpoint SHA256 E47 4bc6caa45eb1451d070713969168cf8111f437f6283e3fd3320d04c987469ddf, B-refit 084e34bcaf39dee94ae86c11355d7f181755854de9b86e4d16e637480ba87200. No train_* or seed47 outputs modified, protected directories not written, no installs. Reports: GATES_SEED48_REPORT.md and GATES_TWO_SEED_REPORT.md. BRT=CST−11h.

- 2026-11-06 05:12:53 BRT：完成 E47 第二轮物体运动只读诊断并追加 EXPERIMENT_LOG.md。报告 runs/e47_p0p1_20261104/diag_motion2/DIAG2_REPORT.md；两 seed 的 fit/test 分层、损失归因、运动误差、手跟随和动作反事实、变化权重离线模拟与 2×2 建议均已写入。GPU 累计 0.11998 GPUh；未训练、未改协议/阈值及受保护目录。

- **2026-11-06 E54 预注册与预检：** 完成 E54 两因素训练代码/配置、CPU 等价性和 DINO 因果变化权重缓存；fit τ=0.2831231356，协议 SHA256 。GPU20步 A2 两次 1.92–2.17 s/步、3.30–3.74×基线且共享 GPU 忙；A1 0.764 s/步。慢速门槛未通过，因此没有启动正式单元；后续需稳定卡上重做 A2 预检。详见 。

- **2026-11-06 E54 预注册与预检：** 完成 E54 两因素训练代码/配置、CPU 等价性和 DINO 因果变化权重缓存；fit τ=0.2831231356，协议 SHA256 ae08760dc1804d60000399bc176ca02d77c815d7f67a664f7998dca6a040a5fa。GPU20步 A2 两次 1.92–2.17 s/步、3.30–3.74×基线且共享 GPU 忙；A1 0.764 s/步。慢速门槛未通过，因此没有启动正式单元；后续需稳定卡上重做 A2 预检。详见 runs/e54_ablation_20261106/TRAIN_LAUNCH.md。

- **2026-11-07 00:48 BRT — E54 2×2 final checkpoint evaluation：** 按预注册 E54 与 E47 原评估器完成 A0–A3 × seed47/48 的 final test(9 episodes/266 windows) + val(4/132) 评估，native/common、gates(a)–(e)、瓶/箱与手均已逐项写入 `runs/e54_ablation_20261106/eval/EVAL_E54_REPORT.md`。A0 两 seed 的 16 条 gate(a)逐窗 win-rate、gate(b)右手 common h32 gain、gate(d)左手 1–32 步 R 与既有 eval 完全一致（差 0）；A2/A3 四个 delta 模型先通过 E54 开关 strict load 与 smoke 推理。test W 主效应 seed47 +0.0121[−0.0039,+0.0283]、seed48 +0.0184[−0.0122,+0.0504]；D 为 +0.0050[−0.0164,+0.0272]、−0.0001[−0.0200,+0.0259]，均未确认有益。护栏出现已确认退化：A1_s47 左手 common IoU 平均下降0.0252；A1_s48 gate(e)为23/24，低于A0的24/24。val 方向独立并列报告。固定 fit 抽样320窗/seed已完成；A0复用 diag_motion2 同规则缓存。累计GPU 0.58647h（≤0.6），仅GPU0；未触碰他人进程或受保护目录。ep10–ep80 的64个checkpoint完整val曲线未在预算内运行，过拟合起点记未确定，报告明确说明。无MuJoCo/模拟器标签，协议 SHA 未变。详见 `runs/e54_ablation_20261106/EXPERIMENT_LOG.md` 与 `eval/`。
- **2026-11-07 01:54 BRT — E54 过拟合起点诊断补曲线：** 在 GPU4 0% util / 79,697 MiB free 条件下，仅运行 A1 seed47/48 的 ep10/40/70 fit 样本（320 窗）及 val（4 episode/132 窗），另评估 ep80 val 并与 final 既有 val 对照。A0 E47 训练目录无 ep*.pt；A2/A3 与未完成 epoch 因 0.25 GPUh 限制未覆盖。fit−val moving bottle IoU gap 两 seed 从最早采样 ep10 已出现并扩大到 ep70；val 采样峰值 seed47=ep10、seed48=ep40，之后未超过峰值。val 的 propose 被省略以控制预算，所以逐 epoch joint RMSE 不可有效计算，报告中如实标为未测。有效 GPU 805.67秒，连同失败 fit 尝试保守计 74秒共 879.67秒=0.2444 GPUh。未用 test、不选 checkpoint、不改协议/已有评估输出；详见 eval/curves/CURVES_REPORT.md 与 A1_overfit_curves.png。

## E60 P0 分层目标探针与编码器对比 — 2026-11-07 07:20 BRT（主机 18:20 CST−11h）

通过 ssh -o BatchMode=yes <gpu-server> 在 /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu 完成 E60 P0。先读设计依据、E54/E47 运动分层协议与数据/模型代码，再冻结 runs/e60_p0_20261107/P0_PROTOCOL.md（SHA256 65bb2cda529b6fb2ae9cb615293cc42762cbf301a79ff8cb9096ad661274a66c），超参按 fit/val 选择后只运行一次 test。确认原始 640×480 Parquet 帧可得，15Hz、93集/27,281帧，rows[96,480)可裁为640×384且与现存192×320帧号逐行完全对应；E1在首批8帧出现非有限输出并停止。复用E54的E0缓存；E2与E4全帧特征完成，E3因本地缺loader/source license/input contract未做。K=8码本仅用fit拟合，seed4705–4708 ARI均1.0；进度η²=0.233、ρ=−0.248。

E2 val macro score最高（0.5808），作为冻结诊断主编码器。P1未过：+128阶段码对copy和时间先验的paired episode CI均跨0，L2/L3 signed macro gap=0.2357，R=0.693，超过冻结0.5线，故报告按协议原句写“远程粗粒度信号在当前表示上学不到”，不进入P1。E1分辨率修正与放弃E2微调均未确认；E4相对E0的L1细节、moving SNR h1/h2及signed fit−test gap变化符合单seed episode-bootstrap换主干条件。结论不代表任务成功率、闭环或世界模型因果效果。

test评分与72条SNR、paired差、criteria、all_metrics均已写盘；p0_test.py在最后test_status序列化处因dict.physical_gpu字段错误退出，保留TEST_FAILURE.json，核对产物后写POSTRUN_AUDIT.json与test_status.json，未重跑test。可核验CUDA事件和已知失败墙钟保守记账约0.496 GPUh；首次Cholesky求解与SAM/Hydra loader初始化失败未留CUDA事件，报告已标注。无OOM、无安装/下载、未触碰他人进程或受保护目录。完整事实、逐encoder×target fit/val/test区间与基线、码本、SNR、判据、权重SHA见 runs/e60_p0_20261107/P0_REPORT.md。
本轮标签仅来自真实观测、已有伪掩码与FK；无MuJoCo/模拟器标签或新增人工标注。

## E60 P0b — incomplete single test pass (2026-11-07)

- Fit/val completed and frozen before test; fixed 80/4/9 split; existing E0/E2/E4 caches; CPU only; no world-model training.
- The sole test pass aborted at the h64 next-event KNN branch with NameError: knn_class is not defined; no test metric rows were committed. Frozen protocol prohibits retry, so no second pass or label reread was performed.
- P0B_REPORT.md contains complete fit/val results, explicit missing test placeholders, failure traceback, and all test-dependent criteria marked 未确认. No task-success, closed-loop, or world-model causal claim.
- Frozen protocol SHA256: f20afad0c2e72edae003d3ba409df600fa40e18b6a40d8b497c650eb098ac26b; test started UTC 2026-11-07T16:30:01.538927+00:00, BRT 2026-11-07T13:30:01.538927+00:00. GPU use: 0.

## 2026-11-08 — E61-A execution stop

E61-A was preregistered and partially executed on <gpu-server>. E1 full raw-frame DINO features completed and passed finite/cache checks. V-JEPA 2 source/checkpoint preflight and A/B fit-val subset extraction completed. The comparison process then OOMed during the V-JEPA A L1 probe; per instruction, GPU analysis stopped immediately without retrying or switching GPUs. No E61 test scores, endpoint-imagination metrics, or world-model training were run. The E1 historical nonfinite cause remains unconfirmed. Full details and artifact hashes are in runs/e61_endpoint_20261108/A_REPORT.md.


- **2026-11-08 02:42 BRT — E61-B 端到端实现：** 新增 tools/e61_config.py、e61_data.py、e61_model.py、e61_train.py、e61_tests.py、e61_eval.py 与 runs/e61_endpoint_20261108/B_impl/ 实现、测试日志、冒烟 checkpoint、launcher 和待审阅草案。E47 seed47 final 回归六项 loss 差为 0；21 个缺键均属新增模块，strict core load 无其他缺失/多余。12/12 开关组合 CPU 前后向有限，t+32 越界 4 窗屏蔽 3 窗，P/cycle 梯度、VICReg 与均值初始化检查通过。最终版四组合各 20 step bf16 smoke，峰值 5.85–5.92 GiB；含梯度修复前审计运行累计 0.08283 分配 GPU 墙钟小时（训练器 CUDA 计时 0.04890 GPUh），未超过 0.4 GPUh。val 单窗 evaluator K=8/K=1 与目标条件接口成功；gate 证据不足全部 inconclusive。完整训练与 test 评测未运行，效果未验证。详见 runs/e61_endpoint_20261108/B_impl/B_IMPL_REPORT.md；协议草案未冻结，launcher dry-run 未启动 worker。


## E63 — 2026-11-08

<!-- E63_REPORT_SHA256:1e8ba774e89a1f208fdbfd89f197eacec60e2e9f06c78410d873f0124a31a91a -->

E63 pipeline finalized: five planned frozen-feature arms, seed 47; pending arms: none. Findings are single-seed observations. Protocol SHA256 1252c6ff197d7d0249811034221708c7c252ab7aaebb959c97ce01043fd5fcd8. Report: /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e63_encoder_e2e_20261108/E63_REPORT.md.


## E63 — 2026-11-08

<!-- E63_REPORT_SHA256:e7efb9e8ad4d1e7e466bf02da68d54979ab7a1f742459a3854c5bc3f623218a0 -->

E63 pipeline finalized: five planned frozen-feature arms, seed 47; pending arms: none. Findings are single-seed observations. Protocol SHA256 1252c6ff197d7d0249811034221708c7c252ab7aaebb959c97ce01043fd5fcd8. Report: /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e63_encoder_e2e_20261108/E63_REPORT.md.

<!-- E64_AUTO_LOG -->
## 2026-11-09 E64

完成 E64 可微候选策略想象训练与开环评估；无 P1 控制组，结果只能作描述性单 seed 观察。详见 runs/e64_imagination_20261109/E64_REPORT.md。

## E65-selection-2026-11-10

- 完成 E65 候选动作选择可行性流水线；报告：`runs/e65_selection_20261110/E65_REPORT.md`。
- 结论：教师有信息，学生学不出来；协议 SHA256：`b0fa8540ba63f012cba321933afde86b19f8ea198f5aa781a802c05916295e5b`。


## E66_E67_BC_WAM_20261110_FINAL

BRT 2026-11-10T07:56:52-03:00。E66 未决。E67 未决，矩阵或配对覆盖不足。N=64 末段仍有改善，存在欠拟合迹象，不能解释为数据量上限。GPUh 2.2395。报告 runs/e66_e67_bc_wam_20261110/E66_REPORT.md 与 E67_REPORT.md。仅探索性开环评估，pending 见报告。

- E68 C 诊断（2026-11-11T00:02:41-03:00 BRT） · 详见 `runs/e68_lang_policy_20261110/C_DIAGNOSIS.md`。token 与动作梯度已连通，训练参数更新正常；语言注意力总份额约 0.3%，动作响应很弱；原始文本与模板存在分布偏移，给定正确模板的动作指标接近原始文本。诊断含重试 0.0654 GPUh，单独记账。

- 下一轮统一 SigLIP2 双塔与动作潜变量对齐方案已写入 notes/2026-11-11-siglip-action-alignment-plan.md，状态为待实施。两臂统一结构与冻结/解冻日程，对比是否加入显式动作对齐。

- E68 D（2026-11-11T00:47:12-03:00） · 已修复 E69 epNNN.npy 缓存导出与消费索引格式差异，93 集审计及 D 真实冒烟通过；后台驱动 1117145、训练 1134268，GPU 3。C 诊断费用已合入 E68 累计预算。

- E70 单种子正式启动，seed6601，S/A 各5400更新；真实冒烟、冻结解冻梯度隔离、五种dev条件与检索诊断及报告通过；成本估算1.37GPUh/8，启动记录位于 runs/e70_siglip_align_20261111/LAUNCH_RECORD.md。

- E68（2026-11-11T01:46:35-03:00 BRT） · 语言条件策略结果见 `runs/e68_lang_policy_20261110/E68_REPORT.md`。

- E70（2026-11-11T02:31:00-03:00） · 单种子统一 SigLIP2 / 动作对齐结果见 runs/e70_siglip_align_20261111/E70_REPORT.md。

- E71 已通过三臂20步真实冒烟和完整dev评估/报告生成；正式驱动PID 3073159，启动BRT 2026-11-11T03:37:40-03:00；预计 2026-11-11T06:12:41.882296-03:00；预算8GPUh。

- E71 2026-11-11T08:31:11-03:00：单种子指令历史/输出锚点/随机起点实验；结果 runs/e71_command_history_20261111/E71_REPORT.md；状态 {'H1': 'complete', 'H2': 'complete', 'R2': 'complete', 'H0': 'complete'}。

- E72 2026-11-11T08:52:47-03:00 单种子成对SigLIP2双塔/独立转移世界模型：runs/e72_transition_jepa_20261111/E72_REPORT.md；{'P0_train': 'complete', 'P1_train': 'complete', 'W0_train': 'complete', 'W0_eval': 'complete', 'W1_train': 'complete', 'W1_eval': 'complete', 'P0_deploy_dev': 'complete', 'P1_deploy_dev': 'complete', 'P1_null_dev': 'complete', 'P1_given_dev': 'complete', 'P1_shuffle_dev': 'complete', 'P1_bank_mean_dev': 'complete', 'P1_W0_dev': 'complete', 'P1_W1_dev': 'complete', 'P0_deploy_holdout': 'complete', 'P1_deploy_holdout': 'complete', 'P1_bank_mean_holdout': 'complete', 'P1_W0_holdout': 'complete', 'P1_W1_holdout': 'complete', 'P0_deploy_historical_test': 'complete', 'P1_deploy_historical_test': 'complete', 'P1_bank_mean_historical_test': 'complete', 'P1_W0_historical_test': 'complete', 'P1_W1_historical_test': 'complete', 'P0_dense_holdout': 'complete', 'P1_dense_holdout': 'complete', 'P0_dense_dev': 'complete', 'P1_dense_dev': 'complete', 'P0_dense_historical_test': 'complete', 'P1_dense_historical_test': 'complete'}

- **2026-11-11T09:40:30-03:00 — E73模型重构完成：** 新增e73_model/data/losses/run；冻结SigLIP2双塔，语言条件flow，动作前缀因果世界预测，低权重未来RGB解码。GPU1完成20步真实fit更新+dev前向，冻结哈希、梯度隔离、因果前缀、检查点模块装配通过。未启动正式训练；decoder_ready/scorer_ready均false，语义选优明确阻断。详见runs/e73_causal_wam_20261111/IMPLEMENTATION.md。

- **2026-11-11T09:51:24-03:00 — E73 RGB gradient target updated:** per-batch detached coefficient targets 5% of primary world-gradient norm. Eight FIT batches: 4.999986%; four DEV batches: 4.999943%. Gradient isolation and frozen-weight hashes passed. World forward/backward median 55.5ms ->108.1ms (visual encoding excluded). No optimizer updates or formal training. See runs/e73_causal_wam_20261111/image_gradient_5pct_verification.json.

- **2026-11-11T10:03:05-03:00 — E73 formal launched:** detached PID3815798, GPU2, seed6601, decoder1500->flow5400->world5400, shared8GPUh cap, automatic eval/report. ETA BRT 2026-11-11T14:03-03:00 to 2026-11-11T16:03-03:00. See runs/e73_causal_wam_20261111/LAUNCH_RECORD.md.

- **2026-11-11T12:28:39-03:00 — E73 formal pipeline complete:** single seed6601, decoder/flow/world sequential, 2.481 GPUh. Report runs/e73_causal_wam_20261111/E73_REPORT.md.


<!-- E74_AUTO_FINAL_d0_stop -->
- **2026-11-11T14:32:22-03:00 — E74 v2 d0_stop：** D0 dev均值 0.470357131，oracle 0.474732357，改善 -0.9302%。累计 0.019814 GPUh；未开主训练。报告 runs/e74_joint_v2_20261112/E74_REPORT.md。

### E75 CPU审计停止 — 2026-11-12T00:01:49-03:00
候选absolute EEF step4600仅保留且77段已评测，但权重PermissionError，合格测试具体版本未绑定。93集command54/q54/frame_index完全等价：fit64/dev4/holdout16分别64/4/16全部在教师训练；frozen9全部在教师验证。teacher_stop，GPU0h，无冒烟/正式启动。报告runs/e75_openwam_teacher_20261112/E75_REPORT.md，逐集表episode_overlap.csv。

- E75 foundation154000真实20更新冒烟通过，FP32零步差0；teacher1.5633s/sample，student缓存后0.3200s/update，预估累计7.7164GPUh。将启动新run正式驱动，先teacher dev gate，未通过即停止；预训练重叠未知按可能污染。详见runs/e75_openwam_teacher_20261112_foundation154000_v1/SMOKE_REPORT.md。

- E75 foundation正式驱动PID 2456322 已启动并核实存活，预算admission通过；当前 running，不继续等待或轮询。启动/续跑记录 runs/e75_openwam_teacher_20261112_foundation154000_v1/LAUNCH_RECORD.md。

- E75 foundation154000 2026-11-12T00:55:50-03:00 teacher_stop；runs/e75_openwam_teacher_20261112_foundation154000_v1/E75_REPORT.md

- E75 foundation用户要求先停止（2026-11-12T01:04:46-03:00）；已核实记录的本任务进程均已退出，无新启动，保留已有产物/断点。请求前状态：teacher_stop。

- 2026-11-12T10:21:09.767671-03:00 E82-L：建立 Piper 标注规范/报告骨架，等待 parent split；仅 train/P1，CPU，未标注评估parent；不得作测试真值/阈值/选型。

- 2026-11-12T10:41:53.657807-03:00 E82-D：Piper parent/split与CPU数据包已交付；P1 6.410h/245parent/109029窗口，P3候选2.611h/150parent，P3质量pending；0GPUh、无训练，test交集0，源stat无变化；见notes/e82_D_piper_data_20261112.md。

- 2026-11-12T13:40:37.566534-03:00 | E82-M | 新建code_e82：29合同测试/真实DINO CPU接入通过（O187/A50/W273，core4.739M）；真实GPU profile无空闲卡阻塞、48GPUh/P99未决；早期CUDA RNG潜在上下文偏差已修复并记保守上界；正式训练未启动。报告 notes/e82_M_model_pipeline_20261112.md。

- 2026-11-12T14:19:46.055864-03:00 | E82-L Piper CPU标注完成：L1 275(274same/1unclear)，L2 115parent(67S/16F/32U，257目标)，L3 20parent/480判断+160过渡；dev/test=0，R1证据不足未通过，L4未做；不得作测试真值/阈值/选型。报告notes/e82_L_piper_labels_20261112.md；labels_codex/READY_labels.json。

- 2026-11-12T14:51:01.018334-03:00 E83-A：5条Piper训练parent Tier A双语长文本试标+自检完成；data/e82_piper_v1/lang_codex/A；CPU-only，继续分层批量，评估集0。

- 2026-11-12T15:07:53.041355-03:00 E83-A部分交付：9 train/P1 parent双语长文本，73过程段/80关键帧；3完整样例、去重/长度统计与断点已写；剩142 P3+76 demo3；CPU-only，GPU0，评估集0；报告notes/e83_A_piper_language_20261112.md。

- 2026-11-12T15:12:00.475222-03:00 E83-B：P1 train减P3候选95 parent；完成94条双语长文本，诊断61号隔离；13条完整分工未录足显式标记；CPU/GPU0，源stat无变化；报告notes/e83_B_piper_language_20261112.md，入口data/e82_piper_v1/lang_codex/B/lang_B.jsonl。

- 2026-11-12T15:33:38.234087-03:00 E84-P e84p_20261112_153338：建立报告/断点，等待F READY_split/READY_fk_gt；未读样本、0 GPUh。

- 2026-11-12T15:45:10.844469-03:00 E83-A续标：登记短片减少阶段/关键帧授权；batch003三条已抽帧待审，完成数仍9。

- 2026-11-12T15:47:51.856441-03:00 E84-F：195/49 parent probe锁定，548 capture/436,116 FK行；CPU-only，图像/目视进行中，报告notes/e84_F_fk_gt_20261112.md。

2026-11-12T15:56:38.774908-03:00 E83-A续标累计12条；本批3条已入库，继续剩余范围。

2026-11-12T15:58:04.974353-03:00 E83-A续标累计13条；本批3条已入库，继续剩余范围。

- 2026-11-12T16:02:33.057985-03:00 E84-F完成：436,116 FK行，5824图，30投影/30HSV/20方向对已目视；READY_fk_gt已交；CPU/GPU0；精确像素及demo6投影未放行。报告notes/e84_F_fk_gt_20261112.md。

- 2026-11-12T16:05:03.340948-03:00 E84-F终审：有效435,404 FK行/5,800图，194/49 parent；18,707静态文件SHA通过，READY_fk_gt可读（像素/HSV限制显式）。

- 2026-11-12T16:37:44.053580-03:00 E84-P e84p_20261112_153338 前置核验交付，正式实验因精确像素真值未批准停止；0GPUh、0真实模型帧；汇总notes/e84_alignment_summary_20261112.md；自有等待进程已停。

2026-11-12T23:35:14.583946-03:00 E83-A续标累计16条；本批已入库，继续剩余范围。

2026-11-12T23:40:36.514912-03:00 E83-A续标累计20条；本批已入库，继续剩余范围。

2026-11-13T00:06:55.990032-03:00 E83-A续标累计73条；本批已入库，继续剩余范围。

2026-11-13T00:15:08.985175-03:00 E83-A续标累计104条；本批已入库，继续剩余范围。

2026-11-13T00:22:17.380847-03:00 E83-A续标累计129条；本批已入库，继续剩余范围。

2026-11-13T00:30:07.105567-03:00 E83-A续标累计152条；本批已入库，继续剩余范围。

2026-11-13T00:44:31.385022-03:00 E83-A续标累计200条；本批已入库，继续剩余范围。

2026-11-13T00:50:29.899163-03:00 E83-A续标累计221条；本批已入库，继续剩余范围。

- 2026-11-13T00:54:38.867998-03:00 E83-A：全部227/227 train/P1 Tier A双语语言标注完成，剩余0；1条左腕不可读显式低置信度例外，短片段按授权缩减；报告 /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e83_A_piper_language_20261113.md。

- 2026-11-13T01:39:33.954120-03:00 BRT E84-G：150帧模型双遍+30帧第三遍，P90重复差5.39px；标定demo4/early/native中位26.90/9.07/18.97px，覆盖不足及误差门槛导致全部NO-GO；报告 notes/e84_G_pixel_gt_20261113.md，READY data/e84_piper_align_v1/READY_pixel_gt.json。

- 2026-11-13T02:17:36.807264-03:00 BRT e84p_20261113_021548：150帧近似GT入口SHA/字段/时间邻对核验完成；GPU 0/8合格，依修订停止，0 GPUh；实验未运行，续跑无需新增授权。汇总 notes/e84_alignment_summary_20261113.md。

- 2026-11-13T04:09:39.291760-03:00 BRT E84-P `e84p_20261113_021548` 共享GPU4续跑完成：22230时刻冻结特征、150帧粗像素诊断、FK运动及位移反事实；GPU 1.2767h/LoRA0；左右不可判定，LoRA等用户确认；汇总 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e84_alignment_summary_20261113.md`（已备份旧版）。

- 2026-11-13T04:36:46.860146-03:00 BRT E82-D2：demo3新增270条P3候选，rev4保持parent分区，P1 559条、P3候选437条，P3_ready=0；验收全通过；详见 e82_D2_demo3_pool_20261113.md。

- 2026-11-13T05:39:52.552221-03:00 BRT `e84lora_20261113_042425`：E83近邻配对不作正式结论，严格仅留11同锚点caption，转 `e84lora_20261113_053731`；先导GPU0.792223h计入4h总预算，旧结果全保留。

- 2026-11-13T06:08:23.503885-03:00 E82-Q：开始437条P3 train候选独立审核；quality_astra；CPU only；RUNNING。

- 2026-11-13T06:52:30.886227-03:00 BRT E84-P `e84lora_20261113_053731`：双塔rank8最后4层Q/V、真实/打乱各768步；原selection全量复提与实验1–3、近参数量冻结头、parent CI/漂移完成；GPU1.7588h，原模型未合并；结论见 `notes/e84_lora_summary_20261113.md`，alignment汇总已备份更新。

- 2026-11-13T07:09:55.422407-03:00 BRT E82-Q：P3 437/437质量审核，183A/65R/189U；严格池116parents/185992帧；QC40三类90% κ=.849906；仅CPU，quality_astra，待用户冻结批准/demo3上限，盲法偏差及E83字段缺失见notes/e82_Q_p3_quality_20261113.md。

- 2026-11-13T08:02:22.703848-03:00 BRT E82-Q2：完成Piper 189 uncertain复核（23 accept/24 reject/142仍不确定）；盲QC76.67% κ0.571；建议V1 206条/1.930306h、V2 207条/1.931528h，未冻结、未训练，待批准。报告notes/e82_Q2_recheck_20261113.md。

- 2026-11-13T12:32:52.436003-03:00 E86-M：新建code_e86，冻结P3 V1（24477窗），首轮38项CPU合同通过；16.55GB特征缓存与≤3GPUh烟雾进行中，无正式训练。

- 2026-11-13T12:57:01.652292-03:00 BRT E82-Q3：87源148候选（78A/16R/54U），demo6救回75轮；建议V3 284条目/34777窗/2.738333h/145parent，QC30/30 κ1（同模型非人工真值），CPU/GPU0；未写P3_ready，待用户采用决定。报告notes/e82_Q3_cycles_20261113.md。

- 2026-11-13T13:16:34.649489-03:00 BRT E82-Q3-V3-ADOPT-20261113T131634：用户维持10轮原判并采用V3；SHA256 beaf61055b61f04064203683e8dc6275eb2bae35810b281fa5853df409802964核验通过，284条目/34777窗/145parent；批准及模型会话交接见/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/quality_cycles_astra/adoption_V3_20261113T131634；未启动正式训练、未写P3_ready。

- 2026-11-13T14:29:36.318199-03:00 E86-M完成：55 passed in 29.24s；rev4 P1/V1 P3可读来源缓存与固定P1/P3/无语言烟雾、48 dev窗、两模式各200次GPU时延；1.993208GPUh<3；无正式训练。见notes/e86_M_v3_smoke_20261113.md。

- 2026-11-13T14:35:32.827162-03:00 E86-M最终交付：55 CPU测试，1.9932GPUh；全池正式单链估算6.79GPUh含余量；可读P1/P3缓存与烟雾/200次双模式时延完成，P1源权限1794窗缺口保留，无正式训练。

- 2026-11-13T15:14:49.625908-03:00 BRT E87-Y-20261113T1447：H20只读17源1503记录→431净片段/389父源/3.391065h；49建议0.281954h、189待核；963路短段通过、3任务各12episode联系表；GPU0，无P1_ready。报告notes/e87_yam_inventory_20261113.md。

- 2026-11-13T15:36:40.668324-03:00 E86-V3-20261113T144520：V3接入/episode142永久排除/增量缓存/73项CPU合同/有界烟雾+dev分层+batch32估算完成；0.385007GPUh≤1.5；无正式训练。见notes/e86_V3_integrate_20261113.md。

- 2026-11-14T00:56:16.891982-03:00 BRT E88-W：v3.1工程完成，108 CPU合同，0.376294GPUh；多时域/残差/链/阶段/三语言/恢复与profile，最长建议+100；夹爪signed基线修复，正式训练未启动。报告notes/e88_W_v31_20261114.md。

- 2026-11-14T00:56:19.723298-03:00 BRT E89-A-20261114T0012_BRT：ABC盘点24,151条/342抽样/1,026路解码通过；严格时间9,833条→8,240通过；S1/S2/S3=50.46/100.25/150.18h，推荐S1+30Hz有界LRU≤60GB；GPU0、无训练/特征缓存。报告notes/e89_abc_inventory_20261114.md；候选未放行，时钟/标注作者/LICENSE缺证。

- 2026-11-14T01:00:40.356899-03:00 BRT E88-W最终复核：0.378883GPUh、108合同；补原dev世界指标（无新标注/特征）；保守正式单链10.79GPUh含余量；FINAL_DELIVERY.json。

- 2026-11-14T03:34:21.339148-03:00 E91：ABC 5h子集英文训练标注启动；116条/5.0543h；纯CPU；data/e91_abc_lang_v1；状态RUNNING。

- 2026-11-14T03:35:20.671969-03:00 E92：yam 49/49英文标注、392帧9.23MB、CPU/GPUh0；报告notes/e92_yam_lang_20261114.md；独立10条复标待授权。

- 2026-11-14T03:42:59.643643-03:00 BRT E90：完整v3.1新包code_e90、CPU133项通过；Piper P1固定5000步流程执行中，步骤2未启动；报告notes/e90_full_v31_p1_20261114.md。

- 2026-11-14T03:58:12.763713-03:00 E91 COMPLETE：ABC 116条/5.054296h/16任务，英文覆盖100%，uncertain64.66%；12条盲复核完整指令50%（核心目标100%），阶段差0；纯CPU/GPUh0；data/e91_abc_lang_v1，报告notes/e91_abc_lang_20261114.md。

- 2026-11-14T04:04:35.575882-03:00 BRT E90完成：133合同、Piper P1 5000步/10ckpt、0.287406GPUh；dev缓存6/51parent、+100仅1窗；results/e90_p1_20261114_01/ckpt1_dev_eval.json；步骤2未启动。

- 2026-11-14T04:05:14.106163-03:00 BRT E90指标补充：dev skill10/25/50约-5.81/-5.54/-5.24%，未胜复制；+100仅1窗；CPU基线固定，不启动步骤2。

- 2026-11-14T04:15:58.883382-03:00 E91b 开始：ABC uncertain CPU补帧复查，128项/75 episodes，报告 notes/e91b_abc_uncertain_20261114.md。

- 2026-11-14 04:22:18 BRT E92b完成 yam49clip密集复判1556帧；uncertain100%保持，改指令20；弱独立复核10/10一致，GPUh0，详见notes/e92b_yam_uncertain_20261114.md。

- 2026-11-14T04:33:25.718149-03:00 E91b 交付lang_abc_v2.jsonl：uncertain128→81，10条总指令更新，1115帧≤24/episode，盲复核13/13一致（同族非真值），CPU/GPUh0；逐段密度限制详见notes/e91b_abc_uncertain_20261114.md。

- 2026-11-14T04:46:03.405680-03:00 E92c：建立独立复核报告；E92b缺10clip清单，等待补充，未读取v1/v2，GPUh=0。

- 2026-11-14T04:58:52.583615-03:00 BRT E93：27/51可评parent、211窗（100=208/26），cache重放差0；v1负skill主要受采样方差惩罚，N16近copy、ridge+6.78/+9.28/+12.70/+22.61%；不训练，0.585266GPUh，详见notes/e93_dev_diag_20261114.md；24parent原合同不可评。

- 2026-11-14T05:04:52.799595-03:00 E91c 完成：12 episodes/60窗口补877帧（761独特新增），复判20项保留uncertain，1条叙述修订，v3=81显式项，盲复核2/2；CPU-only，<300MB；notes/e91c_abc_fill_20261114.md。

- 2026-11-14T05:38:16.628174-03:00 E92c-2完成：yam独立10clip复核，4一致/6部分一致，278帧，GPUh=0；见 notes/e92c2_yam_review_20261114.md。

- 2026-11-14T05:45:15.078595-03:00 BRT E94a：149合同通过，READY_readout v1冻结；M1/M2固定5000步与E93统一评估进行中，见notes/e94a_readout_20261114.md。

- 2026-11-14 BRT E94b：DINOv3目标对照已建独立目录/报告；2parent数值自检通过，27.27GB估算通过；并行CPU解码+单GPU编码进行中，预算4GPUh含失败；详见 notes/e94b_dinov3_20261114.md。

- 2026-11-14T06:31:21.965323-03:00 BRT E94a完成：149合同、M1/M2各5000步，修复标准通过=False；0.721495GPUh/2.1064GB，见notes/e94a_readout_20261114.md。
- 2026-11-14T06:39 BRT 用户决定：yam 6 clip 排除出需插入目标/成功标签的子池（粗粒度 P1 文本池保留），文件 data/e92_yam_lang_v1/exclusions_success_subpool_20261114.json（Claude 写入）。

- 2026-11-14T06:57:14.346679-03:00 BRT E95完成：世界头只读结构审计+CPU诊断，GPUh=0；rank不能解释均值缺口，优先空间/同位置通路，FM与链分开验收；见notes/e95_worldhead_structure_astra_20261114.md。

- 2026-11-14T07:11:26.423266-03:00 BRT E96a 开始：P1 current空间旁路单因素，5000步；报告 notes/e96a_spatial_bypass_20261114.md。

- 2026-11-14T07:31:59.068691-03:00 BRT E94b COMPLETE：DINO缓存/训练统计/2×4与7×7 ridge、b1/b2(M1)各5000步及固定评估完成；b2 dev skill=1.75/5.40/10.69/26.16%，非整体优于Sig M1；1.7138GPUh、保守累计26.556GB；报告 notes/e94b_dinov3_20261114.md。

- 2026-11-14T07:32:02.316157-03:00 BRT E96a 完成：195合同、Piper5000步，dev+10对M1 +3.269 [+2.518, +4.011]pp，预注册通过；GPUh=0.256187；报告notes/e96a_spatial_bypass_20261114.md。
- 2026-11-14T13:45 BRT 用户确认删除 pi05 v8 DeepSpeed 中间状态：集群 0 GPU 任务 dengjianfeng-cleanup-20261114163957（root）删除 92 个 step 目录，保留 step_001500/008000/050000 与 latest；脚本 cluster_checks/cleanup_pi05_deepspeed_20261115.sh（Claude 提交）。

- 2026-11-14T13:47:43.302999-03:00 BRT E97-B1完成，10阶段/21谓词/24转移，226 train parent的1013段中995映射、18未映射，B2建议20 dev parent，0 GPUh，待人工审定与标注，见notes/e97b_stages_predicates_astra_20261114.md。

- 2026-11-14T13:53:37.942559-03:00 BRT E97-A1 完成Piper状态/FK覆盖与R0/R1/R2分解；R2 skill8.10/15.00/22.74/39.44%，E96a状态span外仍0.69–2.81%skill；GPU失败准入0.000344692h，未训练；报告notes/e97a_robot_env_decomp_20261114.md。

- 2026-11-14T14:10:57.733373-03:00 E98-L-train：冻结60个train parent（20/24/8/8），完成三视角粗抽帧，主标与6-parent盲复标进行中；data/e98_predicate_labels_train；GPUh=0。

- 2026-11-14T14:11:18.524560-03:00 BRT E98-G 完成：label_source=model(gpt-6-astra)，非人工；10阶段/21谓词通用化，ABC2484段扩展覆盖78.99%，yam49clip/196叙述39.80%，GPU0h；data/e98_generalization_v1；notes/e98_stage_generalization_astra_20261114.md。

- 2026-11-14T14:13:51.764252-03:00 BRT | E98-W | 实现oracle轨迹条件A/B、5折parent R1、合同通过；训练中，预算≤2GPUh/8GB；见 notes/e98w_robot_conditioned_20261114.md。

2026-11-14T14:15:06.814095-03:00 E98-L-dev：初始化20 dev模型阶段/谓词标注，源20/20及60视频SHA核验；首个parent3段126条，CPU/GPUh=0，IN_PROGRESS，报告 notes/e98_dev_labels_20261114.md。

- 2026-11-14T15:21:37.997469-03:00 BRT | E98-W | completed；GPU 1.076348h，新增≤1.669115GB，冻结文件变化0；报告 notes/e98w_robot_conditioned_20261114.md。

- 2026-11-14T15:29:17.231353-03:00 BRT | E98-W | completed；GPU 1.076348h，新增≤1.760328GB，冻结文件变化0；报告 notes/e98w_robot_conditioned_20261114.md。

- 2026-11-14T15:30:36.593060-03:00 BRT | E98-W最终补审计 | completed；GPU上界1.158522h、新增≤1.760381GB；5000步h布局/冻结映射一致；报告 notes/e98w_robot_conditioned_20261114.md。

- 2026-11-14T15:35:31.451271-03:00 BRT | E98-W completed | A轨迹条件有效，B显式分离无额外收益；18个dev数值无效输出CPU恢复、训练forward逐位不变；GPU上界1.159h，新增≤1.77GB；有效指标recovery_02；报告notes/e98w_robot_conditioned_20261114.md。

2026-11-14T15:42:16.320389-03:00 E98-L-dev COMPLETE，20 parent/367阶段/27600谓词，CPU/GPUh0，label_source=model(gpt-6-astra)，非人工；notes/e98_dev_labels_20261114.md。
<!-- E98 complete assembled_final_v2:cafa4ec8f838259ba6e837b70bd508d02a812ab4c0f2348bec4f908368639400 -->

2026-11-14T16:35:15.672531-03:00 E98-L-train RETROSPECTIVE_DELIVERED_PROTOCOL_INCOMPLETE，60 train parent/789阶段/9956谓词，6独立复核，四类一致74.36%，CPU/GPUh0，label_source=model(gpt-6-astra)，非人工；因果盲看及候选下一阶段完整覆盖待补，notes/e98_train_labels_20261114.md。

- 2026-11-15T00:54:54.177126-03:00 BRT E99-W2：P3完整v3.1计划轨迹条件，进行中；notes/e99w2_planned_traj_20261115.md。

- 2026-11-15T01:02:36.382996-03:00 BRT E99-B3 启动：四方法谓词+B4，独立code_e99b3/results/e99b3_20261115_01；基于模型标注的dev真值，60train内48/12划分，dev一次。

- 2026-11-15T01:20:44.191811-03:00 BRT E99-W2完成：5000步，部署条件机械判定=False，GPUh≤0.349835；P3冻结dev缓存覆盖受限，详见notes/e99w2_planned_traj_20261115.md。

- 2026-11-15T01:25:19.819762-03:00 BRT E99-W2最终补全：原48窗/6parent四时域齐全，只读E93缓存；N1预注册未通过，GPUh≤0.364284；报告末尾为最终有效口径。

- 2026-11-15T01:43:07.805190-03:00 BRT E99-B3 COMPLETE：四法+B4/4074因果dev，基于模型标注的dev真值；GPU 0.676641h，新增≤4.369GB；报告 notes/e99b3_predicates_20261115.md。

- 2026-11-15T01:45:54.567393-03:00 BRT E99-B3审计补充：评分一次；dev首64帧有2次完整A因果合同前向例外，原分数不变，详见最终报告。

- 2026-11-15T02:26:36.851901-03:00 BRT E100-C：每组seed90，C0/C1/C2同初始化5000步；CPU开环隔离测试通过，准备GPU训练。

- 2026-11-15T02:50:07.412053-03:00 BRT E100-C完成：三组单seed90×5000，采用C2；GPUh≤0.988112，报告e100c_chain_fix_20261115.md。

- 2026-11-15T03:51:32-0300 BRT 用户决定（Claude记录）：链式约束默认采用 E100-C C2（删除一致性gap、预测重入严格开环、只保留第二跳端点监督）；不做第二个seed复核。当前默认基线ckpt=results/e100c_c2_s90_20261115_03/ckpt_05000.pt (sha256前16位 a432001ffeb7d99f)，代码 code_e100c。

- 2026-11-15 BRT E103-D：已建报告骨架和隔离目录，CPU数据/时间合同审计及缓存复现准备中；C2基线只读，GPU≤6h/新增≤40GB。

- 2026-11-15T04:14:26.825528-03:00 BRT E101-A 完成Piper C2动作诊断；关节历史岭回归胜复制，夹爪L1仍劣；无正式训练；GPU 0.148378h；notes/e101a_action_diag_20261115.md。

- 2026-11-15T04:16:02.231274-03:00 BRT | E101-S 完成（已披露限制）：S2+b因果HMM，dev acc63.56%/macroF1 .558/recall@30 36.60% [27.69,43.29]，未达0.7；60parent五折、仅12全帧训练轨迹调参；dev20一次+bootstrap2000；GPU0h，新增<0.12GB；报告notes/e101s_stage_controller_20261115.md，无需决策。

- 2026-11-15T04:41:30-0300 BRT 用户决定（Claude记录）：(1) Piper 夹爪只建模开/合两态，不回归开度；夹爪评估改为状态准确率/开合事件F1/时机误差。(2) 阶段谓词打分须输入最近若干帧的因果历史，不只看当前帧。

- 2026-11-15T04:43:45.073348-03:00 BRT E102-A 开始：Piper动作头三组修正；报告 notes/e102a_action_fix_20261115.md；状态PREPARING。

- 2026-11-15T04:44:17.238310-03:00 E102-S 启动：因果历史谓词、全60训练轨迹袋外与S2；run=e102s_20261115_01，dev未开始。

- 2026-11-15T04:54:16.137143-03:00 BRT E102-A failed；GPUh上界=0.11957614279261583；报告 notes/e102a_action_fix_20261115.md。

- 2026-11-15T04:57:33.708658-03:00 BRT E102-A failed；GPUh上界=0.18907656645492857；报告 notes/e102a_action_fix_20261115.md。

- 2026-11-15T05:20:24.939578-03:00 BRT E102-A complete；GPUh上界=1.1129075713044343；报告 notes/e102a_action_fix_20261115.md。

- 2026-11-15T05:21:57.339249-03:00 BRT E102-A 最终审计完成：run03每组2000步；A2关节通过/夹爪显著性未通过/世界条件通过；GPU 1.112908h含失败；报告 notes/e102a_action_fix_20261115.md。

- 2026-11-15T05:32:29.221411-03:00 E102-S COMPLETE：H0-H3因果历史与S2双初态，60训练五折/20dev一次；详见e102s_temporal_predicates_20261115.md。

- 2026-11-15T05:33:21-0300 BRT 用户决定（Claude记录）：默认基线改为 E102-A A2（夹爪两态+动作路径因果本体历史+确定性Δq头，C2链）；ckpt=results/e102a_a2_s90_20261115_03/ckpt_02000.pt (sha256前16位 45e4118ffec7ad6c)，代码 code_e102a。随后做27 dev parent扩展评估（E102-X）。

- 2026-11-15T05:34:42.021088-03:00 E102-S交付核验：预注册未通过，H0/H1/H2/H3 dev AUROC=.7947/.7724/.6926/.7564；训练预选初态S2 R@30=.4150/.3314/.2651/.1931；GPU .4769h，新增约1.08GB，无遗留执行项。

- 2026-11-15T05:38:34.310026-03:00 BRT E102-X：已锁定E93 211窗/27parent，只读扩展评估；code_e102x，results/e102x_extended_s90_20261115_01，报告notes/e102x_extended_eval_20261115.md。

- 2026-11-15T05:43:03.319280-03:00 BRT E102-X COMPLETE：27parent/211窗三组扩展评估，+100 26parent/208窗；GPUh≤0.074467，结果results/e102x_extended_s90_20261115_01，报告notes/e102x_extended_eval_20261115.md，判定{"joint_by_group": {"A0": false, "A1": false, "A2": true}, "world_by_group": {"A0": true, "A1": true, "A2": true}, "gripper_by_group": {"A1": true, "A2": true}, "gripper_fix_effective": true}。

- 2026-11-15T05:44:19.978218-03:00 BRT E102-X审计：49事件/48去重，F1±10配对CI均正；A2 GT世界下降仍显著；原48完全重放，4GPU段closed，约8.89MiB。

- 2026-11-15T06:14:53.830974-03:00 BRT E103-D COMPLETE：无增益；GPU 2.0128h（含失败/中断段），新增9.863GB；D0/D1各5000步。；有效run02，报告notes/e103d_external_data_20261115.md。

- 2026-11-15T12:37:59.060888-03:00 E98-C：启动60训练parent因果盲标，4410时刻/92610计划谓词，GPU0；独立目录data/e98_predicate_labels_train_causal_v1，进行中；报告notes/e98c_train_causal_labels_20261115.md。

- 2026-11-15T12:59:34.273108-03:00 BRT E104-P PREPARING：新路径检查通过，报告骨架已建；预算3GPUh/10GB；预注册CPU核验后锁定，尚无本轮评分。；报告notes/e104p_pretrain_finetune_20261115.md。

- 2026-11-15T13:01:31.307586-03:00 BRT E104-P STOPPED：必需项未全部完成，详见PIPELINE.json；不作增益结论。；报告notes/e104p_pretrain_finetune_20261115.md。

- 2026-11-15T13:02:42.546738-03:00 BRT E104-P PREPARING：新路径检查通过，报告骨架已建；预算3GPUh/10GB；预注册CPU核验后锁定，尚无本轮评分。；报告notes/e104p_pretrain_finetune_20261115.md。

- 2026-11-15T14:09:45.652691-03:00 BRT E104-P COMPLETE：未满足外部预训练有效判据；GPUh上界1.7811；新增1.131GB；全部固定终点训练/评估完成。；报告notes/e104p_pretrain_finetune_20261115.md。

- 2026-11-15T14:11:20.107801-03:00 BRT E104-P AUDIT_PASSED：固定终点全流程完成；连续AdamW/外部冻结状态/Piper成对流/A2重放/输入源码只读审计通过；GPUh上界1.781113，新增1.1314GB；有效主调度e104p_parallel.py。；报告notes/e104p_pretrain_finetune_20261115.md。

- 2026-11-15T14:14:21.687587-03:00 BRT E104-P COMPLETE：F1−F0 no_traj +25/+50/+100=-2.198,-1.949,-0.769pp；F1−F2=-3.180,-3.491,-2.001pp，全部95%CI上界<0；gt非劣性失败，动作/夹爪5%非劣性通过。F2−F0=+0.982,+1.542,+1.232pp（+100 CI跨0）。GPUh≤1.781113，新增≤1.15GB。完整固定终点和CPU审计通过；未完成/待决无。报告notes/e104p_pretrain_finetune_20261115.md；有效control/run03，主调度e104p_parallel.py。

- 2026-11-15T14:39:12.509116-03:00 E105-Y COMPLETE：yam43段/879.03秒，保守2 parent，固定344/344/344/335窗；不足正式三方比较，GPU0；见notes/e105y_yam_feasibility_20261115.md；结果results/e105y_audit_20261115_142944。

- E106-C | 20261115 BRT | PREPARING | F2续训10000→20000与五点收敛；code_e106c / results/e106c_*；命令与结论见 notes/e106c_longer_training_20261115.md。

- E106-C | 2026-11-15T15:58:16.789215-03:00 BRT | COMPLETE | F2同流续至20000，15000/20000固定ckpt及五点曲线；code_e106c / results/e106c_control_20261115_02；GPUh=1.087853315815252；新字节=589360359；默认A2未改；详见 notes/e106c_longer_training_20261115.md。

- E106-C | 2026-11-15T15:58:43.274924-03:00 BRT | COMPLETE | 目标：Piper F2同采样流续训10000→20000并画收敛曲线；改动：新code_e106c/run02，恢复完整AdamW并重放前10000步sampler哈希，固定15000/20000保存；命令：nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -B -u code_e106c/run02/e106c_pipeline.py（start_new_session=True）+CPU postprocess_delivery.py；目录：results/e106c_control_20261115_02、results/e106c_f2_s90_20261115_02（run01失败保留）；结果：五点/27parent+48窗、bootstrap2000 seed90、全部指标与配对95%CI图表已完成；结论详见notes/e106c_longer_training_20261115.md，默认A2不变；资源GPUh上界=1.087853315815252，含失败；新增总量保守<641879565字节。

- 2026-11-15T16:06:57.945310-03:00 E98-C代理模型切换完成：a/b/c均在完整parent边界切至gpt-6-luna high新代理；旧sol锁不改。审计2149/4410时刻、45129谓词、0错误；A分片727–730因模板复用已taint并待盲复标。

- 2026-11-15 BRT 用户决定：默认模型改为 E106-C 第20000步 results/e106c_f2_s90_20261115_02/ckpt_20000.pt（sha dd27fe05c034…，代码 code_e106c）。取代 A2(ckpt_02000)。依据：世界头不给轨迹/给动作、关节全面优于 A2，A2 的 gt_action 掉落已追平 A0；夹爪事件 F1 无稳定趋势（仍是弱项）。

- 2026-11-16T00:33:54.745948-03:00 BRT E107-P：完成默认20000头收缩/多峰代理诊断；方差比0.724–0.857，混合11.1–13.9%但逐爪中间超额未显著；严格语义阶段/人工谓词真值缺失；暂缓完整E107-M；GPU 0.021338h，新增215.1MiB；e107p_multimodal_diagnosis_20261116.md。

- 2026-11-16T00:59:27.137030-03:00 BRT E107-M PRECHECK：新目录已创建；M0/M1固定3000步，CPU测试前置；报告 notes/e107m_multimodal_head_20261116.md。

- 2026-11-16T01:43:55.545779-03:00 BRT E107-M COMPLETE：M0/M1固定3000步，主27parent/附48窗，结论=False，GPUh=0.6038956917426549，报告 notes/e107m_multimodal_head_20261116.md。

- 2026-11-16T01:45:18.056464-03:00 BRT E107-M 交付终审通过；训练run01+评估run03；最终GPUh上界=0.6288431538309023（含压缩尾段与全部失败），新增保守字节=1733924970；FINAL_AUDIT.json/GPU_PROCESS_ACCOUNTING.json；报告notes/e107m_multimodal_head_20261116.md。

- 2026-11-16T02:49:39.835488-03:00 BRT E108-G COMPLETE：夹爪G0/G1，基线F1@10=.245679、FP/min=8.5；四家族因果解码均未达预注册；GPUh=0，新增≈36.13MiB；报告 notes/e108g_gripper_diagnosis_20261116.md。

- 2026-11-16T04:32:59.275171-03:00 BRT E109-G COMPLETE：夹爪train/dev诊断，F1@10 .634891→对齐.460969 / dev.245679；协议占差44.69%，剩余差.215290；GPUh=0、新增保守≤64,447,551 bytes；报告 notes/e109g_gripper_generalization_20261116.md。
- 2026-11-16T05:53:53-03:00 BRT E110-F COMPLETE：D/F公平对照完成；GPU=0.4663h（含失败）、新增425746772B；主集S4单样本未满足预注册非劣且多样性塌缩，默认确定性头不变；报告 notes/e110f_flow_vs_deterministic_20261116.md。
2026-11-16 BRT — E111-F：核对 E110-F，D/F 均继承 A2 185/185 参数且 delta_head optimizer step 26k；flow 速度目标仅训练4k，判为目标曝光不对等。新增续训脚本但服务器 Python 缺 NumPy/PyTorch，F05 又未保存 optimizer/sampler state，确定性恢复单测无法运行；GPU 0h、未续训/未评估。详见 notes/e111f_flow_trainlength_20261116.md。

[2026-11-16 BRT] E112-S：启动 runs/results/data 只读存储盘点；报告 notes/e112s_storage_inventory_20261116.md，清单 results/e112s_inventory_20261116；无清理操作，进行中。

- 2026-11-16T09:37:33-03:00 BRT — E111-F COMPLETE（接续）：完成 F05 确定性重放逐位核验、16k flow 目标追加和4个固定点复评；目标累计20k vs D 26k；最终 S4 单样本关节非劣未过（相对差上界@8 +6.25%、@50 +9.14%），spread-skill .278/塌缩87.96%，不替换默认D。GPU 1.8343h，新增约1.12GB；详见 notes/e111f_flow_trainlength_20261116.md。
- 2026-11-16 BRT — E112-S：完成 runs/results/data 只读盘点。runs 563/745.81GiB，results 92/19.54GiB，data 39 项中 36 项 du 完成/93.44GiB；E98-C 活跃跳过，2 个 data 目录超时。A230、B14、C0、D451；9 个待核 cache 合计约2.85GiB。报告 notes/e112s_storage_inventory_20261116.md，逐目录表 results/e112s_inventory_20261116/；无清理，待用户逐类批准。
2026-11-16T14:22:34.772958-03:00 BRT E98-C COMPLETE：因果首遍4410时刻/92610谓词通过严格审计；回顾对照9956条，独立复标441时刻（Sol/Luna分层），源身份复核180视频；GPUh=0、新增约1.05GB。非人工标签仅供训练，23个疑点时刻隔离；E102-S2待后续。详见 notes/e98c_train_causal_labels_20261115.md。
- 2026-11-16 BRT — E112-S 后续：按用户指示删除 E28 下 72 个 value/features/*.npy 特征数组（约 96.333GiB），保留 checkpoints/评估结果/报告/status；du 从 102.705 降至 6.372GiB。其他 runs ≥10GiB 清单基于 E112 盘点，未再遍历或清理。
- 2026-11-16 BRT — E112-S 按用户要求精简六个 runs 目录：删122个 .npy/.npz（146.054GiB）及129个旧ckpt（44.558GiB），46个训练子目录各留最新mtime ckpt；251个删除目标核验不存在。清理后六目录约11.735/0.001/7.545/14.832/10.868/0.000GiB，详见E112-S报告及selected_six_cleanup_delete_manifest.tsv。

- 2026-11-16 BRT E102-S2/E113-V COMPLETE（Claude直接执行）：因果全量标签使谓词宏AUROC .795→.908(CF_H0)，但阶段recall@30未改善(.415→.346)，历史无收益；VLM多帧+具体提示 .629→.683，仍低于SigLIP2线性头.795。见 notes/e102s2_e113v_20261116.md。

- 2026-11-16 BRT E102-S2b COMPLETE：HMM发射用因果阶段标签重拟合，recall@30 CS_H0 .326（vs CF .346，差CI含0；vs 原E102-S .415更低）；因果/回顾阶段同帧仅50%一致；评估口径仍为回顾式，见 notes/e102s2_e113v_20261116.md。

- 2026-11-16 BRT E114-H：人工阶段复核页面（code_e114h，端口18795，输出 results/e114h_human_stage_20261116），模式①盲标边界②盲复核E98-C因果阶段；已自测，测试数据已清除，尚无人工标注。

- 2026-11-17 BRT E114-H 页面修复：旧版在有未保存修改时用 confirm 阻止切换片段（嵌入浏览器里 confirm 无法确认，导致只能反复保存同一片段，v7），改为自动存草稿后直接切换；Ctrl+S/下拉焦点/草稿自动保存。dev 人工标注仅 1/20。

- 2026-11-17 BRT 放宽容差评估：±90帧下阶段边界召回 R_H0 .726、E101 .709、CF .683；历史H2/H3仍显著更差、H1≈H0；因果标签/因果阶段拟合无提升。人工4片段：模型边界±90帧内覆盖80%人工边界。见 notes/e102s2_e113v_20261116.md。

- 2026-11-17 BRT E115-P1：阶段条件动作头（S0/S1/S2，各4000步）在160窗/20 parent上与无阶段对照无差异（含真实阶段上界），不采用；见 notes/policy_assembly_design_20261117.md。

- 2026-11-17 BRT E115-P2：预测token上的谓词/阶段进展信息 > 复制当前（+50/+100显著），真实动作>预测动作>无轨迹>错配动作；见 notes/policy_assembly_design_20261117.md。

- 2026-11-17 BRT E116-P3：纯推理语言-图像对齐未过门槛（阶段top-1 .036 [.030,.041]，谓词AUROC .520 [.486,.584]）；监督CF H0阶段top-1 .621；GPU 0.0199h，新增落盘<150MiB。报告 notes/e116p3_language_alignment_2026-11-17.md。
- 2026-11-17 BRT E117-D：完成学习式语言对齐设计报告 notes/e117_learned_alignment_design_2026-11-17.md；比较A/B/C/D，推荐先冻结SigLIP2双塔小投影，规定parent/描述族持出、噪声处理、L1-L3判据和≤0.1 GPUh路线；未训练，GPUh=0，未下载模型。详情见docs/experiment_log/EXPERIMENT_LOG.md。
2026-11-17 BRT | E118 | 启动文字↔未来图像/动作差分对齐小试点；报告骨架与判据已落盘，源代码/训练环境核查中。
2026-11-17 BRT | E118 | 完成 FK 标签来源/缓存粒度审计；未训练或评估，NO-GO（证据不足），详见 notes/e118_guidance_alignment_pilot_2026-11-17.md。
2026-11-17 BRT | E118状态更正 | 本轮仅源码/收据审计，未训练评估（GPUh=0）；NO-GO=证据不足，见 notes/e118_guidance_alignment_pilot_2026-11-17.md。

- 2026-11-17 BRT E118（Claude执行）：FK运动文字↔图像差分对齐在留出parent上可学习但效应小（粗类平衡top1 dI .105 vs cur .083，差+.021 [.017,.026]）；动作差分.283属平凡；见 notes/e118_guidance_alignment_pilot_claude_20261117.md。

- 2026-11-17 BRT E119（Claude执行）：语言条件预测：观测丢弃70%后文字显著改善被描述手臂dA(R2 .543→.667)与dI(+.03)，错配文字更差，反事实可控；但改写句泛化失败；无丢弃时模型忽略语言。见 notes/e119_language_conditioned_prediction_20261117.md。

- 2026-11-17 BRT E120（Claude执行）：联合空间先验检验：夹爪闭合/张开ΔV对ΔT方向一致(close vs open AUROC .665，弱但符号翻转)；MaskCLIP桥使橙色/白色/黑色逐patch定位AUROC .65–.93，原始隐藏状态<0.5(反向错位)；见 notes/e120_jointspace_priors_20261117.md。

- 2026-11-18 BRT E121（Claude执行）：桥后联合空间特征不加丢弃时语言利用与原特征几乎相同(MLP +.023 vs +.021；注意力→文字≈.08<均匀.2)，丢弃才是关键(+.11~.14)；惰性主因是视觉捷径。见 notes/e121_bridge_modal_inertia_20261117.md。

- 2026-11-18 BRT E122（Claude执行）：P1边界窗口重分析(阶段无收益，S2略差)；历史丢弃探针：只丢历史使语言增益由.0235升至.040、可控.028→.042，远低于全特征丢弃(.117/.142)，代价小(R²−.017)；见 notes/e122_history_dropout_and_p1_boundary_20261118.md。

- 2026-11-18 BRT jerk平滑度检验：S2/S1 与 S0 的预测加速度/jerk 能量相同(差<1%且区间含0)，阶跃扰动机理不被支持；所有预测比真值平滑约2-3.5倍，切换窗口也不更尖锐。见 notes/e122_history_dropout_and_p1_boundary_20261118.md。

- 2026-11-18 BRT 交接文档：notes/HANDOVER_20261118.md（公开仓库副本 JeffWAM/docs/HANDOVER_20261118.md，占位符脱敏）。

- 2026-11-18 BRT 实验按时间索引：notes/TIMELINE_20261118.md（桌面副本 实验记录_按时间索引_20261118.md）；交接文档磁盘一行已按11-16清理更新。

- 2026-11-18 BRT E123（V1，Antigravity执行）：世界头局部动作敏感性与候选排序检验（211窗/27parent，零训练，0.022 GPUh）：潜在特征显著区分0.5s延迟/50%欠幅/gt vs n1（+50胜率81%、跨阶段91%）及+25/+100对应腕部相机的夹爪反事实，但对±20%幅度不敏感；从潜在特征→池化21谓词→朴素因果下一阶段分发生两层退化（49/160误选反向rev05、偏爱过冲s12），排除了直接用下一阶段发射概率做无约束动作重排。见 notes/e123_local_action_sensitivity_20261118.md。

- 2026-11-18 BRT E124（P1-v2，Antigravity执行）：解冻6层主干+50%观测/本体整块丢弃复验阶段条件动作头（U0/U2/D0/D2，各4000步，同E115-P1流）：阶段上界仍无收益且强制依赖显著变差（全160窗L1_50 U2−U0=-0.00022[-0.00079,+0.00035]，切换30窗+0.00107；D2−D0全窗+0.00218[+0.00081,+0.00391]、切换窗+0.00548），未触发因果臂U1/D1；意外发现单靠解冻主干纯确定性动作训练（U0，无阶段）使预测加速度能量由7.51e-7跃升至1.77e-6（真值2.59e-6），切换窗L1_50优于S0（-0.00215[-0.00450,-0.00005]），+50/+100世界头n1保留率升至25.1%/23.9%。见 notes/e124_p1v2_unfrozen_dropout_20261118.md。

- 2026-11-18 BRT E125（Antigravity执行）：全量145 parent池4000步同流对照（U_full解冻纯动作头 vs J_clean去流匹配干净联合 vs J_legacy原联合，对比冻结D与E106C，211窗/27parent）：L1_50 U_full−D=+0.00049 [-0.00048,+0.00132]、J_clean−J_legacy=-0.00091 [-0.00170,+0.00000]；总耗时53.3min。见 notes/e125_fullpool_unfrozen_and_clean_joint_20261118.md。

- 2026-11-18 BRT E126-D（Claude，只读诊断）：默认 E106-C 对比只用关节历史的盲策略MLP（211窗/27parent）：L1_8 模型0.00790 vs 盲0.00611（差+0.00178[+0.00108,+0.00250]），L1_50 0.0431 vs 0.0372（差+0.00582[-0.00052,+0.01035]）；模型仍含盲策略没有的信息（+50回归系数0.37[0.10,0.57]）；夹紧时刻末端误差11.3 vs 11.1mm；张开事件±10帧命中仅4/16。见 notes/e126d_workability_20261118.md。

- 2026-11-19 BRT E127-R（Claude，预注册判据未通过）：残差动作头（盲策略+视觉残差，冻结主干4000步，D样本流sha一致；预注册 code_e127/PREREG.md）。R_L1 对盲策略：L1_8 +0.00022[+0.00001,+0.00046]，L1_50 +0.00047[-0.00086,+0.00160]；G1/G2 不过，G3/G4 过但无意义。换图像代价 R_L1 1.2mm vs D 5.2mm（残差基本忽略视觉）。对照：C_L1（仅换L1）仍输盲策略；R_MSE L1_50 +0.0038。结果 results/e127r_eval_20261118_01。

- 2026-11-19 BRT E128（Antigravity执行）：①盲策略优势瀑布拆解：同历史(6,12,18,24)单种子下盲MLP(0.00855/0.04283)实际差于WAM(C_L1 0.00754/0.04192、J_clean 0.0417)，原盲策略优势全部来自新增t-3速度(-0.00196/-0.00325)、等权L1与3种子集成(-0.00047/-0.00235)，再加t-1速度可达0.00512/0.03649；②切断本体的纯视觉残差(WAM_PV/SpatialPV)训练残差降16~17%(否定无梯度)，但跨parent袋外相关≈0(最优收缩α=0，仅松爪时刻WAM_PV改善-3.69mm)，而端到端耦合模型混合(0.75*blind_fold5+0.25*E106C)显著优于blind_fold5(0.03554 vs 0.03662)。见 notes/e128_blind_decomposition_and_pure_vis_20261119.md。

- 2026-11-19 BRT E129（Antigravity执行）：端到端升级版联合WAM（补t-1/t-3速度230维+等权L1+余弦退火+世界头25% detached_N1消除暴露偏差，211窗/27parent）：单种子WAM_v1_s90 l8/l50=0.00709/0.03999，WAM_v2_s90=0.00710/0.04001；3种子集成WAM_v2_ens3 l8/l50/e50=0.00708/0.03995/34.81mm（对比原盲策略3s 0.00611/0.03723/35.41mm）；+50世界头N1保留率=10.6%（N1−no=+0.60pp）。见 notes/e129_upgraded_end2end_wam_20261119.md。

- 2026-11-19 BRT E130（Antigravity执行）：三项机制核验（211窗/27parent）：①逐关节/逐帧拆解+正运动学反事实；②去状态噪声重训：N0(仅去state_noise/drop) l8/l50=0.00710/0.03995，N1(+本体快LR)=0.00692/0.03988，N2(+直连运动学skip)=0.00669/0.03981（e50=35.11mm）；③同幅扰动真值驱动WAM_v1世界头(+50保留率)：pred_N1=11.6%，gt+跨窗同幅误差=39.5%，gt+逐窗等L1误差=57.6%，镜像误差(2gt-N1)=82.7%。见 notes/e130_diagnostic_breakdown_and_clean_state_20261119.md。

- 2026-11-19 BRT E131（Antigravity执行）：MuJoCo 3.8.1 双臂Piper真机工作台闭环验证（完全对齐piper.json：±0.3027m基座、z=-0.0044m台面、kp=10/kd=0.8+重力前馈，FK对齐<0.00mm，211窗/27parent）：S2视觉+物理重规划(+50末端/松爪)：GT底座=10.77/11.45mm，E106C=43.22/33.13mm，WAM_v1=40.84/32.84mm，WAM_N2=41.20/32.53mm，blind_t1=43.74/34.51mm；S3高频K=4反馈(WAM_v1 e50=49.68mm, jerk=1.90e-07)；S4开局15mm扰动恢复(+50末端)：WAM_v1=50.76mm vs blind_t1=59.75mm。见 notes/e131_mujoco_bimanual_piper_sim_20261119.md。

- 2026-11-19 BRT E132（Antigravity执行）：①补测三维末端姿态旋转误差(deg)：+50步 WAM_v1=11.13° vs blind_t1=10.43° vs E106C=11.40°；②N2去视觉仅保留kin_skip测试：l8=0.01054 vs N2全量=0.00669；③MuJoCo重规划滞后修复(WAM_v1 K=4 +50步末端/姿态)：M1朴素=49.68mm/16.82° -> M2期望参考锚定=44.05mm/14.73° -> M3(+MIT速度前馈)=42.96mm/14.35°（0碰撞）；M3开局15mm扰动恢复：WAM_v1=45.58mm vs blind_t1=59.77mm。见 notes/e132_replan_lag_fix_and_orientation_20261119.md。

- 2026-11-19 BRT E133（Antigravity执行）：第一梯队三项改进4组阶梯对照（G1三维位姿+速度斜率损失 -> G2+夹爪切换加权与跳变帧头 -> G3+15%视觉遮挡 / G4+20%遮挡+直连skip，211窗/27parent）：G3 l8/l50/e50/rot50=0.00727/0.04039/35.07mm/11.09°，G4=0.00676/0.04008/35.88mm/11.07°；夹爪F1@±10/松爪命中：WAM_v1=0.316/31.2% -> G3=0.385/50.0%；+50世界头N1保留率：G3=12.9%；MuJoCo M3 S2闭环(+50)：G3=36.73mm/12.95°。见 notes/e133_pose_vel_grip_tokendrop_20261119.md。

- 2026-11-19 BRT E134（Antigravity执行）：流匹配(Flow Matching)动作头与梯度解耦夹爪头4组对照（211窗/27parent）：①解耦确定性D_dec：e50/rot50=34.95mm/10.99°，夹爪F1@±10=0.293，松爪命中=37.5%；②残差流匹配FM_res最优：mean8 e50/rot50=34.12mm/10.85°(全系最低)，单样s1松爪位置/姿态=22.69mm/6.87°，Best-of-8 e50=29.31mm(松爪19.86mm)，世界头+50/+100保留率升至19.1%/8.6%(vs WAM_v1 11.6%/2.6%)；③但单样位移RMS仍停在0.0966(GT=0.1566)、win166仍139.4mm，证伪仅换流匹配动作头即可打破幅度收缩；④整集420帧(14s)连续MuJoCo闭环揭示离线稀疏缓存视觉+自回归本体产生视本不一致死锁。见 notes/e134_flow_matching_and_decoupled_grip_20261119.md 与 results/e134_full_ep_video_20261119_02/。

- 2026-11-20 BRT E135（Antigravity执行）：真机实拍原视频+校准3D URDF半透明虚影与50步轨迹叠加可视化验证（episode_62 / piper_01a709ceeae49b530c694814，550帧=18.3s连续11段50步M3闭环）：①纯机械臂轮廓自动标定将cam_high掩码IoU从0.381提至0.643（相机外参=[+6.7mm, +29.9mm, +689.1mm]，GT URDF与真机黑臂像素级重合）；②直观揭示两类核心差异：在右瓶取扎带(t=1.3s)与左提右松滑移(t=14.5s)处，FM_res与D_dec均准确触发夹爪开合，而无视觉blind_t1漏抓/死夹([GRIP DIFF])；③同时直观拍到快速上提阶段(t=14.5s)WAM预测轨迹方向正确但上提幅度滞后真机约5~8cm（幅度收缩的直接视觉证据）。见 code_e135/ 与 results/e135_ar_urdf_overlay_20261119_02/。
