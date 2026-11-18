# E52：Tianji 双臂 + Wuji 灵巧手 MuJoCo 闭环评测方案

日期：2026-11-05  
性质：只读调研与方案设计；未安装依赖、未运行仿真、未占用 GPU。`/mnt/workspace/colleague-e` 与 `/mnt/workspace/colleague-c` 未写入。

## 0. 结论与评测边界

MuJoCo 已经不是从零开始：项目内有可编译的 54-DoF Tianji+Wuji 模型、桌/瓶/箱场景、顶部相机闭环、15 Hz 绝对关节目标执行器、脚本抓放成功基线，以及 `LeWAM proposal → 候选变体 → MuJoCo rollout → outcome score → 只执行首帧` 的 oracle-state MPC 骨架。当前缺口主要是资产和接触参数未做真机辨识、顶部相机仍用近似 `fovy=60`、碰撞网格过粗、批量评测与正式状态机不完整，而不是缺少基本仿真入口。

建议固定项目当前 `mujoco==3.17.0` 作为 E52 基线。该版本是 2026-11-05 核实到的官方最新 release，且项目 `requirements-mujoco.txt` 已锁定它；3.17 新增的 `mujoco.batch` 可用于同一模型的多状态 CPU rollout，但 MVP 先用容易审计的多进程 episode worker，候选 rollout 再按 profiling 决定是否迁移到 batch API。版本依据见 [MuJoCo 3.17 release](https://github.com/google-deepmind/mujoco/releases/tag/3.17.0) 和 [3.17 batch API changelog](https://mujoco.readthedocs.io/en/3.17.0/changelog.html)。

评测分成三条必须分开报告的轨道：

1. `physical/no-assist`：无焊接、无吸附、无物体位姿覆盖，作为唯一可用于评价仿真内抓取能力的主结果。
2. `oracle-state closed loop`：策略仍只看顶部 RGB+q+dq；候选生成/评分可读取瓶 pose/velocity 与箱几何。它验证规划接口和仿真后果模型，不验证视觉物体定位或真机成功。
3. `assisted transport/place`：只有在真实接触已满足明确 gate 后才临时约束瓶与手，用于隔离搬运/投放接口。所有结果必须带 `assisted=true`，不能和主成功率合并，也不能声称抓取保真。

第一个性能里程碑必须是**真实示教 54D 关节轨迹在 MuJoCo 中的开环重放成功率**。如果真实成功轨迹在合理参数带内不能重现抓起/入箱，就不能用该仿真评价闭环模型好坏。

## 1. 现状盘点

### 1.1 `sim/` 已有内容

核心目录是 `sim/mujoco_pick_place_poc/`：

- 双臂资产：`assets/m6_left/robot.urdf`、`assets/m6_right/robot.urdf`，每臂 8 links、7 revolute joints；每个 link 有 inertial，visual/collision 都引用 8 份 STL。
- 双手资产：`assets/wuji_hand/{left,right}/hand.json`、生成后的 `hand.xml` 和每手 22 个 STL。每手 27 links、20 revolute、6 fixed；viewer bundle 只给运动学/视觉网格，没有可信质量、惯量或执行器参数。
- 当前装配：`simulate_wuji_ep56.py::build_model` 用 `MjSpec` 直接加载左右臂 URDF、生成并 attach 双手 MJCF；固定物理顺序为 A arm 7 + B arm 7 + left hand 20 + right hand 20。完整模型为 54 个位置执行器；瓶 freejoint 使现有报告中的 `nq=61, nv=60` 合理。
- 当前手动力学占位：手根质量 0.20 kg、每个可动指节 0.012 kg，指节对角惯量约 `[4e-6,4e-6,2e-6] kg m²`；手 actuator 力矩上限 2.0（转动关节下应解释为 N·m）。这些都是 replay 占位，不是 Wuji 实测参数。
- 当前碰撞：默认 `mesh_hull` 将每个手部 STL 作为 mesh geom；MuJoCo 对 mesh 碰撞实际使用其凸包。另有 capsule/box 代理模式，但当前修复后的成功脚本使用逐 link convex hull。双臂 URDF 的 collision STL 仍过于精细且历史零位出现 8 个自碰撞、最大约 60.9 mm 穿透；当前主要靠碰撞 bitmask 关闭机器人内部 arm-arm、arm-hand 和手内自碰撞。
- 场景：`scene_teleop_viewer_task.xml` 及 EP056 对齐/contactsafe 变体，包含桌、开放箱、复合刚体瓶、支架、`top_camera`。当前 physics `timestep=0.0005 s`、`implicitfast`、`cone=elliptic`。oracle planner 运行时还会把瓶重置为 upright、中心约 `(-0.1564,0.6060,0.6550)`，与 scene XML 默认 `(-0.32,0.73,0.655)` 不同；正式评测必须把 reset pose 写入 manifest。
- 控制与观测：EP056 345 帧/15 Hz/54D 实测 q 回放、实时顶部相机闭环、q/dq 因果反馈、关节限位/速度/加速度整形、OSMesa 无头渲染均已接通过。修复后 scripted baseline 已完成一次抓起、搬运、释放入箱，并在抓取点 X/Y ±5 mm 的 5 个诊断偏移上通过；这不是随机场景成功率。
- planner：`src/lewm_tianji/planning_contract.py`、`src/lewm_tianji/object_mpc.py` 和 `plan_lewam_object_mpc_20261028.py` 已定义 `PlannerObservation`、`ActionCandidate[H,54]`、候选生成、`mj_copyData` 状态复制、物理 rollout、分阶段 outcome score、候选拒绝原因和 `execute_frames_per_replan`。默认 policy horizon 为 10，现有 planner 默认 rollout 5 帧、执行 1 帧。

### 1.2 仍缺少的关键项

- Wuji 整手的权威 URDF/CAD、逐 link 质量/质心/惯量、gear/torque/velocity limit、指腹材料和关节摩擦；现有 JSON→MJCF 只能作为可复现几何起点。
- 双臂和手的专用低复杂度 collision model，以及经过审计的 allowed-contact matrix；当前“全部内部碰撞关闭”无法统计真实自碰撞。
- Link7→手根真实装配标定。现有 archived transform 的 `accepted_for_metric_use=false`，法兰到手根仍是显示假设。
- 真实桌、瓶、箱的尺寸、质量分布和摩擦实测；当前物体摆位是 sim baseline，不是 real-scene twin。
- 根据 E46 外参和真实内参配置的 MuJoCo calibrated camera 与 Brown-Conrady 后处理；现有 `top_camera` 只是 `fovy=60` 的近似。
- 真机控制器的 PD/阻抗、延迟、插值和饱和行为；当前 arm `kp/kd=160/30`、hand `8/0.4`、全机器人 gravcomp=1 只是仿真假设。
- 正式 episode reset/randomization、状态机、置信区间、失败分类、批量 runner、manifest 和 open-loop replay 基准。

## 2. URDF/JSON → MJCF 资产流程与验收

MuJoCo 官方建议对 URDF 加必要 compiler extension，加载后保存为 MJCF，再在 MJCF 侧用 include 扩展；URDF 只能表达 MuJoCo 模型的子集，而且 URDF 内自定义 `<mujoco>` 属性不会做 schema 校验，拼写错误可能被静默忽略，见 [官方 URDF 建模说明](https://mujoco.readthedocs.io/en/3.17.0/modeling.html#urdf-extensions)。建议采用以下可再生流程：

1. 为左右臂保留原始 URDF/STL 为只读 source，写转换 manifest（源 hash、MuJoCo 版本、compiler 参数）。以 `MjSpec.from_file(urdf)` 加载，显式设置 asset root；当前代码已发现把 `meshdir=meshes` 再叠加 URDF 中 `meshes/...` 会重复路径，这一行为必须有回归测试。
2. 首次转换用 `discardvisual=false`、不静默 fuse fixed body；编译后保存规范化 arm MJCF。`balanceinertia=true` 只能作为诊断/临时兼容选项，不能用它掩盖负特征值或不满足三角不等式的惯量。生产资产应修正源惯量并记录来源。
3. Wuji 当前没有整手 URDF。建立一个 canonical hand intermediate（27 links、20 revolute、6 fixed、joint axis/origin/range、mesh transform、canonical 20D index）；由同一 intermediate 同时生成审计用 URDF 和运行用 MJCF，避免手工维护两份顺序。若后续拿到厂商 URDF，先与 intermediate 做 name/axis/origin/limit/FK diff，再替换 source，不能按 XML traversal order映射 20+20 关节。
4. 每个 link 的 visual 与 collision 分层。动态三角网格不能被误认为凹形精确碰撞：MuJoCo 除 SDF 外只处理凸 geom，用户 mesh 碰撞会被替换为凸包，见 [collision checking 文档](https://mujoco.readthedocs.io/en/3.17.0/computation/index.html#collision-detection)。建议手指每节用 capsule 或 1 个手工凸 hull，指腹另加小 box/capsule pad；掌/腕用 2–4 个 convex/primitive。臂 link 用 capsule/box/少量凸 hull。复杂凹形若需要保留，必须拆成多个凸 geom，不能只给一个凹 STL。
5. 给 collision geom 单独命名和 contact class：`arm_shell`、`palm_shell`、`finger_pad`、`finger_shell`。先关闭相邻 parent-child 碰撞，再逐步打开同手非相邻手指、双臂、手-前臂；允许抓取的 finger-pad↔bottle 不能被 bitmask 误关。
6. 建立唯一 `joint_contract.json`：54 个 index、策略名、MJCF joint/actuator 名、方向符号、零偏、hard/soft limits、effort/velocity、来源。启动时按名字双向校验，缺失、重复、顺序漂移直接 fail。

资产验收门槛：

- 恰好 54 个受控 hinge，顺序和单位正确；每手 20 关节均能在上下限内独立正负扰动。
- 零位及至少 20 组随机姿态与源 FK 对齐，工程门槛建议 link position max <1 mm、rotation max <0.5°；这只验证转换，不等于真实装配准确。
- 每个动态 link 质量>0、惯量正定且物理合理；报告总质量、质心和各 link inertia source。当前手部占位参数必须带 `estimated=true`。
- 碰撞代理覆盖 visual 的指腹/外壳，但不产生初始穿透；逐 link 可视化 convex hull。静止 10 s 无 NaN、爆炸或持续关节限位力。
- 关节 limit、actuator ctrlrange/forcerange 一致；不能通过修改 `qpos0` 实现张开手，因为它会改变 hinge 零位。沿用现有 keyframe reset 做法。

## 3. 场景、相机和随机化

### 3.1 nominal 场景

先复制现有可比基线，不把它称为真机数字孪生：

- 桌面 1.30×1.10 m，台面 z=0.55 m。
- 箱中心约 `(0,0.78,0.575)` m；外底 0.28×0.22 m，墙厚约 20 mm，因此当前内腔约 0.26×0.20 m；墙口约 z=0.675 m。实现时把 `bin_inner_aabb` 和 rim height 作为显式 site，不从 geom 名字临时推导。
- 瓶初始中心当前约 `(-0.32,0.73,0.655)` m；碰撞体最大直径约 68 mm，总高约 256 mm，质量 0.42 kg，惯量为 `[0.0018,0.0018,0.00035] kg m²`。它是刚性 PET+内容物的复合 cylinder/ellipsoid 近似，无薄壳形变或液体晃动。
- 当前 scene friction：全局 `0.9 0.02 0.002`，瓶 `0.75 0.02 0.002`，手 `1.2 0.02 0.002`。MuJoCo 同 priority 接触会对 `solref/solimp` 混合，并逐项取较大的 friction；因此“瓶写 0.75”并不代表手-瓶有效 sliding friction 是 0.75。E52 必须用 contact pair 或 priority 明确各材料对的实际值。

### 3.2 顶部 RGB 相机

真实顶部相机记录为 Orbbec Gemini 335L，640×480；存档内参为 `fx=366.5326538, fy=366.5903625, cx=321.3335876, cy=238.5860748`，Brown-Conrady `D=[-0.02863151,0.03352833,0.000352739,-0.000154852,-0.01185825,0,0,0]`。MuJoCo 3.17 支持 `resolution/focalpixel/principalpixel/sensorsize` 的离线 calibrated pinhole camera；这些属性和 principal point 行为见 [camera XML reference](https://mujoco.readthedocs.io/en/3.17.0/XMLreference.html#body-camera)。renderer 本身不模拟 Brown-Conrady 畸变，因此流程应为：

1. 将 E46 使用的 `T_world_from_top_rgb`/robot base 坐标约定显式换成 MuJoCo camera frame（camera 看向 local -Z，`xyaxes` 分别对应像素 right/up），用点投影单测确认，而不是复制现有 `pos/xyaxes/fovy`。
2. 以 640×480、上述 K 渲染无畸变 RGB；对每个 distorted output pixel 用 Brown-Conrady 的逆查找映射采样 ideal render，记录 pixel-center convention。边缘不足时渲染略大 overscan 后 remap，避免黑边成为策略捷径。
3. **策略输入沿用当前活跃实现的完整 640×480 → `ImageOps.contain(224,224)` → 黑边居中 → ImageNet normalization**。E46 的 rows 96:480、640×384 crop 是几何投影/ROI 评测口径，不是当前 `preprocess_top_rgb` 的策略预处理。两条路径分别测试，禁止在 E52 中悄悄换训练预处理。
4. 每个 observation 同步记录 render time、q sample time、policy start/end、command apply time。E46 的 `time_lag_frames=1` 只表示离线 `image[t]` 对 `measured q[t-1]` 拟合较好；projector 不移位，闭环动作也不能整体平移。评测显式消融 observation latency 0/1/2 个 15 Hz frame。

E46 约束必须原样继承：`camera_delta=I`，掌点在同分布可见姿态下可作 2D ROI 软先验；A/B 掌点严格入 mask 约 98.2%/97.8%，但原 URDF 指尖投影未通过，米制高度 proxy 未通过。A/B base 均 `accepted_for_metric_use=false`，A 姿态子集敏感性 65.71 mm/6.57°，B 为 5.56 mm/0.90°。这些数值不是 camera covariance，不能直接当随机化标准差。相机验收用 heldout q 运动学回放比较 robot silhouette/掌 ROI；桌面/瓶/箱绝对 3D 仍需独立量测。策略观测全程只使用这一台顶部 RGB，相机阵列中不创建或读取手腕相机。

### 3.3 无头渲染与速度

Linux 官方支持 GLX、OSMesa 软件无头和 EGL 硬件无头，见 [MuJoCo rendering 文档](https://mujoco.readthedocs.io/en/3.17.0/programming/rendering.html)。项目已有 OSMesa 隔离库成功运行，不需要修改主机系统；旧 README 记录该工作站当时 EGL 不可用，故 EGL 必须先 smoke test，不能假定可用。

- CPU-only 可复现基线：OSMesa，单 renderer 640×480；物理候选 rollout 不创建 renderer。
- 可选加速：EGL 只在独立环境 smoke 成功后使用；不为此改驱动或系统包，也不计入物理 CPU 并行能力。
- 必测四段耗时：physics-only、render-only、policy-only、端到端；分别给 p50/p95 和 simulated-seconds/wall-second。已有实测仅能作为量级：344 个 15 Hz step、22.93 s 仿真，实时 top RGB+CPU LeWAM 约 46.7–48.2 wall s（7.1–7.4 decision Hz）；scripted render+physics 约 8.84 Hz。E52 尚未实测 calibrated K+D 或 MPC，不能直接沿用这些数值作为承诺。

### 3.4 随机化分层

先固定 nominal 通过资产和开环重放，再使用冻结 seed set 分层随机化：

|组|建议范围（初始工程范围，待实测收缩）|
|---|---|
|瓶初态|XY ±20 mm nominal stress、±40 mm hard stress；yaw `[−π,π]`；roll/pitch ±2°；初始速度 0|
|瓶动力学|质量 0.42 kg ±15%；质心 z ±5 mm；finger-pad/bottle sliding friction 0.6–1.2；bottle/table 0.4–0.9；restitution 0–0.05|
|箱/桌|箱 XY ±10 mm、yaw ±2°；桌高 ±5 mm；尺寸误差单独 ±3–5 mm，不和 pose shift 混为一个变量|
|控制|gain/effort ±10–20%；零偏 ±0.5°；观测/命令延迟 0–2 帧；仅在有日志后加入 backlash/deadband|
|相机|baseline 不移；诊断带 ±5 mm/±0.5°，stress ±10 mm/±1°；E46 A-base 65.7 mm/6.57°只作极端 sensitivity，不作概率分布|
|外观|光强/方向/色温、曝光/白平衡、桌/箱/瓶颜色与粗糙度、轻微 blur/noise/JPEG；保留无随机化诊断域|

每个 episode 保存全部 sampled parameters。正式报告至少分 nominal、physics、camera、appearance、latency、combined，不能只给混合均值。

## 4. 54D、15 Hz 控制接口

模型/adapter 对外契约是每 1/15 s 接收 `q_target[54]` 绝对弧度，顺序固定 A7+B7+left20+right20。若底层 LeWAM 仍输出 normalized residual chunk，只允许在 proposer 边界用**本次重规划 fresh measured q**重建绝对目标；不能把预测 q 当下一次观测。

执行链：raw absolute target → name/order/finite 检查 → hard limit → hand posture约束 → velocity/acceleration/braking envelope → position actuator target → physics step → 新 q/dq/contact/RGB。raw、projected、shaped 和实际 q 必须分别记录，不能只存最终 command。

### 4.1 actuator 与 gain

- 每个 hinge 使用 `position`/等价 general actuator，显式 `ctrlrange` 和 `forcerange`。`kp/kd` 是连续 PD，不用直接写 `qpos`；位置 actuator 定义见 [官方 actuator reference](https://mujoco.readthedocs.io/en/3.17.0/XMLreference.html#actuator-position)。
- 当前 arm `kp=160,kd=30`，effort 来自 URDF；hand `kp=8,kd=0.4, |force|≤2` 可作 initial profile，不能作真机值。至少按 arm、thumb CMC/MCP、其他 MCP flex/abd、PIP/DIP 分组。保留 robot gravcomp=1 作为显式 ablation，因为现有代码启用了它。
- 先做无接触 step/sine tracking，比较 rise time、overshoot、steady error、effort saturation；再做准静态夹持。`Kp≈tau_limit/e_sat` 给起点，Kd 按反射惯量/临界阻尼附近扫。若没有真实 controller log，只报告 sensitivity band。

### 4.2 timestep、子步与插值

主基线沿用已通过 scripted grasp 的 `dt=0.0005 s`、`implicitfast`，即 2000 physics Hz。15 Hz 周期不是整数子步（133.333），所以不能写死 133；用绝对目标时间/累加器交替执行 133/134 step，确保长期无 drift。另做 `dt=1/1500 s`（恰好每控制帧 100 step）和 1 ms sensitivity；只有结果稳定且接触穿透/solver residual 合格才可加速。

不建议对 raw policy 再加任意低通，因为它会改变模型动作并掩盖失败；需要的最小整形是：

- hard joint/effort limits；现有 provisional `vmax arm=1 rad/s, hand=2 rad/s`、`amax arm=4 rad/s², hand=8 rad/s²` 只作起点；
- 在 physics rate 将上一 applied reference 向本帧 bounded target 做 slew/acceleration-limited interpolation，避免 15 Hz 阶跃激起接触冲击；ZOH 作为对照；
- 重规划后从当前 applied reference 和 measured q 继续，不重复从 chunk 首帧启动；若 projection 大于门槛，候选记 invalid，而不是大幅 clip 后仍当成功。

## 5. 灵巧手抓取保真度

### 5.1 接触配置与微基准

建议 nominal：`solver=Newton`、`cone=elliptic`、`condim=4` 仅用于需要 torsional friction 的 finger pad，普通 shell/table 用 condim=3；iterations 从 50–100、tolerance `1e-8` 起，根据 `data.solver` 收敛统计而不是盲目加迭代。elliptic 更接近连续摩擦锥，pyramidal 作为 solver sensitivity。MuJoCo 官方说明 elliptic 更物理、pyramidal 有时更快/稳；`noslip` 是后处理，只能作为敏感性项，不能用来强行锁住瓶子。

`solref/solimp` 表示软约束，不等于真实软指腹。当前 runtime 曾把 active geom `solref` timeconst 设到 0.5 ms，和 timestep 相同；官方建议正 timeconst 通常至少约 `2×dt`，否则会被 `refsafe` 限制或产生刚性/步长耦合。E52 从 `solref timeconst=2–5 ms, dampratio≈1`、`solimp d0=0.95–0.99` 做固定网格扫描，报告 penetration、creep、solver iterations 和 grasp survival，不逐 episode 调参。参数解释见 [contact modeling guidance](https://mujoco.readthedocs.io/en/3.17.0/modeling.html#solver-parameters)。

在完整任务前冻结三项 micro-benchmark：

1. 瓶-桌斜面：测开始滑动角/减速度，确定 bottle-table friction band。
2. 双指/多指准静态夹持：逐步减小法向力，测滑落阈值、接触 patch、穿透和 chatter。
3. 抓持扰动：垂直/横向加速度和小角度腕旋转，保持 2 s；报告瓶位姿漂移、接触切换和最小法向力。

### 5.2 已知不稳定点

- 40 个手关节、多个 convex hull 会产生大量同时接触；raw convex hull 可能把凹槽/指腹填平，接触位置看似稳定但物理错误。
- 手部惯量、关节摩擦、力矩和装配均未实测；高 gain+轻 link 会造成 chatter，高 friction/高 solimp 会人为“粘住”。
- 同 priority friction 采用逐项最大值，当前 hand 1.2 会支配 hand-bottle；必须显式 pair，避免把材质字段误读成有效 pair friction。
- 关闭所有手内自碰撞会允许手指穿手掌/彼此交叉；完全打开又可能因初始 hull 重叠导致 solver 爆炸。需要 allowed-contact matrix 和 signed-distance startup audit。
- 刚体 PET、无液体和无柔软指腹限制了 torque closure；不能从单一参数组成功推断真机。
- position servo 饱和时实际 Kp 不再决定接触力；必须同时报告 actuator force saturation，而不是只看 q error。

### 5.3 可降级辅助抓取（削弱结论）

主结果禁止辅助。若物理抓取尚未稳定，可用以下两级隔离后续接口：

- `contact_latch_assist`：只有同一手的 thumb+至少两个非 thumb finger 有效接触持续 ≥0.2 s、瓶已离桌趋势为正、姿态/速度门槛满足时，激活一个有有限 force/torque、可调 compliance 的 weld equality；松手或接触丢失即释放。它只能评价 transport/place 与 replanning，报告必须写“抓取成功由辅助约束保证”。
- `kinematic_attach_smoke`：瓶在 gate 后直接随手根，用于最早接口 smoke。它连搬运载荷/滑移都不验证，只能证明状态机和放置路径连通。

辅助结果使用独立表格和分母；不得用于 `physical_grasp_success`、sim2real 结论或与无辅助方法做合并提升。

## 6. 成功判据与指标

所有 task ground truth 来自 MuJoCo state/contact；视觉 policy 不读取。oracle planner 可读取但必须标记。

1. `reach_contact`：目标手 pad 与瓶距离≤2 mm 或发生接触；诊断项。
2. `grasp_latched`：同一手 thumb 与至少两个非 thumb pad 接触，三指组合在 0.2 s 内稳定出现；瓶不能仅由桌/箱支撑，横向漂移≤35 mm、倾角≤15°。wrist/palm shell 单独接触不算。
3. `lift_success`：在 `grasp_latched` 之后，瓶碰撞体最低点离桌≥25 mm，连续≥0.5 s；另报≥50 mm/2 s 的严格门槛、最大离桌高度和滑移。
4. `transport_success`：lift 已成功，瓶 envelope 进入箱口内缩安全区域上方，过程中离桌≥25 mm、未掉落、倾角≤20°。
5. `place_success`：瓶 collision envelope 的水平投影在箱内边界内（建议 margin 5 mm），最低点低于 rim，瓶与箱底/内壁接触；手已打开且无 hand-bottle 接触，随后 1.0 s 内瓶线速度<0.02 m/s、角速度<0.2 rad/s且不弹出。只检查 center XY 不够。
6. `task_success`：lift 和 place 顺序成立，且无 timeout、物体出界、严重碰撞、NaN、hard-limit violation。

主指标：task success；`P(grasp)`、`P(lift|grasp)`、`P(transport|lift)`、`P(place|transport)`及 Wilson 95% CI。辅助指标：time-to-stage、重规划次数、掉瓶率、最大倾角、接触力/冲击、瓶滑移、q tracking RMSE、actuator saturation、raw/projected target差、solver nonconvergence、关节 hard/soft-limit次数。

碰撞违规单独计数：arm/table、arm/bin、非目标手/object、同手非相邻指穿透、双臂碰撞；按事件数、累计 physics steps、最大 penetration/force 报告，不能用一个 contact count 混合。开发用 30 fixed seeds；配置冻结后 nominal 和各 shift 组各 100 seeds，使用同 seed 做配对比较。

## 7. Receding-horizon 与候选评分接口

沿用现有 contract：观察 `top_rgb[480,640,3] + two-frame history + measured q + causal dq rad/frame + validity`；oracle baseline 另读取 bottle pose/velocity 和 bin geometry。LeWAM chunk 每次从 fresh q 锚定成 `H×54` absolute targets，`generate_candidates` 加入 object-relative IK、hand synergy、raw LeWAM、bounded residual/hold；每个候选从同一 snapshot rollout，`score_outcome` 按阶段评分。

执行 `k∈{1,2,4}` 个 15 Hz command 后获取新 RGB/q/dq 再规划；默认 `k=1`。approach 可在开发集比较 k=2/4，首次接触、闭手、lift、place 强制 k=1。horizon 先保持现有 H=5 rollout / policy H=10，只有 short-horizon 不能区分候选时再扩展，避免成本无界增长。

需要保留/补齐的 hook：

- `snapshot()`/`restore()`：qpos、qvel、act、ctrl、time、warmstart、plugin/user state 和 RNG；候选之间不得污染。现有 `MjData + mj_copyData` 是起点。
- `propose(observation)->list[ActionCandidate]`：每个 candidate 带 family、raw/projection幅度、seed、fresh-q hash。
- `rollout(snapshot,candidate,H,scenario)->PredictedOutcome`：无渲染 physics-only；对随机 rollout 使用 common random numbers。
- `score(outcome,phase)->terms`：保存每个 term、eligibility 和 rejection reason；不能只存赢家分数。
- `select_or_hold()`：全候选 invalid 时 hold/安全张手，仍保留完整分母。
- `execute(first k)` 后对 predicted vs realized object/q/contact 做 calibration，报告 top-1 success、oracle regret、candidate ranking disagreement。

## 8. 批量 CPU 评测和耗时估计

主机只读检查为 2 sockets ×48 physical cores ×2 SMT =192 logical CPUs、2 NUMA nodes。不要默认占满机器。建议：

- 第一轮 8 个 episode processes、每进程 1 physics thread，`spawn` 启动；逐步 benchmark 8/16/32 process。限制 BLAS/torch thread=1–2，避免 oversubscription；先在单 NUMA node 内绑定，再测跨 socket。
- 每 worker 加载一次只读 `MjModel`，每 episode 新建/重置 `MjData`。renderer 与 EGL/OSMesa context 不跨 fork/thread共享；每个真正执行的 episode 每 15 Hz 渲染一次，候选 physics rollout 完全不渲染。
- 相同 model 的 K candidates 可在第二阶段试 MuJoCo 3.17 `mujoco.batch.Batch`。它能在内部 thread pool 步进多状态并保持结果与普通 loop bit-identical；但本项目每子步还有自定义 contact/阶段统计，先验证 callback/observation是否能覆盖，再替换简单多进程。

耗时只能给可验证的区间与公式。已有无 MPC 端到端为约 47–48 s/22.93-s episode。设每回合 N 次 replan、K candidates、rollout H 个 15 Hz frame，则额外物理量约 `N*K*H/15` simulated seconds；渲染和 policy 每 replan只做一次，不应乘 K。以 N≈344、K=8、H=5 为例，counterfactual physics 总量约 917 simulated seconds，实际 wall time取决于 contact-heavy physics RTF，当前没有实测。

工程预算先按无 MPC 40–60 s/episode、K=8/H=5 MPC 约 1–4 min/episode估计，并明确标记为 planning estimate；8 workers 跑 100 episodes 约 13–50 min，另加启动/汇总。MVP 首次交付必须用 1/8/16 workers 和 K=1/4/8 实测吞吐、p95、内存、NUMA scaling，之后替换估计。

## 9. sim-to-real 域差与第一个里程碑

### 9.1 真实示教开环重放基线

对所有可用且真实标注成功的示教，建立 episode manifest：真实 54D measured q、15 Hz timestamp、顶部 RGB、任务成功标签、瓶/箱初态（能量测则写实测；不能则写 uncertainty）。每条做两种回放：

1. `kinematic replay`：逐帧设置 q，只验 joint order、FK、相机 silhouette、场景对齐和是否几何可达；它不产生物理成功率。
2. `dynamic open-loop replay`：q[0] 初始化后，将真实 measured q[t] 当作 15 Hz position target，经同一个 actuator/整形/physics 执行；不读取仿真反馈改变后续命令。记录 q tracking、接触阶段、lift/place。

measured q 不是当时真机 command，dynamic replay 因而是“轨迹可实现性基线”，不是精确复制真机控制器。若能找到真实 command/effort，应另加 command replay。核心指标是 open-loop dynamic task/stage success rate，在 nominal 与冻结 physics parameter band 中报告 sensitivity；同时按 episode 给 predicted contact timing 与真实视频中人工阶段标注的偏差。

晋级条件建议：资产/FK/相机检查通过；至少 80% 的真实成功示教在某个**预先冻结且微基准合理**的 nominal 参数组中重现 lift，至少 60% 重现 place，且结果在相邻 friction/gain setting 不发生灾难性翻转。阈值是首轮工程门槛，需在看到结果前冻结；未通过时优先修资产、初态、控制和接触，不进入 learned closed-loop 排名。

### 9.2 其他域差检查

- control：同一真实 q target 比较 real/sim tracking lag、overshoot、速度、steady error、saturation；用 step/sine 和示教频谱识别。
- contact：斜面/夹持/扰动真实小实验与 sim 参数带比较；在没有实测前只报告 sensitivity。
- visual：同 q 运动学回放比较 full-frame robot silhouette、E46 palm ROI、边缘 Chamfer、颜色/曝光/blur；冻结 encoder embedding与策略动作差异仅作行为域差诊断。透明 PET 的反射/折射单列。
- task：同一初始化/动作在 nominal、physics-shift、camera-shift 下的阶段 disagreement；成功率单值之外报告失败转移矩阵。

## 10. 里程碑、工作量与 MVP 清单

以下按 1 名熟悉 MuJoCo/机器人控制的工程师估计，不含获取厂商 CAD/惯量、真机小实验和外部排期：

|阶段|交付物|估计|
|---|---|---:|
|G0 资产/协议闸门|54D manifest；URDF/hand intermediate→MJCF；FK/inertia/limit/collision audit；固定 3.17.0|2–4 天|
|M1 开环重放（第一个性能里程碑）|真实示教 kinematic+dynamic replay；open-loop阶段成功率；sim/real tracking报告|3–5 天|
|M2 场景/相机|K+principal point、D remap、E46 palm/silhouette验收、nominal scene manifest|2–4 天|
|M3 接触/控制|PD/effort/整形；斜面/夹持/扰动；无辅助 scripted grasp sensitivity|4–7 天|
|M4 闭环 MVP|oracle state、k=1、H=5、候选 hook、状态机、30-seed报告|3–5 天|
|M5 批量正式评测|8/16/32 worker benchmark；分层随机化；100-seed配对和CI|3–5 天|

可用 MVP 约 3–4 人周；若需要重做 Wuji collision/inertia 或补真机控制/摩擦数据，较可信的 sim-to-real sensitivity 报告约 5–7 人周。

最小可行实施顺序：

1. 冻结 3.17.0、源资产 hash、54D joint contract 和 nominal scene；不改系统环境。
2. 生成/保存规范化 MJCF，做 20 姿态 FK、limit、inertia、初始穿透检查；先建立低复杂度 collision v1。
3. 接 calibrated pinhole K/principal point 与 Brown-Conrady remap；保留当前完整 640×480 policy preprocessing，并单独做 E46 crop 诊断。
4. 实现真实示教 kinematic/dynamic replay runner，先交付 open-loop replay success baseline。
5. 无接触标定 position actuator；冻结 raw→bounded→interpolated command 和 0.5 ms scheduler。
6. 完成斜面、夹持、扰动 micro-benchmark；冻结 nominal friction/contact及两个 sensitivity配置。
7. 用 scripted grasp 在无辅助轨道复核 lift/place；不通过则不归因于 LeWAM。
8. 接现有 oracle MPC，先 k=1/H=5；统一阶段状态机、日志、失败原因和 assist flag。
9. 30 nominal seeds 调试；冻结 k、threshold、randomization和参数。
10. benchmark 1/8/16 workers、K=1/4/8；选择不会压满主机的默认并发。
11. 正式 nominal+shift 各 100 seeds，报告阶段成功、碰撞/超限、CI、参数 sensitivity 和 predicted-vs-realized outcome calibration。

## 11. 主要风险与停止条件

1. **开环成功轨迹无法重现**：说明资产、初态、控制或接触没有校准。停止 learned policy 排名，先修 M1。
2. **高摩擦/辅助约束制造成功**：主结果固定无辅助，并报告相邻 friction/solimp/gain sensitivity；一旦成功只存在于窄参数点，结论限定为接口 smoke。
3. **手部 convex hull 错误**：指腹位置/凹形被填平会同时导致假抓住和假碰撞。以 primitive/多 hull v1 和 visual overlay验收。
4. **自碰撞过滤过宽**：关闭全部 robot internal collision 会产生不可实现手型。逐步 allowed-contact matrix，记录非相邻指 signed distance。
5. **相机标定过度解读**：E46 只支持掌点 2D 软先验，base 未 metric accepted；不使用指尖/高度作硬真值，不把 base sensitivity 当 camera noise。
6. **15 Hz target 与 2 kHz servo 不匹配**：ZOH/插值、PD、effort/latency会显著改变接触。raw/applied分开记录，并用真机 tracking收缩参数。
7. **candidate rollout 过拟合仿真**：用 common seeds、参数 shift、无辅助主结果和 open-loop replay gate；oracle score 的提升不等于视觉或真机提升。
8. **并行假加速**：renderer/context、BLAS线程和双 NUMA 会造成 oversubscription。逐级 benchmark，默认只占 8–16 physical cores，候选 rollout禁渲染。
9. **评测泄漏**：dev 30 seeds选择参数后冻结；正式 100 seeds一次性运行，保存 manifest。历史 EP056 是开发样本，不能称盲测。

## 12. 核实来源与未确定项

项目内只读核实：`PROJECT_DIRECTION.md`、`AGENTS.md`、E51（仅复用非 Isaac 接口/判据/观测/资产分析）、`sim/mujoco_pick_place_poc/` 的 README、scene、URDF、Wuji JSON/MJCF、控制、scripted grasp 和 oracle MPC；`notes/2026-11-04-e46-recalib.md` 与 `runs/e46_recalib_20261104/` 的 adopted params、constraints、height verdict、verification。

官方文档于 2026-11-05 核实：MuJoCo 3.17.0 release；URDF extension/conversion；mesh collision convex hull；camera intrinsics；position actuator；solver/contact；Linux EGL/OSMesa；3.17 batch API。实现时应链接 3.17.0 固定版本文档，避免 `latest` 行为漂移。

未确定且必须实测/补资料：Wuji 质量/惯量/torque/摩擦；真实 Link7-hand transform；真实桌/箱/瓶尺寸、质量和 pose；真机 controller gain/插值/延迟；EGL 在当前节点可用性；calibrated K+D 渲染速度；contact-heavy physics RTF；MPC 每回合耗时。上述均不得在结果中默认为已知。

## 修订记录（2026-11-05）

按用户原则修订评测边界：删除并作废 oracle-state MPC（MuJoCo rollout 生成/评估候选并打分）轨道，以及 M4 中与该候选评分路径对应的内容。第 7 节保留“每 k 步重规划”的接口，但候选评分只能由模型内部完成，禁止 MuJoCo rollout、状态或结果参与候选排序、评分监督或评分校准。MuJoCo 仅作为验证最终输出动作策略的环境，用于开环重放和闭环成功率验证：接收 54D 关节指令，返回顶部相机 RGB、q、dq。仅使用顶部相机，不使用腕部相机。

本修订不覆盖原文；修订前版本备份于 `notes/2026-11-05-e52-mujoco-closedloop-plan.md.bak_before_20261105_revision`。旧版相应 oracle-state/M4 描述保留为历史记录，但不再是本项目有效方案或实施指令。
