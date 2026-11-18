# E51：Tianji 双臂 + Wuji 灵巧手 Isaac Sim / Isaac Lab 闭环仿真评测备选方案

日期：2026-11-05（调研与方案设计；未安装 Isaac、未运行 GPU 仿真）

## 0. 结论与边界

建议把 Isaac 路线定位为 MuJoCo CPU 闭环的**异构仿真对照与视觉域差评测**，而不是立即替换主线。第一阶段只做单环境、单瓶、顶部 RGB、oracle 物体状态的闭环复现；它验证 54D 控制契约在第二物理引擎中的可移植性，并暴露碰撞、摩擦、驱动和渲染差异。只有在同一初态、同一动作序列的 MuJoCo/Isaac 逐步对照通过后，才扩展随机化或并行环境。

必须保留当前项目契约：关节顺序固定为 `A arm 7 + B arm 7 + left Wuji 20 + right Wuji 20`，角度为 rad；模型输出是 15 Hz 的 54D 绝对关节目标；每次重规划必须从最新仿真实测 `q` 重建/锚定动作，不能把上次预测状态当观测。输入只用顶部相机的因果两帧、measured q、因果 `q[t]-q[t-1]`（rad/frame，使用既有训练统计）和 validity bit，不引入腕部相机或 VLM。

本方案默认 Isaac Lab Direct workflow（便于显式控制闭环时序、状态机和逐阶段指标），PhysX 为第一后端。Isaac Lab 3.0 主文档在 2026-11-05 指向 Isaac Sim 6.2 extra；Isaac Sim 6.2 的 URDF importer 3.0 和多物理层资产流程与 5.x 有破坏性差异。因此实现前应把镜像 digest、Isaac Sim、Isaac Lab commit/tag、驱动和资产转换器版本写入 manifest，不混用 5.1 教程生成的 USD。官方版本依据见 [Isaac Lab 安装说明](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)、[Isaac Lab 资产导入说明](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html) 和 [Isaac Sim 6.2 发布说明](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html)。

## 1. 已有资产审计与 URDF→USD 流程

### 1.1 现状

- 双臂：`sim/mujoco_pick_place_poc/assets/m6_left/robot.urdf` 与 `m6_right/robot.urdf`，每臂 8 links、7 revolute joints，8 份 visual、8 份 collision，URDF 中各 link 均有 inertial。
- Wuji：没有整手 URDF。每手来自 viewer JSON，27 links、20 revolute joints、6 fixed joints、22 个 STL；顺序为每指 4 关节，左手 `q[14:34]`、右手 `q[34:54]`。JSON 只给运动学和视觉网格，现有质量/惯量、胶囊/盒/球碰撞代理是 MuJoCo 侧近似。
- 手安装：现有 MuJoCo 使用 `T_link7_from_sdk_flange_m @ T_sdk_flange_from_hand_root_m`；该标定仍标记 `accepted_for_metric_use=false`，所以只能作为可复现起点，不能声称是真实装配真值。

### 1.2 转换与验收

1. 在项目的独立 staging 目录生成左右手 URDF（或直接生成规范化 USD），逐项保留 JSON 的 joint name、parent/child、origin、axis、limit 和 canonical 20D 顺序；为每个 link 明确质量、质心和惯量来源。未知项用“估计值 + uncertainty”记录，不能静默使用 importer 默认密度。
2. 分别转换左臂、右臂、左手、右手为版本化 USD，再在一个 assembly USD 中以固定关节连接 Link7 与手根。不要先把整机烘成不可检查的单层 USD。
3. 使用 Isaac Lab/Isaac Sim 6.2 的 importer 3.0；显式设定输出目录、固定基座、position target、force drive，并保存 converter 配置。官方 importer 支持 fixed-joint merge、visual 生成 collision、凸包/凸分解、自碰撞、密度和刚度/阻尼覆盖，见 [URDF 导入教程](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/import_urdf.html)。
4. 首次转换时 `merge_fixed_joints=false`，以便审计 27-link 手图和安装链；通过 link/joint/FK 对照后再评估 merge。手指 visual STL 不能直接当动态三角网格碰撞体：对近似凸的指节用 convex hull；掌部和复杂凹形件用少量凸分解或手工 primitive。视觉网格与碰撞网格分层保存。
5. 建立资产门槛：54 个可控关节恰好且顺序唯一；零位和 20 组随机限位内姿态的 link transform 与现有 FK 对齐（建议平移 max <1 mm、旋转 max <0.5°，作为工程门槛而非真实标定精度）；质量、惯量正定；无 NaN；静止 10 s 不发散；关节限位、力矩限位与 URDF/控制器记录一致。

### 1.3 碰撞、接触和抓取摩擦

- 碰撞过滤先复刻 MuJoCo 的意图：机器人与桌/瓶/箱接触；默认关闭同一手内部相邻指节、手-所属前臂以及不需要的双臂互撞，再用 allow-list 逐步打开掌-指、非相邻手指自碰撞。完全开启 54-DoF 自碰撞会显著增加接触数，并可能因初始网格重叠导致爆炸。
- PhysX 使用 TGS、较小 physics dt（MVP 建议 1/240 s，4 个子步对应一个 60 Hz drive tick；15 Hz 策略目标保持 16 个 physics steps），位置迭代先 8、速度迭代先 2，再按穿透和抓取稳定性调。contact offset 应仅略大于 collision margin，rest offset 接近 0；CCD 只给高速、薄壁或经实测会穿透的瓶/箱壁开启。参数应通过扫参锁定，不能逐案例手调。
- 物理材质分组：指腹、非指腹手壳、PET/瓶盖、桌面、箱底/箱壁。初始扫参可取指腹—瓶 dynamic friction `0.6/0.9/1.2`，static friction 为 dynamic 的 `1.0–1.25×`，restitution `0–0.05`；这是搜索范围，不是实测值。摩擦 combine mode 固定并记录。MuJoCo 当前全局 friction 为 `0.9`，手部代理 `1.2`，瓶碰撞体 `0.75`，只能作为初始化参考，因为 PhysX 与 MuJoCo 的摩擦锥和求解器不可直接等价。
- 用三项小测试标定：斜面滑动（瓶-桌）、双指准静态夹持（最小法向力/滑落）、抓持后垂直与横向扰动（保持时间和位姿漂移）。真实瓶若没有质量、表面和挤压数据，摩擦只能报告为 sensitivity band。
- PET 瓶 MVP 仍用刚体复合几何（圆柱+肩部+颈+盖，质量 0.42 kg 是现有仿真假设）。不在 MVP 引入薄壳变形、液体晃动或软指腹；这些效应是灵巧手保真度的主要上限。

## 2. 场景、相机与随机化

### 2.1 场景基线

先复制 `scene_teleop_viewer_task.xml` 的可比基线：桌面约 1.3×1.1 m、台面 z=0.55 m；箱中心约 `(0, 0.78, 0.575)` m、内腔约 0.26×0.20 m、壁高约 0.10 m；瓶中心约 `(-0.32, 0.73, 0.655)` m。上述尺寸与位置尚未对真实采集场景做度量标定，必须标注 `sim_baseline`，不能冒充 real-scene twin。桌、箱、瓶各自独立 USD，便于质量/摩擦/位姿随机化和 ground-truth 查询。

### 2.2 顶部相机

真实顶部相机是 Orbbec Gemini 335L，640×480；工厂内参 `fx=366.53265, fy=366.59036, cx=321.33359, cy=238.58607`，Brown-Conrady 系数 `[-0.0286315, 0.0335283, 0.000352739, -0.000154852, -0.0118582, 0, 0, 0]`（OpenCV 顺序 k1,k2,p1,p2,k3,k4,k5,k6）。Isaac 6.x 相机应使用原生 OpenCV pinhole/lens distortion schema，不用旧 fisheyePolynomial 近似；5.1 文档已说明原生 OpenCV distortion 支持，[相机标定接口](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html#calibration-and-camera-lens-distortion-models)。

外参必须从 `camera_calibration_20261015/.../calibration.json` 的坐标约定显式换算为 USD/Isaac 相机坐标，不能直接复制 MuJoCo `pos/xyaxes/fovy`。E46 的 `camera_delta` 为 identity，采用的是既有标定；但原始 factory intrinsics 文件自身写明 `robot_extrinsics=null`，机器人基座外参来自另一份 hand-eye 结果。A 基座姿态子集敏感性高达 65.71 mm/6.57°，B 为 5.56 mm/0.90°，两者均 `accepted_for_metric_use=false`。因此相机验收采用：

- 先渲染 640×480，再完全复用真实管线 crop rows 96:480、缩放到 320×192/策略 224×224 的像素中心约定；不得用只匹配视场角的相机代替 K+D。
- 用 9 个 E46 留出 episode 的 measured q 回放，比较掌点/机器人 silhouette。E46 只证明掌点在可见同分布姿态下可作 2D ROI 软先验：A/B 掌点严格入 mask 约 98.2%/97.8%；原 URDF 指尖投影未通过，米制高度 proxy 未通过。因此不能用指尖像素或 E46 高度作为相机验收真值。
- 在获得独立桌面/重力/已知高度标定板之前，相机—桌面绝对 3D 只列为假设；MVP 成功判据使用 simulator oracle 几何，不用 E46 推导高度。

### 2.3 随机化分层

先固定基线完成单因素验收，再分层随机化，防止用随机化掩盖建模错误：

1. 几何/动力学：瓶 XY ±20–40 mm、yaw 全角、质量 ±15%、瓶/指腹/桌摩擦范围、箱 XY ±10 mm；相机位姿只在标定 uncertainty 内随机，且 A/B 外参不确定性单独报告。
2. 渲染：环境光强/色温、主光方向、桌/箱/瓶材质色与粗糙度、曝光/白平衡、轻微 blur、传感器噪声、JPEG/resize；保留一个无随机化“诊断域”。
3. 控制：关节零偏、target latency 0–2 个 15 Hz frame、驱动 gain/effort ±10–20%。随机化区间最终应由真机日志或系统辨识收缩。

每个 seed 保存全部 sampled parameters；评测至少分 nominal、physics-shift、appearance-shift、camera-shift、combined 五组，不能只给混合平均值。

## 3. 54D、15 Hz 绝对关节控制接口

建立一个唯一的 `JointContract` 表，列出 54 个 canonical index、策略 joint name、USD joint path、方向符号、零偏、上下限、effort/velocity limit。启动时从 articulation 读取 joint names 并做双向一一匹配；任何缺失、重复或顺序变化都 fail fast，禁止依赖 USD traversal order。

策略每 `1/15 s` 给出绝对 `q_target[54]`。闭环 adapter 的顺序为：读取最新 simulated measured q → 调策略/候选生成器 → 若模型内部输出 residual，则只在此处用 fresh q 重建绝对 target → 限位与每步速度/加速度限制 → 写入 articulation position target → physics 子步执行 → 读取新的 q/dq/RGB。手臂和手均使用 position drive；不把 15 Hz target 当瞬时 teleport，也不直接写 joint state。

MVP 先用 Isaac Lab implicit actuator/PhysX continuous PD。按组设置而非一个全局 gain：A/B 臂、拇指、其余 MCP flex、MCP abduction、PIP/DIP。力矩限制取 URDF/真机控制器记录；手部未知限制必须标 `TBD`。增益标定按以下顺序：

1. 无接触阶跃/正弦跟踪，比较真实或现有 MuJoCo 的 rise time、overshoot、稳态误差和 15 Hz target 间轨迹；
2. 用 `Kp = tau_limit / e_sat` 给出不会长期饱和的起点，`Kd` 从临界阻尼附近向下扫；
3. 接触抓取中检查 actuator effort、饱和占比、穿透和 chatter；
4. 若有真机电机延迟/摩擦数据，再换 explicit actuator 或加入 delay/friction。Isaac Lab 将 implicit actuator 视为物理引擎内连续 PD，explicit actuator 可表达物理或学习电机模型，见 [actuator 文档](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.actuators.html)。

日志必须同时存 commanded target、simulated q/dq、applied/estimated effort、limit clipping、contact count/force、policy time 和 physics time。15 Hz 仅是控制频率，不代表模型推理能在 66.7 ms 内完成；离线评测同时给 wall-clock p50/p95 和 real-time factor。

## 4. 观测与顶部 RGB 域差评估

策略输入严格只取顶部 RGB，不启用腕部相机。渲染按真实 640×480 K+D，随后复用训练预处理与两帧因果历史；相机 capture time、q sample time 和 action apply time分别记日志。E46 的 `lag=1` 只表示离线 `image[t]` 对 `q[t-1]` 的 fit-only 选择，projector 本身不移位；Isaac 中应显式模拟/消融 0、1、2 帧延迟，不能把候选动作整体平移。

域差评估分三层：

- 像素/成像：同姿态回放的 crop 覆盖、silhouette IoU、边缘 Chamfer、颜色直方图或 ΔE、亮度/对比度、blur/noise 统计；透明 PET 的反射/折射单列失败案例。
- 冻结视觉编码器：对配对或近配对真机/Isaac 顶视帧计算 cosine distance、CKA/embedding distribution distance，并按手姿态、遮挡、瓶位置分层。这里只评估 frozen official LeWAM encoder，不训练 domain adapter。
- 行为：同一 q/history 下分别输入 real frame、nominal render、randomized render，比较 54D target 的 MAE/max、限位触发率、候选排序一致率，再跑闭环阶段成功率。行为差异是最终判据，视觉距离不能替代任务指标。

配对基准优先用 EP056 与 9 个 E46 留出 episode 的 measured-q replay；由于真实场景物体和仿真桌/箱未完全度量对齐，报告中区分 robot-only mask 与 full-scene 指标。建议门槛由 real-real（相邻帧/重复采集）噪声底和 MuJoCo-real 当前差距共同定义，不先拍脑袋给单一 FID 阈值。

## 5. 成功判据与指标

所有物理成功判据由 simulator ground truth 计算，视觉策略不可读取这些量；oracle planner baseline 可以读取，但报告必须显式标记 `oracle-state`。

阶段状态机与建议门槛：

1. `reach/contact`：目标手进入瓶周围，拇指与至少两个非拇指形成有效接触；只作诊断，不算抓取成功。
2. `grasp_latched`：上述接触组合持续 ≥0.2 s，瓶横向漂移/倾角未超过安全阈值，且非桌面支撑不能单独满足条件。
3. `lift_success`：瓶底离桌 ≥25 mm，持续 ≥0.5 s；与当前 MuJoCo short-lift 口径对齐，并同时报告最大 lift、滑移和姿态变化。
4. `transport_success`：lift 已成功，瓶中心进入箱口上方的内缩安全区域，且过程中未掉落。
5. `place_success`：瓶的碰撞体/质心投影在箱内安全边界内，瓶最低点低于箱口、瓶与箱底或内部接触、手已释放，随后 1.0 s 内不弹出；不能只用 XY inside。
6. `task_success`：依次满足 lift 与 place，未发生超时、严重自碰撞/桌撞、关节超限或对象出界。

主指标：episode task success；阶段条件成功率 `P(grasp)`, `P(lift|grasp)`, `P(transport|lift)`, `P(place|transport)`；另报 Wilson 95% CI。辅助指标：完成时间、重规划次数、掉瓶率、最大瓶倾角、瓶/箱冲击、手指滑移、关节限位/effort saturation、碰撞违规、平均/峰值接触力、real-time factor。每个设置至少 30 seeds 做开发诊断，冻结后 100 seeds 做正式对比；同 seed、同初态做 MuJoCo/Isaac 配对，报告成功率差和逐阶段 disagreement。

## 6. 与模型的 receding-horizon 接口

沿用当前 planner 的边界：LeWAM 生成动作 chunk；oracle 版本从 Isaac 读取瓶 pose/velocity 与箱几何，生成 object-directed IK 和保守 finger-closing variants；每个候选从同一 state snapshot rollout，按阶段代价打分；只执行最优候选前 `k` 个 15 Hz target，再采新 RGB/q/dq 重规划。

开发时比较 `k∈{1,2,4,8}`：`k=1` 最接近现有 MuJoCo 每帧重规划，反馈最好但推理/rollout 开销最大；较大 k 可提高吞吐，但接触阶段更脆弱。建议按阶段调度：approach 可用 2–4，首次接触/闭手/lift/place 用 1；正式评测前在 dev seeds 冻结，不能按测试结果选择。候选 rollout 必须 deep-copy/restore 所有 articulation、rigid body、drive target 和随机数状态，避免候选间污染。

接口输出统一 JSON/NPZ schema：版本、seed、初始场景、每轮 observation timestamps、fresh q anchor、proposed chunk、candidate scores、selected id、执行 k、oracle state、阶段事件和终局指标。这样同一上层 planner 可切换 `MujocoOutcomeModel` 与 `IsaacOutcomeModel`，不修改 LeWAM proposer；未来学习后果模型也复用该 contract。

## 7. 硬件、环境与当前集群可行性

只读检查到 `<gpu-server>` 为 Ubuntu 24.04.3、8×NVIDIA H20 96 GB、driver 580.126.09、约 1 TiB RAM，Docker 可用；`/mnt` 余量约 4.2 TB 但已使用 96%。未发现现成 Isaac/Omniverse 镜像或项目内容；`/mnt/workspace/colleague-c` 与 `/mnt/workspace/colleague-e` 本次仅做目录级只读检查，不应作为安装位置或可写缓存。

当前结论是：**CPU/RAM/VRAM/OS 足够，但该 H20 集群不能认定能运行 Isaac Sim RTX 顶部 RGB，因此当前不能确认满足本任务。** 官方 Isaac Sim 6.2 x86_64 最低项列 RTX 4080、16 GB VRAM、Linux driver 595.58.03，并明确“无 RT Core 的 A100/H100 不支持”；H20 是 Hopper 数据中心卡，官方 requirements 没有把 H20 列入支持矩阵。driver 580.126.09 也低于当前文档的 595.58.03。这里对 H20 的不支持判断是基于产品族和官方缺席的推断，最终必须以 6.2 容器的 Compatibility Checker 与最小 RGB render smoke 为准，见 [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)。本调研按要求没有 pull 镜像或占 GPU，因此未做该验证。

推荐两种隔离方案：

- 首选：有受支持 RTX GPU 的独立工作站/云机，固定 `nvcr.io/nvidia/isaac-sim:6.2.0@sha256:...`，在镜像内安装与其匹配的 Isaac Lab commit/wheel；项目与输入资产只读 bind mount，输出、shader/cache、Kit config 全部挂到 `/mnt/workspace/dengjianfeng/.../runs/isaac_*` 下独立可写目录。官方容器流程和 compatibility command 见 [container installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)。
- 备选：若集群管理员确认 H20 支持并允许使用，先在隔离容器内单卡运行 compatibility checker 和 640×480 RGB smoke；失败即停止，不改主机驱动。driver 升级属于管理员操作，不能为本项目直接修改 GPU 主机系统环境。

不建议把 Isaac pip 包装进项目现有 CPU/训练 venv，也不在系统 Python、`/usr` 或共享他人 workspace 安装。容器无法使用时，可在本项目专属目录建立 uv/conda 环境，但 Isaac RTX 渲染仍依赖兼容主机 driver/GPU，环境隔离不能补足硬件能力。

## 8. 工作量、里程碑与 MuJoCo 对比

以下为 1 名熟悉机器人仿真的工程师估计，不含真实 CAD、惯量/摩擦实验和集群管理员排期：

|里程碑|交付物|估计|
|---|---|---:|
|M0 环境闸门|固定版本/digest；Compatibility Checker；headless RGB smoke|0.5–1 天（若需换硬件则外部阻塞）|
|M1 资产|双臂 USD；Wuji URDF/USD；54D/FK/limit/inertia/collision 验收|4–7 天|
|M2 场景与相机|桌/瓶/箱；真实 K+D；外参/crop 回放对齐；nominal render|3–5 天|
|M3 控制与接触|15 Hz adapter；PD/limit；摩擦小测试；稳定 grasp/lift|5–8 天|
|M4 闭环 MVP|oracle receding horizon；阶段状态机；日志与 30-seed 报告|4–7 天|
|M5 对照评测|随机化；100-seed MuJoCo/Isaac 配对；RGB/encoder/行为域差|5–8 天|

可交付 MVP 约 3–4 人周；较可信的对照报告约 5–7 人周。若缺手部质量/惯量、真实摩擦、装配 CAD 或 RTX 机器，日历时间会更长。

|维度|MuJoCo 当前主线|Isaac Sim / Isaac Lab 备选|
|---|---|---|
|部署|CPU、轻量、现有闭环已接通|依赖受支持 RTX/driver，环境较重|
|物理迭代|单环境和状态 snapshot 快，调试简单|PhysX/GPU 并行、接触传感和批量随机化强|
|视觉|当前 OpenGL/OSMesa 外观较简化|RTX 材质、光照、OpenCV 镜头模型和 Replicator 更适合域差研究|
|资产|现有 MJCF 代理已工作|USD 资产管线更规范，但 Wuji 需新增转换与大量验收|
|灵巧抓取|同样受碰撞、摩擦、刚体瓶假设限制|不会自动更真实；求解器差异可提供有价值的 sim-to-sim 压力测试|
|本项目角色|首选 outcome model 与闭环基线|异构复核、视觉对照、规模化随机化；通过后才考虑扩大用途|

## 9. 主要风险与处置

1. **灵巧手抓取保真度**：手质量/惯量、力矩限制、指腹顺应性和摩擦均未实测；20+20 DoF 会放大接触参数误差。处置：先通过斜面、夹持、扰动三项微基准；报告摩擦/gain sensitivity，不只报最佳参数。
2. **手安装与相机外参**：E46 只支持掌点 2D 软先验，指尖和高度未验收，A 基座外参不稳定。处置：仿真中区分可复现假设与 metric truth；补采已知高度板、桌面/重力和可见指尖人工点后再升级 digital twin 声明。
3. **sim2real 外观差**：透明 PET、反射、曝光、遮挡与真实背景难复刻。处置：real/sim 配对回放 + frozen encoder/动作差异 + 分层随机化；不以“看起来像”作为验收。
4. **sim-to-sim 不一致**：PhysX 与 MuJoCo 的摩擦、接触和 drive 定义不同。处置：使用同一初态/动作、逐步 state/contact 对照；把 disagreement 当 sensitivity 信号，不强求参数数值相同。
5. **硬件不可用**：H20 很可能不满足 RTX renderer，当前 driver 也低于文档测试版本。处置：M0 是硬闸门；失败则转有 RTX 的机器，MuJoCo 主线不受影响。
6. **并行环境造成假加速**：顶部 RGB 和 54-DoF 接触可能先受 renderer/VRAM/solver 限制。处置：MVP 单环境；先测 1/4/16 env 的 FPS、RTF、VRAM，再决定是否批量。
7. **评测泄漏**：用测试 seeds 选择 k、摩擦或随机化范围会高估成功。处置：dev 冻结配置和阈值；正式 seeds 一次性评测并保留完整 manifest。

## 10. 最小可行版本（MVP）分步实施清单

1. 冻结 Isaac Sim 6.2 镜像 digest、Isaac Lab 3.0 commit/tag 与驱动要求；在候选 RTX 主机执行 compatibility checker 和单相机 headless smoke。
2. 将左右 M6 URDF 转 USD；从 Wuji JSON 生成左右手可审计 URDF/USD；生成 canonical 54D joint map manifest。
3. 以 20 组姿态做 arm/hand FK、零位、axis、limit、安装变换对照；完成质量/惯量正定与 collision 可视化审计。
4. 建立单环境 tabletop USD：固定桌、刚体瓶、开放箱；先不用随机化。
5. 配置顶部相机的 640×480 K、Brown-Conrady D、标定外参和 rows 96:480 crop；用 EP056 measured-q replay 做 robot-only 视觉对齐。
6. 接通 54D 绝对 position targets：1/240 s physics、15 Hz policy；无接触跟踪调 gain/effort/limit，保存 q/dq/effort。
7. 做瓶-桌斜面、双指夹持、抓持扰动微基准；冻结一组 nominal friction/contact 参数和两个 sensitivity 变体。
8. 实现 oracle 状态机与成功判据；先用 scripted approach-close-lift-place 跑通，避免把资产/控制错误归因于 LeWAM。
9. 接 LeWAM proposer 与候选 rollout，先 `k=1`；每轮从 fresh simulated q 与两帧顶部 RGB 重规划，只执行首帧。
10. 运行 30 个 nominal seeds，输出阶段成功率、失败分类、RTF、limit/contact 指标；与相同 seeds 的 MuJoCo 配对。
11. 通过 nominal 后才加入 physics/appearance/camera 单因素随机化；在 dev 上选择 k 与范围并冻结。
12. 运行 100-seed 正式对照，给出 MuJoCo/Isaac 的阶段 disagreement、成功率差、视觉 encoder/动作差异和 sensitivity band。若 grasp/lift 对摩擦或 gain 极敏感，结论限定为接口压力测试，不宣称 sim2real 抓取能力。

## 11. 本次核实来源

- 项目内：`PROJECT_DIRECTION.md`；`AGENTS.md`；`sim/mujoco_pick_place_poc/` 的 README、场景、资产 manifest、控制与 oracle planner；`notes/2026-11-04-e46-recalib.md`；`runs/e46_recalib_20261104/` 的 adopted parameters 与标定限制。
- 官方文档核查日期 2026-11-05：Isaac Sim 最新文档显示 6.2.0 容器和 requirements；Isaac Lab main 显示 Python 3.12、Isaac Sim 6.2 optional extra、URDF importer 3.0。未来落地时必须重新核对版本，因为 `latest/main` 会变化。
- 未核实/待实测：H20 是否能通过 Isaac 6.2 Compatibility Checker；Wuji 真机质量/惯量/力矩/摩擦；真实桌/箱/瓶精确尺寸和位姿；相机—桌面绝对高度；真机控制增益与延迟。上述项目均不得在报告中默认为已知。
