# E80：E79 统一架构预训练数据与权重盘点

日期：2026-11-12。状态：报告完成；区分已核证据、未核缺口与未来建议。

磁盘快照：开始 /mnt 97% / 约3.2TiB可用；交付前复查98% / 约3.1TiB可用。共享盘动态变化，按最新快照规划；本轮仅写约62KB报告。

## 范围与证据标准
只读核查服务器与公开网页，仅本报告写入。无训练、GPU 使用、数据/权重下载或大文件复制。同事目录只读。仿真/retarget 不推荐参与训练；模型标注不作为测试真值；冻结评估集不加语言标签；不平移动作或标签；天机仅顶视，Piper 可用腕相机。未核实的数字/许可不视为已确认。

## 上一轮线索（待本轮核验，不当作最终结论）
- colleague-e/wuji_bottle50_20261012：50 条、30Hz、38,999 帧；与本项目 93 条原始采集 ID 无交集。colleague-d 三相机包来自同一批，不能重复计数。
- colleague-a/Datasets/WuJi（含 posttrain/data）：纸巾、积木、挤纸巾、笔帽、试管等；约 100 条 10 月 24–25 日数据。标定指向同一顶视相机序列号，画面像同一白色双臂+黑色 Wuji 手，跨日构图变化，旧裁剪不能直接沿用。
- 共享目录线索：Cosmos3-DROID、AgiBotWorld-Beta、OXE、RoboDojo_real_lerobot、egoscale、egoscale_probe；Bridge 下载不完整；RoboMIND 部分无读取权限；colleague-c/datasets/EgoDensePrompt 存在。Wuji retarget/仿真链排除。
- /mnt 使用率约 97%，可用约 3.3TB，待刷新。

## 1. 天机同机存量

**结论：有。当前核实 435 条可定位的新真实采集，顶视播放/采集时长约 2.88 小时；不是数十小时。** 另有魔方原始第43条（处理版剔除）待检查，可使原始录制上界到436。这里“未使用”指未进入本项目93集/fit64，不表示从未被同事训练使用。以下各行选一个版本计数，train/val也只是同事划分；未来需重新按本项目隔离规则取允许部分。

令 `H=/mnt/workspace/colleague-a/Datasets/WuJi`，`D=H/posttrain/data`，`B=H/posttrain_data`。

|真实来源及读取版本|条数；帧数；时长|采集日期；相机|动作/状态与结果标注|建议|
|---|---|---|---|---|
|colleague-e/wuji_bottle50_20261012/training50.tar；已有解包视频 `B/wuji_bottle50_three_camera21_executed_allframes_20261014/packages/*/head.mp4`|50；38,999；0.3611h（21.67min）；顶视MP4合计0.280GB；原tar 1.966GB|原始ID为2026-10-12；50个容器均640×480、30fps|raw54有动作/反馈；相机转换版含测量/FK衍生坐标，不能直接充当E79的54D命令；未核到逐条人工成败。旧转换manifest顶视重复帧计4,935（12.65%），实际独立曝光少于存储帧数|最高优先P1；与93原始ID交集为0。任务是瓶入纸杯托，与瓶入箱相近但不同|
|`D/episodes100_20261025_flange_tcp_command58_state58_absolute_chunk32_val5_groupequal`|原始100，处理99；47,018；0.4354h；顶视0.239GB|10月24–25日；640×480/30fps|魔方入托盘；原始q_cmd/q_fb，处理58D=双臂TCP位姿18+双手40；原始ep43因无有效32步未来窗口被剔除，不等于失败|P1优先；99包已核，ep43仅在raw复核后补入短时距视频池|
|`D/episodes02_20261025_flange_tcp_command58_state58_absolute_chunk32_val5_groupequal`|50；23,459；0.2172h；0.124GB|10月25日；640×480/30fps|取纸巾；有命令/反馈及58D转换；未核到独立人工成败标签|P1优先，重视拉扯/释放的真实变化|
|`D/stack_building_block_20261026_flange_tcp_command58_state58_absolute_chunk32_val5`|62；34,207；0.3167h；0.167GB|10月26日；640×480/30fps|积木堆叠；有动作/反馈；自动停止手势不是成功真值|P1优先，接触/遮挡与放置变化|
|`D/squeeze_tissues_20261027_flange_tcp_command58_state58_absolute_chunk32_val5_groupequal`|53；52,134；0.4827h；0.183GB|原始ID为10月26日，目录日期是处理日期；640×480/30fps|挤纸巾；命令/反馈，原MP4时间须用sidecar而非假定CFR；未核逐条成败|P1次优，手部形变有用，任务不同|
|`D/pen_cap_20261029_flange_tcp_command58_state58_absolute_chunk32_val5_groupequal`|34；34,475；0.3192h；0.172GB|10月29日；640×480/30fps|笔帽装配；命令/反馈，原JPEG wall_ns；val1/all34版同源不加量|P1次优，顶视接触细节可见性有限|
|`B/tube`；对应 `D/tube_20261105_flange_tcp_command58_state58_absolute_chunk32_val5_groupequal`|50；原顶视JPEG 38,517张、约3.000GB；首末时间戳跨度总1,297.47s=0.3604h|11月5日；抽查640×480，目标30Hz；有效约29.69图/s，不能把采样网格当逐曝光精确30Hz|试管盖取放；原始marvin/Wuji动作及状态JSONL俱在。处理包没有head.mp4，需读原JPEG；有COMPLETE/停止记录，无独立结果真值|P1次优，直接引用JPEG；不必先打视频/复制原图|

**同一设备的证据强弱。** 本轮CPU抽取并查看每组一帧，与93集原顶视裁剪对照：均为白色双臂、黑色五指Wuji手、相同白色桌面及房间布局；不同日期桌面位置/取景确有变化。标定文件 `B/wuji_head_camera_calibration_20261014/20261014_rgb_factory_intrinsics.json` 与bottle50的 `calibration.json` 指向顶视序列号 **CP28563000JZ**，640×480，fx=366.53265、fy=366.59036、cx=321.33359、cy=238.58607；腕相机另为CV2F861000F5/CV2F86100083。但 `factory_file_audit.json` 明确 **historical_capture_metadata_serial_recorded=false**、**user_confirmed_same_head_serial=true**；`plan.json` 的 original_capture_serial_verified=false。所以结论是“标定沿用＋既往人工确认＋本轮画面高度一致”，**不是每条原始录制均携带序列号的硬证明**。后续任务合同也写 reused calibration；同一实体臂的资产编号未找到。采集人需确认设备及相机未更换，跨日应重新在训练帧人工确认ROI；绝不直接沿用93集 rows96:480裁剪。

**去重及许可。** 本轮读取50条tar内的小conversion manifest（不解包数据），与93条conversion manifest比原始采集ID，交集0；六个后续任务manifest与93、bottle50及彼此也交集0（合计398）。这仅排除声明ID重复，未全量图像哈希或排除重命名复制。同一相机包、head/three-camera、rawjoint/EEF/Link7/optical、train/val、all34等全部按原始ID合并。colleague-d线索中的三相机数据沿用这批50条，未发现可独立计量的新实录；其tianji/extracted是机器人URDF资产，LAM/retarget/native replay等不计入。H/PATH_MAPPING.json记录数据由colleague-b历史路径迁入，2647条符号链接已重定位，不能按历史绝对路径与当前路径双计。

同事数据未发现授予本项目训练/发布的独立许可文本：须向贾天才、何海森及原始采集/所有者（colleague-b、相关天机采集团队；colleague-d若提供副本亦确认）取得用途同意；读取可见不等于训练授权。本轮不联系任何人。停止手势、COMPLETE、转换passed均不等于人类成功标签；P1可用失败与恢复，P3成功BC不可据这些字段自动收录。

**搜索边界。** 已针对上述目录、colleague-d相关入口及共享数据根作有限层级核查；colleague-b/data所见主要GenieSim/RoboCasa线排除；colleague-c/RealMachineData实际为已有optical93派生，不加量。未穷尽全服务器所有人的原始录像、空闲/调试片段。现有证据不能宣称没有更多数据。原始zip的自动停止尾段已经被采集侧删除，不能假定服务器还能恢复。下一优先动作是让采集者提供未筛选session索引，而非全盘扫文件。



### 1.1 后续定向核查新增：10月9日37条真实积木数据

`/mnt/workspace/colleague-e/data/lerobot/local/tianji_wuji_raw54_30fps_after_20261009_165437` 实有37个Parquet，共2.143GB；meta给出42,000帧、30Hz、640×480三视图、54D动作/反馈，任务“Put the blocks in the box.”，播放时长0.3889h。本轮小批读取一张嵌入顶视JPEG并查看，白色双臂/黑色Wuji手/工作台与后续集合一致，场景构图不同；该包未核到逐episode相机序列号或人工成败。conversion_manifest有37个20261009原始ID，与93、bottle50交集均0，与10月24日以后任务日期不重叠。因此新增总量修正为**435条、约2.88h**（含试管真实时间戳跨度；未经成败筛选和评估session隔离）。优先级与bottle50同属首批P1，须贾天才/采集者授权，不使用旁边VAE/retarget衍生标签。原始索引重复检查尚不等于全量像素去重。

## 2. Piper 存量

以下数量由本轮读取原始HDF5的action长度、旁路JSON和LeRobot info重新核对；未逐段观看/重新判定全部结果。原始HDF5有些只放状态，真实MP4在相邻 `aloha_mobile_dummy/video`；不能因为没有 `observations/images` 就说没有录像。每相机流不能重复计为额外交互小时。

|集合（H=/mnt/workspace/colleague-a）|本轮盘点|与当前训练的关系/用途|
|---|---|---|
|`Datasets/piper_processed/demo4_independent_rebuild_20261023/dataset`|173片段、211,571帧、1.959h；E77核154父组|现有数据，不是新增。原173来源232,630帧，约多11.45min源时间跨度；恢复真实停顿是P1增量，不能跨拼缝当均匀时间|
|`Datasets/piper/piper_raw_incremental/raw/demo4_teleop_realsense_v2_hdf5`|85条，170,978帧；79成功164,344帧＋6失败6,634帧|6失败约3.69min可供P1；成功79基本已在rebuild|
|同根 `demo4_dagger_realsense_v2_hdf5`|96条，70,585帧；94成功68,286帧＋2失败2,299帧|2失败约1.28min；94成功接管已用。父失败的成功纠正段不算失败BC|
|同根 `demo4_inference_realsense_v2_hdf5`|136条HDF5、94,272帧、0.8729h，结果None/unreviewed|首要新增P1；先关联parent/session，和接管同次过程不可独立计数。E77曾核90个非空父ID，本轮更广sidecar有额外记录，不能把sidecar数量当HDF5数量|
|`demo6_raw_20261030` 与processed训练分辨率版|原35条149,732帧；33成功147,922帧，2失败1,810帧；处理失败1,694帧（0.941min）|成功处理33条147,607帧已计入总206片段/3.326h；两失败新增极少，E77指出ep32纯静止|
|incremental/raw下 `demo3_teleop_realsense_v2_hdf5`|50条：36成功98,651帧；9失败20,261帧；5未知11,584帧；合1.2083h|demo6外的相关任务资产，优先核任务/parent及重命名；成功不必都加入P3，P1可含失败|
|同根 `demo3_dagger_realsense_v2_hdf5`|94条：88成功74,221帧；6失败6,057帧；0.7433h|恢复段有价值；需与demo3 inference父链合并，不当94次独立重置|
|同根 `demo3_inference_realsense_v2_hdf5`|143条、134,629帧、1.2466h；未人工复核结果|P1高优先候选；约50个父ID量级（E77式字段定义需统一），按父链取片|
|demo3 legacy：`demo3_teleop_hdf5 / demo3_dagger_hdf5 / demo3_inference_hdf5`|16/1/8条；50,160/1,174/24,421帧；合0.7014h|旧相机/时钟合同弱于RealSense版；暂作第二批，不直接与新合同P2混合|
|`demo2_openpi_raw28_20261009`|28条、15,501帧、0.1435h；成功未知；任务Insert the test tube|形状14D、30Hz，有反馈与动作；早期三视图。21条filtered只有12,065帧，是筛选版，不另计|
|`/mnt/workspace/colleague-e/data/lerobot/local/piper_bimanual_h50`|30个Parquet、8,881帧、15Hz=0.1645h、0.495GB；640×480三视图、14D；任务笔入笔筒|9月2日前后早期数据，目录h50不是50条。meta声明leader动作，与当前follower next-state不同；同一实体设备/相机尚未证明。原raw多条绝对符号链接已失效，Parquet实际存在；0.469GB tar.gz不再解包|
|同事 `piper_demo34_threading_stage_20261021_v1` / `piper_sft_center80_224_coarse132_20261021`|132切片、153,339帧=1.420h；来源READY明确demo3 dagger13、teleop35，demo4 dagger5、teleop79|从colleague-a demo34原始集合切出，**不是新增独立数据**。54个rejected proposals是阶段筛选拒绝，不等于54次失败；应回原raw读，不重新计量|

**同机/相机核查。** 本轮分别抽取demo3 teleop/dagger、demo4 teleop/dagger的HDF5 attrs，四者从臂序列号一致：左 `003D00204648571520303731`、右 `002E001A4648571020303731`；相机SDK序列号顶 `262322075245`、左腕 `262422074340`、右腕 `262422074159`。这提供比目录名更强的同机证据。demo6抽样SDK序列号为顶 `262022070197`、左腕 `261922074380`、右腕 `262322073587`，**不是同一组三相机**。应保留日期、分辨率、黑边、曝光偏差和相机ID分组，不把demo6标为相机完全未变。demo4/early原三路640×480、顶视有效640×360；demo6 native顶1280×720/腕640×480，训练版顶448×252/腕224×224，30Hz（E77已有容器核验）。

**可获得的量与限制。** demo4尚未使用的rollout+8条失败+源停顿约1.15h；demo6失败不足0.02h；demo3列出的存储播放时长合约3.90h。故除现有3.326h成功集外，有约**5h相关存量候选**，但这不是去重、完整视频、parent隔离后的可训练净小时：demo3→demo4重标、父子片段重叠、旧版复制会进一步减少；先按3–5h新增上界计划，净量必须通过allowlist实测。本轮未完成所有视频逐帧完整性核验和全部父链重建，不把该上界写成确定收益。

**不得相加的目录。** demo34_collection_raw_20261020_2152是incremental的旧快照；`.task_relabels/demo3-to-demo4-20261019-episodes63-67`及其staged是显式重标副本；`.views`、legacy/realsense v2.1/v3.0、micro/fiveview、align/ft/newdagger、demo6 nobars/448/原生版均须追溯原文件/父ID。本轮扫描确实遇到relabel副本，以上推荐计数已避开它们。demo3_success_micro与fiveview同为126片段205,123帧，不能加两次。

**动作与许可。** 当前colleague-a原action是同帧follower反馈，processed是原始t+1反馈目标；沿用现有合同，不再平移、不伪称motor command。失败原记录可做P1，不全量当成功动作示范；经过人类确认的恢复段才候选P3。H目录和贾天才数据需要所有者用途同意。未找到更多可直接确认、独立于这些来源的同事Piper任务集；共享RoboDojo另在§4，且本地robot_type=dual_x5，不能因OpenWAM权重名含Piper就断言同本体。



### 2.1 补核：另一批独立的9月5日Piper笔任务

`/mnt/workspace/colleague-e/data-pen/lerobot/local/piper_bimanual_h50` 本轮info和validation核到31条、23,610帧、30Hz=0.2186h；validation源ID均为20260905，与前述9月2日30条/15Hz包不是同一采集日期。动作/状态14D，三相机；README说明历史左右臂字段曾交换后纠正（物理left=can0/right=can1，图像没交换）。本轮不再改动作/标签，也不直接并入当前Piper P2；先由所有者证明映射、leader/follower语义及相机/实体设备一致。原MP4索引93条已记录；无独立逐episode成败核实，caption/latent目录不作真值。可作为额外P1小候选，和9月2日包合约0.383h，但未进入前述“约5h穿带相关候选”估计。权限同样需贾天才/采集者确认。

## 3. 污染排除规则

**P1没有动作标签仍是训练暴露。** 任何评估录像不得进LW、视觉LoRA、归一化、ROI框拟合、目标缓存训练、P2/P3，不能用“只是视频预训练”绕过隔离。固定原始通用E*无需在本项目评估视频上适配。

|必须排除/隔离对象|执行规则|
|---|---|
|天机E66 holdout16|局部episode编号 `[5,6,9,12,13,23,25,37,39,49,54,57,62,78,80,81]`，以E66 manifest为准；下表给原始ID|
|天机历史frozen9| `[7,18,38,56,64,66,77,88,90]`；不加语言标签，不读入预训练|
|天机dev4及fit64以外原93|由93减fit64减上述25得到 `[0,1,2,3]`；首轮P1也不使用dev4。最稳妥：原93只允许E66 A_N64/B_N64那64条；新增集合另经来源审查|
|所有93派生包|optical93_*、colleague-c/datasets/RealMachineData/optical93（DATASET.json明确93集、870个33帧切片）、colleague-c/datasets/tianji_optical93_eef72_15hz、原raw、head/three-camera、224/192×320/腕图、gaze、事件、teacher缓存按源ID并归；同事“all93训练”的模型/LoRA不作清洁初始化|
|Piper已知跨集parent|E39的103(train)/104(val)，110(train)/111(test)/112(val)，155(val)/156(test)必须整父并组；不能把其训练兄弟片段放P1|
|Piper历史proposal留出|dev `[2,20,49,67,73,80,89,103,104]`；test `[0,4,10,38,41,42,45,76,87,110,111,112,123,141,150,155,156,161]`（demo4 rebuild局部ID）。这是proposal_only，不冒称已经生效；一旦采用，应排除它们的全parent/session及所有raw、rollout、接管、gaze、crop版本|
|Piper未来新test parent|先由原始trajectory_id、recording_session_id、test_run_id、controller_job_id、source HDF5标识建关联图，连通分量仅属于一个分区；同一自主尝试→失败→人工接管→成功全部跟随父分区。只有整数episode编号不足以匹配跨目录|
|跨版本/跨任务重标|demo3→demo4 episodes63–67、relabel/staged、.views、贾天才132阶段片段、旧快照均继承原父；任务名改了不等于新数据|
|无法证明来源的录像/模型|隔离区，不进入允许训练名单；对只有近似名称/时间戳的疑似重复，保守整session排除。已有大模型训练范围未知的checkpoint不可当无污染初始化证明|

建立canonical key建议为 `owner/source_collection/raw_capture_uuid/camera_serial/timestamp_range`；已有原文件哈希用于联结（此次未重hash大文件），再用源索引/时间区间查重叠。未来可在自己目录CPU抽稀图像指纹并人工检查疑似近重复；不需要复制大文件。窗口的历史、当前和所有未来帧须位于同一连续segment，禁止demo4跨230处拼缝；E77的3,221个跨缝H50起点剔除规则必须同时用于对照两组。缺帧不插值为真实监督，不平移动作。

新增435条与93原始ID不重叠是有力第一道证据，但不能排除同一天/session相邻重复、重命名副本或同事模型曾见评估集。瓶50、37条及后续任务原始日期均与93的10月16日不同，降低直接重叠风险；如采集者确认有重用源片段仍需剔除。

**评估定位。** E79已认定旧dev/holdout/frozen反复参与研究决策；它们只作探索和历史回归，不再包装成全新确认集。协议、模型与推理模式冻结后另采新日期/摆放/物体session；新测试所有parent从采集起就禁止P1/P2/P3、ROI和统计使用。模型生成事件/成败/评分不作测试真值、阈值或选型依据。

### 3.1 天机具体禁止原始ID清单

|分区|E66 episode|原始采集ID|
|---|---:|---|
|holdout16|5|20261016_195046_1295a8d8|
|holdout16|6|20261016_195117_8c0fa04b|
|holdout16|9|20261016_195306_15646224|
|holdout16|12|20261016_195514_359f5d35|
|holdout16|13|20261016_195556_811e38c1|
|holdout16|23|20261016_200105_9e8fb9ed|
|holdout16|25|20261016_200204_0fb5766a|
|holdout16|37|20261016_201012_331e8ad6|
|holdout16|39|20261016_201140_be67925d|
|holdout16|49|20261016_201807_5a09a66e|
|holdout16|54|20261016_202043_691caebd|
|holdout16|57|20261016_202219_04e131d2|
|holdout16|62|20261016_203029_a39ab750|
|holdout16|78|20261016_203601_811ef0e9|
|holdout16|80|20261016_203815_9c861e8e|
|holdout16|81|20261016_203845_2a48906d|
|frozen9|7|20261016_195200_70d7d465|
|frozen9|18|20261016_195852_dc126b3d|
|frozen9|38|20261016_201104_68aa2e90|
|frozen9|56|20261016_202146_f9f24f40|
|frozen9|64|20261016_203100_0dde9efe|
|frozen9|66|20261016_203133_1fc83c1a|
|frozen9|77|20261016_203534_8cf3be5a|
|frozen9|88|20261016_204347_58db164f|
|frozen9|90|20261016_204511_b6e74b2a|
|dev4|0|20261016_194432_fef90afd|
|dev4|1|20261016_194503_f7b99bc8|
|dev4|2|20261016_194817_42c86061|
|dev4|3|20261016_194845_80159bd7|


## 4. 共享目录与权重

### 4.1 已有数据（本地实物，不把上游全量当本地可用）

磁盘量：`du -sh`为分配占用（G为GiB量级）；GB/TB为十进制逻辑字节或下载receipt。多视图只按同步时刻计一次小时。目录可见、下载完成、视频可解码、许可可用于项目是四件不同的事。

|目录|本地核实规模/磁盘|视角、本体与动作|许可/可用性与建议|
|---|---|---|---|
|`dataset/Cosmos3-DROID`|success 57,639集/18,691,281帧，failure 14,268集/3,721,431帧；合71,907集、22,412,712帧、15Hz=415.05h；707GiB，逻辑758.395GB|Franka Panda单臂；两个外部视图＋腕视，640×360；关节7+夹爪1，笛卡尔6+夹爪1等多动作表示|.download/status=complete，3797/3797文件历史校验完成；本轮未全解码。ModelScope nv-community镜像，[NVIDIA卡](https://huggingface.co/datasets/nvidia/Cosmos3-DROID)标OpenMDW1.1，上游DROID另保留来源/许可。首选外部P1，小量真实桌面/容器任务；不是Cosmos Policy权重|
|`dataset/AgiBotWorld-Beta`|**OpenWAM转换发布，不是原始43.8TB全量**；receipt全量123,175,420帧/15Hz=2,281.03h、约4.005TB；本地仅52/229最终文件，顶层压缩包1.028TB，du1009GiB含partial；status记录下载1.083TB后stopped_low_disk|AgiBot G1双臂、头部/腕部，手型随任务；本地tar.gz未解包，逐包动作与轨迹数尚未核完|源repo OpenWAM/AgiBotWorld-Beta revision35e1dd2；原Beta为CC BY-NC-SA4.0，转换卡不能消除原限制。**不可按52/229线性估小时**；分片task可能不齐。先不解压、不补下载、不列首批|
|`dataset/OXE`|360GiB，逻辑386.346GB；Bridge仅1049/1158文件，109失败、约1.044GB partial；其余4子集下载receipt完整|真实机器人RLDS/TFRecord；不统一本体/动作/相机|上游按子集许可；本地README记录来源，未统一转换。P1备用，不能把完整dataset_info宣称为Bridge本地已齐|
|OXE/fractal20220817_data|87,212轨迹；119.260GB|Google Robot单臂、外部/头视，末端控制及夹爪，含移动/终止字段，非统一14D|优先级低于桌面同机；许可按[OXE官方目录](https://github.com/google-deepmind/open_x_embodiment)对应子集核|
|OXE/taco_play|3,603轨迹；51.289GB|Franka桌面/厨房，外部＋腕视，末端6+夹爪类型动作|P1可挑接触/放置；P2独立adapter|
|OXE/jaco_play|1,085轨迹；9.921GB|Jaco单臂，桌面外部/腕视；末端平移/夹爪等字段|较小，P1可选，动作非Piper同语义|
|OXE/bc_z|43,264轨迹；86.477GB|Google Robot单臂，外部视图，末端6+夹爪控制|P1备选，接触细节/视角不优于同机|
|OXE/bridge_orig|源metadata60,064轨迹（53,192 train+6,872 val）；已完成119.399GB|WidowX单臂、第三人称/桌面，常用7D末端变化＋夹爪|CC BY4.0；原Bridge V2发布60,096与该RLDS60,064是版本差别。仅白名单完整shard可读，不把partial当数据|
|`dataset/RoboDojo_real_lerobot`|600轨迹/665,071帧/30Hz=6.158h，6任务；1.6GiB，逻辑1.656GB；1800 MP4实际文件、2 Parquet|meta robot_type=**dual_x5**，双臂夹爪，14D状态/动作，顶＋双腕640×480|体积小是高压缩视频，非缺视频的推断；未全量视觉质量检验。当前上游[HF数据卡](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)标Apache-2.0，官方代码README标MIT；本地转换来源/real子集对应revision仍需锁定，不能用代码许可替代数据许可。Piper P1/P2候选，未证明就是本机Piper|
|`dataset/egoscale/raw`|37,401个命名任务包（根有37,402入口）；名称起止帧合61,760,140，按抽查30Hz估**571.85h**；该数字是目录区间估计，非全片解码/去重认证。磁盘全量du因共享FS元数据遍历耗时17分钟未完成，已停止自有扫描；不伪报字节数|orb_stereo，人类第一视角双RGB1920×1080；手21点、MANO、head pose；**没有Wuji真实20D命令**|内部OSS来源，未核公开/内部训练授予。只考虑经授权的真实RGB用于P1；derived_wuji、retarget、模型动作/评分全部排除。先不做大规模预处理|
|`dataset/egoscale_probe`|probe_report明确为egoscale同源抽样/质检；磁盘限时25s遍历未完成，未给总量；一个示例1530帧51s，历史目录inventory的810,558是清单项，不是可训练episode或小时|原视频＋质检/模型语义/衍生缓存|不是新增独立语料；不得把模型caption/quality当测试真值；未全量去重|
|`colleague-c/datasets/EgoDensePrompt`|catalog为空；docs/directory_layout.json status=design_only，但目录有后续runs/cache/reviews；磁盘限时25s未完成|分段/标注工程及缓存，源视频设计为留在原处|不能因为存在该目录就新增一个人类视频数据集；规模、源关联、使用权待所有者确认|
|`dataset/RoboMIND/pick_place`|du可读部分2.7GiB（有PermissionError，非完整总量）；4个lerobot任务入口；其中一个读取直接PermissionError，其余可见层基本为空/preview；未核到完整可训练规模|agilex等线索，动作/视频实际schema未全部读取|不改权限、不绕过；不将上游107k/310k直接计本地。待所有者提供可读manifest|
|GenieSim、RoboTwin、LIBERO、RoboCasa、RoboDojo_Simulation及rollout/retarget链|目录存在；未为本任务展开计量|仿真或混合来源，非同机真实录像|**不进入训练、标签、打分**；RoboDojo必须明确real来源，不能仅凭文件后缀判断|

这些外部数据没有声明来自本项目原93或Piper采集，但“公开”不是零污染证明；若团队上传过本域录像、共享benchmark或模型含本域后训练，仍须canonical provenance过滤。外部本体不同不要求挪动标签来对齐。

### 4.2 权重盘点：视觉先验已有，共享6×256核心先验没有

本轮只stat文件/读配置，不反序列化checkpoint、不hash大权重、不推理，不把存在当作能加载。

|资产|实物/体积|对E79预训练的裁决|
|---|---|---|
|`dengjianfeng/models/dinov2_official_20261031`|vits14 .0883GB，vitb14 .3464GB；目录415MiB|首选原始B/14作在线初始化与固定E*；已有资产，不需下载。原始通用预训练与同事任务LoRA分开；[官方](https://github.com/facebookresearch/dinov2)条款应随包保存|
|`dengjianfeng/models/siglip2/model.safetensors`|1.5008GB|备选视觉，不是视频核心；E79的矩形位置适配仍需独立实现/验证|
|`dengjianfeng/eyerobot2-top-stereo/checkpoints/pretrained/dinov3`|S .0865GB、S+ .1149GB、B .3429GB|已有细节备选，E78记录自定义DINOv3许可；不当作Apache通用许可资产|
|`dengjianfeng/models/vjepa2_ac/vjepa2-ac-vitg.pt`|11.7607GB，目录11GiB|离线视频表征候选；AC预测器7D/4Hz与54D/14D不匹配。精确本地AC来源/条款链仍依E78待闭合。未确认独立V-JEPA2.1 B/L权重|
|`base_model/OpenWAM-Alpha-Pretrain-Foundation-Model`|checkpoint_step_154000.safetensors 24.8138GB，目录24GiB|架构不兼容，不能直接初始化新6×256；E75适配已停，本轮不恢复；训练来源若混仿真不能充当本方案无污染先验|
|`base_model/OpenWAM-Alpha-Real-RoboDojo-Piper`|step30000 24.8138GB，目录25GiB；config/normalization在|名称不保证Piper穿带动作、相机、关节顺序兼容；不直接移植或做测试教师|
|`base_model/fastwam`|目录31GiB；E78核Wan2.2-TI2V-5B/VAE/T5基础组件|未证实有训练好Fast-WAM动作专家，不适配小核心初始化|
|`models/lewam-drawer-tc-official`|E78核lewam_best.pt：22D、z192、4层、3视图|任务专用旧结构，不是通用可搬核心；许可需精确drawer-tc来源|
|`base_model/flux-3-action-base`、pi05/VLM类|已有目录是历史资产；本轮未加载|不用VLM、不加新动作专家，不当本次E79现成权重；FLUX有自定义许可|
|`base_model/Robometer-4B`|本轮du2.5KiB，未见可用权重|不可用，也不得当测试评分真值|

**关键判断：不需要为了P1新下视觉权重；瓶颈是未学过时间变化的共享Transformer，加载图像encoder不能替代这一步。** 本域同事已训练checkpoint可能见过全部93或全部173，应排除作为公平“预训练初始化”的来源。P1应从干净原始视觉权重＋随机共享core，在合规allowlist真实录像上得到可追溯初始化。


## 5. 开源数据主页、论文与许可（只查网页，未下载）

核查日期2026-11-12。**相关度为本报告工程判断，0–5分，不是实验成绩或模型评分**；主要衡量真实视觉动态、视角、双臂/接触任务接近程度，分别给“天机顶视瓶入箱 / Piper双臂穿带”。许可未闭合的数据只列候选，不进入默认训练。小时按物理交互计，多相机不乘倍；未知体积明确写未核实，不能用HF自动viewer行数当episode数。

### 5.1 机器人数据

|数据与第一手来源|规模与下载量|视角；本体/动作；双臂/灵巧手|许可与相关度（天机/Piper）|
|---|---|---|---|
|[DROID主页/下载说明](https://droid-dataset.github.io/droid/the-droid-dataset.html)；[RSS论文](https://www.roboticsproceedings.org/rss20/p120.pdf)|论文76k轨迹、350h；RLDS1.7TB；原始双目HD8.7TB，非双目5.6TB；100集调试包2GB。**无需再下，已有Cosmos3转换**|两外部相机＋腕相机（原始支持双目）；Franka Panda7关节＋夹爪，关节8D或末端6+夹爪=7D；单臂、平行夹爪|原论文明确CC BY4.0；Cosmos3另OpenMDW1.1。**3/3**：桌面物体动态好，非同机顶视、非双臂穿带|
|[Open X-Embodiment官方](https://github.com/google-deepmind/open_x_embodiment)；[论文](https://arxiv.org/abs/2310.08864)|论文级1M+轨迹、22本体；不是单一均质数据集。全量体积随子集，不给一个伪精确数；本机5子集约386GB，4个完整|第三人称/腕部因子集异；主要夹爪，有双臂ALOHA等；原始动作可为关节/EE/速度/绝对，不能把RT-1-X的7D接口当全部源数据动作|软件Apache2.0与各子集数据许可分开；[官方清单](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit)本轮网页读取失败，故不为全部子集作统一授予判断。**整体3/3，挑ALOHA可2/5**|
|OXE本地相关子集：Bridge、TACO Play、Jaco Play、BC-Z、Fractal|本地源metadata分别60,064/3,603/1,085/43,264/87,212轨迹；磁盘见§4；不再下载|Bridge常用EE6+夹爪7D；TACO一般EE6+夹爪；Jaco有平移/夹爪/终止字段；BC-Z EE6+夹爪；Fractal另有base/terminate，维数以schema为准。均不是Wuji20D真实手|Bridge已核CC BY4.0，其余逐子集许可尚待锁定，不推定等于代码Apache。**Bridge3/2，TACO3/3，Jaco2/2，BC-Z2/2，Fractal2/2**|
|[AgiBot World Beta官方README](https://github.com/OpenDriveLab/AgiBot-World)；[发布方卡](https://huggingface.co/agibot-world)；[论文](https://arxiv.org/abs/2503.06669)|1,003,672轨迹、2,976.4h；官方原包约43.8T；Alpha92,214是Beta子集，不能相加。2026新G2系列是另一个版本，不混计|移动G1双臂、头/腕/其他传感器；7+7臂，夹爪或6DoF灵巧手，常见臂手子空间16或26维，另有全身字段，精确释放schema待选包核；不是Wuji双20D|CC BY-NC-SA4.0，需确认研究/商用范围。**3/4**：双臂协调有价值，第一/头视、手型不一致；当前本地仅部分转换压缩包|
|[ALOHA/ACT](https://tonyzhaozh.github.io/aloha/)；[原论文数据段](https://arxiv.org/html/2304.13705v1)|原6真实任务5×50+1×100=350演示；每任务约10–20min；RT-X静态集合另为825演示（Mobile ALOHA论文口径），不是把两者相加；具体真机发布包下载体积本轮未核实|固定双ViperX、14D关节目标（每侧6+夹爪），50Hz；4路640×480，顶/外部＋双腕；双臂、非灵巧手|ACT代码公开≠所有真实演示许可已闭合。需指定RT-X/作者发布真机子集核许可。**2/5**：精细双手接触最接近Piper，Thread Velcro也不是穿塑料锁头的同一任务。ACT README显眼下载链接是仿真示范，**不可误下用于本任务**|
|[Mobile ALOHA主页](https://mobile-aloha.github.io/)；[论文](https://arxiv.org/html/2401.02117v1)|论文7任务：多数50、Cook Shrimp与High Five各20，共290任务演示；共训静态825另计。主页有真实数据Drive入口，整包小时/体积本轮未获得可靠清单|双臂14D＋移动底盘线/角速度2D=16D；全局/头部＋腕视；双臂、夹爪|精确数据包许可待核，代码许可不替代。**2/4**；可选静止接触片段，不能删base动作后冒称完整动力学标签，不照搬论文时间移位做本项目标签对齐|
|[ALOHA 2](https://aloha-2.github.io/)|硬件/系统论文与开放硬件，不等于一个统一可下载的大规模数据发布；未核到官网完整真机语料规模/体积|改进双臂夹爪、通常14D关节目标，多视图；非20D手|逐具体数据发行核许可；官网同时有MuJoCo模型，**排除**。**潜在2/4，已确认可用规模=未知**|
|[RoboMIND官方](https://x-humanoid-robomind.github.io/)；[数据卡](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)；[论文](https://arxiv.org/abs/2412.13877)|V1.x107k轨迹/479任务，含5k真实失败；HF仓库12.3TB。另[2.0论文](https://arxiv.org/abs/2512.24653)是310k+双臂轨迹、6本体、739任务，不能用V1规模/许可体积拼成V2统计|Franka、UR5e、AgileX双臂、Tien Kung双灵巧手；外部/头视＋腕视；动作随本体，单臂EE6+夹爪或关节，AgileX常见双6+夹爪14D；灵巧手schema未核（官方yaml访问401），不猜20D|HF卡Apache2.0且需要接受访问条件；真实与Isaac Sim数字孪生严格分开。**3/4**；优先双臂真实失败子集，但本机相关目录未完整可读|
|[BridgeData V2主页](https://rail-berkeley.github.io/bridgedata/)；[论文](https://arxiv.org/abs/2308.12952)；[下载索引](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/)|60,096轨迹=50,365遥操+9,731真实机器人脚本rollout；24环境。官方压缩411G+30G，RLDS较小；小时本轮未从第一手发布页核实|WidowX单臂夹爪，外部多视角/桌面，通常5Hz、7D EE增量+夹爪；非双臂/灵巧手|CC BY4.0。**3/2**；脚本控制的真实录像与仿真不同，但首轮优先人工示范。与OXE Bridge是同源，不双计|
|[RoboDojo数据卡](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)；[官方repo](https://github.com/RoboDojo-Benchmark/RoboDojo)|全仓库7.82TB含不同资产，不能称全部真实；本机real600条/6.158h，仅1.656GB|本地dual_x5、14D、顶＋腕三路，双臂夹爪；非灵巧手|当前数据卡Apache2.0，代码MIT；本地转换revision仍待接链。**3/4**；只取real并核任务，排除sim和来源不明rollout|

### 5.2 人类手视频与灵巧手数据

|数据及第一手来源|规模/体积|视角、手/动作语义|许可；相关度（天机/Piper）与裁决|
|---|---|---|---|
|[Ego4D主页](https://ego4d-data.org/)；[下载/许可流程](https://ego4d-data.org/docs/start-here/)|主页3,670h日常生活；官方文档Full Primary约7.1TB、Entire30+TB（旧段落5TB与更新表不同，以版本清单为准）|人类第一视角，有单双手活动；非机器人动作，无统一20D控制标签|专用Ego4D协议，当前访问迁至Lab22，需正式接受；**1/2**。大量非操作、头动，当前无须下载|
|[EgoDex官方repo](https://github.com/apple-aiml-research/ml-egodex)；[论文](https://arxiv.org/abs/2505.11709)|829h、194任务、30Hz1080p；725h train+7h test+97h additional；训练5×300GB、test16GB、additional200GB≈1.716TB压缩|Apple Vision Pro第一视角；人类双手/手指/head/上身3D追踪。是pose annotation，**不是Wuji20D命令**|数据CC-BY-NC-ND（与代码Apple许可分开）。**2/3**；桌面任务强于Ego4D，但域差和许可限制明显，不下载整库；ND/衍生训练使用范围需确认，不作商用默认|
|[DexWild主页](https://dexwild.github.io/)；[作者链接数据卡](https://huggingface.co/datasets/boardd/dexwild-dataset)；[论文](https://arxiv.org/html/2505.07813v1)|论文人类9,290示范/93环境；当前发布卡人+机器人9,505集、33+h/3.5M+transition、2.14TB，发布统计与论文不能混用|掌部双目＋可第一/第三人称tracking相机；人手与LEAP/xArm6/Franka真机，手关节与腕位姿组合，非固定Wuji20D；单双手依任务。发布动作还含重定向/滤波流程|发布卡MIT；**2/3**。真实RGB可候选P1，**重定向动作不推荐P2**；喷瓶/收玩具相关，掌视不能当顶视|
|[DexCap官方repo](https://github.com/Qingxin-Wang/dexcap)|提供raw dataset入口；本轮未核到可采信全库小时/下载体积，正文访问有超时，不能据论文演示数假造已开放量|胸前RGB-D/动捕手套，人类双手；双LEAP+臂默认46D控制输出（非20D/手）。官方说明以PyBullet IK重定向|数据授权范围未闭合；**2/3**。仅原始真实RGB候选，重定向标签排除，暂不首选|
|[DexYCB官方](https://dex-ycb.github.io/)|1,000抓取序列、约582k多视图图像；8路不能乘8作物理时长；官方压缩119G|8台第三人称RGB-D、人类单手抓20类物体，MANO/物体pose，无机器人动作；非双臂|CC BY-NC4.0；**2/1**。多视图抓取可见性好，短抓取片段不提供长时双臂穿带动力学|
|[UniDex（CVPR2026）](https://github.com/unidex-ai/UniDex)；[论文](https://arxiv.org/abs/2603.22264)|50k+轨迹、8种6–24DoF手；官方README说全部处理后可达80TB；不是8种机器人各自真实遥操50k|源H2O/HOI4D/Hot3D/TACO人类视频，转为多手重定向，包括Wuji20DoF实验；第一视角为主|代码MIT不覆盖所有源数据；**原RGB可2/2，重定向训练推荐分0/0**。按用户约束排除其retarget标签，不能解决缺少真实Wuji命令—后果数据|
|[HandsOn](https://github.com/TsinghuaC3I/HandsOn)|有公开下载语料，但明确是Isaac Sim 5.1仿真基准；本轮不扩计规模/体积|双ROKAE+Wuji Hand1，每侧20D手；52D世界掌位姿+手命令；双臂灵巧手|即使形状接近也属于仿真；**训练推荐0/0，排除**。不把真实定性演示等同已发布真实大库|
|[RynnWorld-Latent第三方来源说明](https://github.com/alibaba-damo-academy/RynnWorld-Latent/blob/main/THIRD_PARTY_LICENSES.md)|只开放3个tianji_wuji_data quickstart chunk及1条marvin_wuji downstream真实样本；训练/评估全库明确不再分发；小时/视频包体积未核|Marvin-Wuji相关是真线索，但只是少量样本；其latent/world生成数据不是新的真实命令后果|repo Apache2.0不自动覆盖外部数据；**若来源独立P1可4/1，但规模不足且极需与内部数据查重**。不引入其模型生成视频/标签，不因本体名相同认定可P2大预训|

**Wuji公开数据判断。** 不能再笼统说“没有任何Wuji相关开放资源”：2026已有UniDex、HandsOn、RynnWorld等。但本轮未核到一个规模大、完整公开、许可明确、含本机顶视＋真实54D命令与反馈且独立于本项目评估集的Wuji语料。已找到的主要是人类重定向、仿真、或极少quickstart。对本项目，内部435条真实记录比“20DoF”关键词匹配到的这些数据更直接。

上述未核项（某些ALOHA/DexCap小时/体积/许可、RoboMIND灵巧手schema、OXE子许可）是明确审计缺口，不作为已批准下载或训练的候选。首批选择只依赖已经落盘、可溯源的小量真实视频；无需为补齐百科表而先下载数据。


## 6. P1/P2/P3 与最小验证建议

以下均为未来方案，**本轮没有执行**。优先级是“同机真实视频小先导＋有针对性的任务补采”，不是大规模下载再扩模型。没有证据证明单纯预训练一定比追加演示便宜。

### 6.1 P1用哪些，如何读，准备多少磁盘

|线/顺序|候选与小时预算|采样及用途|新增下载/磁盘|
|---|---|---|---|
|天机P1-0，最高优先|先10月9日37条积木＋10月12日50条瓶托，0.75h；再加入魔方/纸巾/积木/挤纸巾/笔帽/试管，**上界2.88h**，授权及隔离后争取2–2.8h净量|只读cam_high/camera_top；按任务→独立capture/session→窗口分层采样，避免长片/重复帧支配；初期一半抽瓶/放置近任务，一半其他接触任务（采样比不是新增小时）|下载0；原位读取已有Parquet/MP4/JPEG。CPU索引小文件<100MB量级；首轮追加缓存/日志/两组权重合计建议≤25GB|
|Piper P1-0|现有成功训练parent（3.326h总集合的训练部分）＋demo4 rollout/失败/恢复、demo3 RealSense真录像；先以**4–8h去重后候选预算**启动，实际不足就如实用不足小时，不补重复片冒数|先parent隔离，再按连续原时钟取片；保留失败前后、短暂停顿与恢复；三视图独立camera/age/missing token；legacy和15Hz老Piper留第二批|下载0；首轮建议新增≤100GB（更多视图、原生腕ROI）；原片不复制，未授权或净量未核就不启动|
|Piper P1-1，仅本域先导有收益后|本地RoboDojo real至多6.16h中选装配/插入/双手协作2–5h；再从Cosmos3-DROID选5–10h真实近距离接触（需要时才加）|先确认real与动作/相机来源；P1可忽略动作但不忽略时间；本域/外域按1:1采样作固定扩展配方，不让415h外域淹没本域|下载0；在原目录读取，缓存仍受≤100GB上限；不新增全量副本|
|天机外域扩展，后置|只考虑经人工确认的**真正顶视/俯视桌面**DROID、Bridge或RoboDojo允许片；当前未量出满足视角约束的净小时，不能保证凑出5–10h|天机只用顶视：DROID两个exterior不自动等于top；人类ego/腕视不放入天机首轮。可先不加任何外域数据|下载0；不足就补采本机顶视，不为满足小时目标放宽相机合同|
|暂不做|AgiBot全量/续传解包、Ego4D/EgoDex全量、egoscale全库、任何retarget/仿真；无来源同事权重|许可、域差和磁盘成本大于首轮价值；人类真实视频只作为Piper后续受控候选，原手pose不转换成测试真值|新增下载0；无大文件复制/展开|

**P1模型合同。** 与E79同一个6×256核心、同一个O/W tokenizer与固定E*；A token **不存在**（长度0、无动作投影调用/无A attention/无LA），不是零动作或噪声假动作。O只读O；W读O/W，未来真实RGB只生成监督目标，不进入O。不存在状态/上一命令/FK就用availability mask或缺失槽，不能填伪造q；同本体有状态的真实视频也可在首轮统一mask，避免把“无动作视频先验”与额外状态监督混为一谈。W用E79绝对固定特征目标，不引入教师生成图像/事件标签。

**公平先导的默认：P1冻结在线DINO基础及LoRA（LoRA为初始零增量），训练O/W输入输出投影和共享core。** 这仍是同一实现的训练开关，直接检验共享core先验；P3两组才按E79同样开启末2层rank8 LoRA。若P1同时更新LoRA并整体迁移也可以作为第二个明确实验，但那回答“整个预训练包”收益，不能把提升全归因于Transformer。目标E*始终固定，统计只用允许训练parent。

**时间与分辨率。**

|项目|建议|
|---|---|
|抽anchor频率|天机初筛2–5 anchor/s，Piper5 anchor/s（穿入/恢复可提高到10，初期固定配方）；原30/15Hz时间轴保留。不把抽anchor率等同模型控制率|
|两帧与未来|天机保持15Hz合同：O为t−4,t，W为+8/+16/+32，即约0.533/1.067/2.133s；Piper30Hz：O t−6,t，W +10/+25/+50，即0.333/0.833/1.667s。用真实时间戳找源帧并记录偏差；无合格源帧就mask/排除，不复制末帧、不跨剪辑|
|变帧率/重复帧|按传感器frame_number/时间戳识别重复曝光并记录age，bottle50不能把4,935重复帧算独立观测；选anchor时降权长静止段，但保留接触前后真实停顿|
|天机像素|每个训练session人工确认顶视有效工作区，保留比例至192×320，DINO补边至196×322；不可机械继承93的rows96:480。P1无FK时用训练帧固定区域或missing ROI，仍保留全图；不以未来手位置选框|
|Piper像素|顶448×252，腕全图保比例224×224；细节ROI从原640×480腕图取，不能从224回放大冒称细节。原生MP4直接解码，保持相机ID/同步偏差；不先横拼三图|
|数据格式|优先原位解码现有视频/嵌入JPEG，自己目录只写索引、来源、必要小缓存。确有CPU/I/O瓶颈才做少量选中帧的JPEG/WebP缓存；有LoRA的当前图像必须保留像素前向，不能用固定当前特征冒充端到端|

**为什么不全patch缓存。** 196×768×2=301,056字节/帧≈301KB（294KiB）；每小时每视图在5/15/30fps分别约**5.42/16.26/32.51GB**，三视图再×3。10h×15fps三视图约488GB；415h单视图15fps约6.75TB，已超过可用盘。E79实际天机原生322patch还更大，不能用196patch成本宣称全量可行。以20–60KB/JPEG作低分辨率规划假设，5fps约0.36–1.08GB/h/视图（不是实测压缩率）；原始试管640×480JPEG本轮平均约77.9KB/张。

固定目标可在**pool后**缓存：天机77×768×2=118,272B/端点，Piper全三视图合91×768×2=139,776B/端点。按每anchor3未来端点：
- 天机2 anchor/s约2.55GB/h，2.88h约7.36GB；5 anchor/s约6.39GB/h，2.88h约18.4GB。首轮选择2 anchor/s或按需LRU，目标缓存上限约10–15GB，加小权重/日志/解码缓存控制在25GB。
- Piper5 anchor/s约7.55GB/h，8h约60.4GB；加索引、像素LRU/权重，控制在100GB内。不要把91个跨视图合并token又乘3相机。
- ROI随anchor当前框变化时，缓存键必须含source frame、anchor/crop box、camera、E*版本与pool/统计版本；不能按frame_id复用不同框的目标。可以缓存固定全图pool，动态ROI按需编码。固定目标与在线LoRA观察特征不能混用。

这些预算是上限设计，非本轮已生成的缓存。/mnt结束复查98%、剩余约3.1TiB（开始核查97%/3.2TiB），共享空闲不是本项目配额；不自动清理任何人文件，不为预训练展开1TB AgiBot压缩包。两线原位先导合新增盘预算≤125GB；未取得资源安排前保持当前只读状态。

### 6.2 P2值不值得：分开判断

|线|判断|适用数据与动作投影|
|---|---|---|
|天机|**首轮不做公开跨本体P2**。更值得做已核真实同机多任务中期训练（属于同本体有动作预训练），但应在P1是否有效之后单独验证|435条内部候选很多有真实q_cmd/q_fb，可回原始54D命令合同；同事58D TCP+40手转换不与54D直接混头，不逆推/平移成命令。优先瓶/积木/放置；无可靠命令合同的样本只P1。Wuji公开retarget/仿真量大不代表真实20D接触经验大|
|Piper|**有条件值得做，小预算后置**。双臂夹爪与ALOHA/真实RoboDojo/AgileX更接近，预计比天机跨本体P2更有希望，但P1与恢复补采优先|先用许可/来源明确的真实双臂14D关节数据；再考虑DROID7D EE作为独立schema。不同相机、控制频率、夹爪符号/单位和command vs follower语义分开，不因“都是14D”共用head|

默认**每本体×动作语义独立投影**：`a_r → Linear/小MLP(d_r,256) + embodiment/time embedding`，输出 `256→d_r`，核心attention/FFN共享。状态/上一命令也使用本体adapter、训练组独立统计；角度rad、夹爪m，时间以秒显式输入。天机固定A7+B7+左手20+右手20顺序；Piper左6+夹爪+右6+夹爪顺序。每组损失按有效步/维度均值、按本体平衡，避免54D压倒14D；训练源的动作是command还是next-state需要schema ID。

补齐到统一最大维度+mask可用于**batch打包**，但不作为“前14维就自动语义对齐”的共享投影。缺维从输入、输出和loss有效性中剔除；P1的整套A缺失仍是删除token，不是padded action。P2只迁共享核心/视觉相关参数，目标机器人动作头独立初始化或从本体干净BC继承。无动作人类视频不伪造动作；不要将手重定向或模型IDM预测作为真实动作监督。

### 6.3 不先放大共享Transformer

先保留E79的6×256，观察**真实动作指标**是否从P1获益；“数据更多所以模型必须更大”并不成立。新数据主要只有数小时，来源偏差和接触可见性仍是限制。对6×256已证明收益、训练/开发曲线都显示容量不足且token/动作合同已固定时，下一次只试E79建议的6×384。

|核心（FFN=4d，忽略小bias/head）|约12Ld²参数|相对6×256的线性/FFN计算|attention矩阵运算项约LN²d倍率|
|---|---:|---:|---:|
|6×256|4.72M|1×|1×|
|6×384|10.62M|2.25×|1.5×|
|8×384|14.16M|3×|2×|
|8×512|25.17M|5.33×|2.67×|

这不是整pipeline时延倍率。总时延近似 `T_decode+T_vision+T_core+T_transfer`；若core只占30%，core×3/5.33才对应约1.6/2.3倍总时延示例，不能当H20实测。LoRA不省去视觉前向。O前缀可缓存的部分、joint421/509 token与action_only190/236 token、去噪步数都影响开销。天机执行8步约533ms、Piper执行10步约333ms（穿入5步约167ms）是重规划期限；不是要求每66.7/33.3ms完成整个模型。扩容只有在全流程P99/观测年龄/deadline miss实测可接受后才考虑；本轮无GPU时延测量。

### 6.4 最小验证实验：只比较预训练初始化与随机初始化

**目的：在相同E79架构、相同任务微调流程下，判断P1能否改善前8步真实动作误差。** 首轮不同时加入P2、扩容、换视觉塔、新ROI算法或伪标签。

|步骤|固定合同与比较|
|---|---|
|0：CPU准备|先完成授权、allowlist、parent分区、真实连续窗口与相机合同；基准原fit64=2159窗口，不把新增任务动作直接混入微调以免改变问题；Piper先修split。预先登记训练/评估manifest和指标，未来写自有新目录|
|1：配对初始化|同一个原始DINOv2-B、固定E*、同shape随机core seed S0。R组保留S0；P组从S0做P1 O→W。首轮P1冻结视觉/LoRA，只更新共享core、O/W投影；A头在P3两组用同一随机张量|
|2：P1小先导|天机允许视频≤2.88h，Piper去重允许池先4–8h；最多5k更新（有效batch32–64）为首轮，不因LW持续下降自动延长。训练loss只作优化/崩塌诊断；checkpoint取固定终点，不能用测试latent误差选|
|3：相同P3/R1|两组相同fit64/parent窗口、batch与采样顺序、动作噪声随机流、优化器、LR/LoRA开关、更新数与验证网格；天机预先固定2700或5400更新，不能让R短训、P长训。Piper预算内固定5k–10k起点，profile后在比较前锁定。缺失W的BC warm-start及之后R2如果启用，两组流程完全相同|
|4：首个判定|先比较同一6×256 BC/R1的前8步；R1无收益不要以R2加复杂度救场。通过后两组各从自己的BC做相同R2配方，再比较WAM；另报joint/action_only，不按测试选择开关|
|5：重复/确认|一对seed仅筛明显失败；通过才第二对seed，小差异再第三对＋新session。旧holdout/frozen只历史回归，最终结论在协议冻结后新增隔离测试|

指标沿用E79/E66现有真实动作误差定义及共同训练统计，不重新发明单位。主指标为每episode/parent等权的 `E_1:8`，报告 `Δ=(E_pre−E_rand)/E_rand`（负值好）、配对父组bootstrap区间、各seed结果；天机额外分双臂/双手原单位误差，Piper分12关节rad/2夹爪m，不直接把不同单位裸平均。Piper本轮用户指定**前8步为主**，同时保留E79前10步为部署护栏，避免默默更换旧合同。教师特征loss不用于最终选型。

**预注册工程阈值建议**：开发集两seed平均前8误差相对下降≥5%，且至少没有seed明显反向、配对区间支持改善，才值得扩大；手部/夹爪/前10护栏退化不超过预设2%量级（开跑前定，不看测试调）。≥5%是项目投入门槛，不是统计学定律或数据集公认阈值。若效果2–5%且CI跨0，判“不确定”，优先补独立parent而非加大模型；离线动作误差改善不能冒充任务完成率改善。

|预算桶（H20占卡小时上限，非测得用时）|天机|Piper|
|---|---:|---:|
|未来100更新profile＋必要目标编码/小缓存|1–2h|2–3h|
|P1固定5k更新先导|4–8h|6–10h|
|R/P同架构R1微调一对＋固定网格评估|8–12h|12–20h|
|首轮总上限|**13–22 GPUh**|**20–33 GPUh**|
|仅通过后第二seed/相同R2扩展追加桶|10–18h|16–24h|

先跑天机一对能最低成本验证core先验；Piper可能更有世界建模潜力，但先清parent。预算含加载、编码、失败重试、评估；按卡数×实际占用墙钟，排队另计。未来profile用 `1.25×(编码秒数+更新数×秒/更新+评估秒数)/3600`，超桶就减更新/缩数据后**重新锁定两组一致流程**，不偷偷增加卡。若P1 5k在8h（天机）/10h（Piper）内做不完，不承诺跑满，先检查I/O与token成本。当前实际消耗**0 GPUh**。

**止损。** 出现parent泄漏/时钟不连续/混入retarget立即停止比较；真实BC在固定预算仍不工作则先修可见性/监督，不能预训练掩盖。第一seed前8误差恶化≥5%且训练稳定时不扩库，可做一次有限优化检查；第二seed无稳定≥5%改善，停止P1扩量/P2/扩容。LW明显下降而动作无改善就是无业务增量证据。部署P99超既定期限则不升级模型；不能删去视觉/传输时间美化报告。

### 6.5 补采通常更划算：具体采什么

**我的判断：在现阶段，补采任务相关变化和恢复，比投入大规模外域预训练更划算；免费存量P1可做一次有界先导，但不应延误补采。** 2.88h同机跨任务视频能教一般运动，却不能让不可见的指尖接触变可见；Piper已有成功偏置与同parent相关性，新增真正独立失败/恢复提供的信息更直接。

|线|建议数量（规划，不是已采）|内容与记录|
|---|---|---|
|天机新增训练|首批**80–120个独立重置episode**，至少3个日期/session；约1–2h有效交互是目标，不为凑时长拖长静止|约40–60个不同瓶/箱相对位置、初始姿态/双手分工的完整成功；20–30个滑脱、抓空、碰箱沿、放置不稳；20–30个真实恢复/重抓。保留尝试前到结果后的连续顶视、真实命令/反馈、手状态、相机序列号与时间戳；人工标结果及可见性，不加模型真值|
|Piper新增训练|先复核现有rollout/接管，再补**100–150个独立失败→恢复parent**＋**50–100个完整成功parent**；至少3次不同摆放/日期；目标5–10h新交互，实际按录制计|尖端偏离/锁头角度变化、握点偏、弯折/滑脱、穿入前退出重对齐、部分穿入再拉紧、换长度/宽度/颜色；保留三相机原分辨率，关键腕视清楚；一个parent可多片，但计一次独立尝试|
|新确认测试（完全另留）|天机20–30个新episode、Piper30–50个新parent/重置起步，按成功率CI再扩充|在模型/协议冻结后采新session，不参与P1/P2/P3、ROI/统计、教师适配；只人工/真实动作及实机结果作真值|

若顶视仍看不清天机抓取接触，先在**允许的顶视**改善照明/镜头构图/原始分辨率、调整物体摆放使结果可见；不通过腕相机绕过用户约束。采集成本应记录人员小时与有效独立parent，而不是只与GPU小时直接等价。失败全过程可用于P1，动作BC只收经过人工确认的正确示范/恢复动作，不奖励整段失败动作。


## 7. 待确认事项（本轮只列清单，不发送消息、不申请或下载）

|需确认者/事项|需要的具体答复|不确认时怎么处理|
|---|---|---|
|贾天才及天机原采集团队|10月9日37条、bottle50的研究训练/内部交付/后续商业用途授权；相机CP28563000JZ及同一机器人是否全程成立；是否还有未筛选失败/调试raw session|这些数据保持只读候选，P1不启动；不把能读文件当授权|
|何海森、colleague-b及后续天机任务采集者|后续348条可用处理/原图记录的来源和训练用途；原始第43条能否只做短时距P1；是否有其他session/失败被截去；自动停止含义|不将COMPLETE/yes手势当成成功；无明确来源的条目隔离|
|Piper所有者/操作者|demo3/4/6的完整parent/session联结与未来测试parent清单；demo3→demo4重标列表；同一臂与demo6相机更换记录；允许使用未审阅rollout做P1、恢复段做P3的范围|不把全部根目录混训；原raw只读，测试相关整父排除|
|贾天才Piper早期数据|9月2日30条和9月5日31条的设备/相机、leader命令语义和左右映射，原始链接失效后的权威来源；132阶段数据仍继承何海森parent|只作P1候选；动作schema未清不得P2；caption/latent不当真值|
|张鹏/egoscale所有者|真实RGB使用权、人体视频许可/隐私范围、与probe/EgoDensePrompt的源ID关系；RoboMIND可读manifest而非改权限|不动权限，不用模型标注，不计重复小时，不扫描凭据|
|公开数据许可|Cosmos3-DROID上游归属与OpenMDW1.1 notice；RoboDojo本地real转换对应release；OXE各子集；AgiBot NC-SA、EgoDex NC-ND、Ego4D专用协议能否覆盖实际用途|只使用条款闭合的来源，未核许可的不进入默认训练清单|
|磁盘/下载决定|推荐首轮**下载0、额外盘≤25GB天机＋≤100GB Piper**。如另选EgoDex约1.716TB、DROID原包1.7/5.6/8.7TB、AgiBot全量43.8TB、RoboMIND12.3TB、DexWild2.14TB，须另给明确分片/体积/存储安排|不下载、不解压、不清理他人数据；结束复查约3.1TiB空闲不是许可或配额|
|未来训练与补采|是否采用§6固定对照、首轮预算和止损；优先天机13–22GPUh还是先做Piper治理；补采80–120天机、150–250 Piper训练parent，以及另留新测试|本报告不启动训练、GPU查询/占用或上位机采集|

这里没有需要用户批准才能完成的本轮操作：报告、只读核查与网页检索均已在授权范围内完成。上述事项是**未来使用/训练前提**，不是现在要求用户确认以便继续写报告。


## 8. 一页摘要

|关键问题|本次结论与行动建议|
|---|---|
|天机有无93条之外的同机真实录像？|**有，435条候选约2.88h**：10月9日积木37、bottle50、魔方99、取纸巾50、堆积木62、挤纸巾53、笔帽34、试管50。原始ID与93无交集；另魔方原ep43待复核。画面/标定高度支持同套设备，历史文件无逐条序列号硬证据，须采集者确认|
|Piper有无漏用存量？|**有，约5h穿带相关候选上界**：demo4未审阅rollout0.873h、失败/停顿，demo3约3.90h及demo6极少失败；去重和测试parent排除后会减少。另早期笔任务30+31条合0.383h。132阶段集/各种处理版是副本，不新增独立数据|
|最关键污染排除|天机holdout16+frozen9+dev4共29原始ID已列；所有optical93衍生跟随原分区。Piper整parent/session隔离，旧3组跨集parent、demo3→4重标、rollout→接管、132切片均联结。P1看测试视频仍是泄漏；冻结9不加语言|
|现成外部资源|Cosmos3-DROID完整71,907集/415.05h、707GiB；RoboDojo real600集/6.16h、1.6GiB；OXE4子集完整、Bridge不齐。AgiBot只是部分OpenWAM转换压缩包，52/229文件，不能计全库小时。egoscale为人类ego，不是Wuji真实动作|
|P1首选|天机先授权后用2–2.8h同机顶视；Piper先整理本域4–8h候选，收益成立再加少量真实RoboDojo/DROID。**不新增下载**。天机严格顶视，外部exterior/ego不能冒充顶视|
|缓存与磁盘|/mnt开始97%/3.2TiB，结束已98%/约3.1TiB；原位读源。196×768 fp16约301KB/帧，415h单视图15Hz约6.75TB，禁止全patch全量缓存。只缓存有界pool后的固定目标；建议天机≤25GB、Piper≤100GB新增盘|
|P2/模型规模|天机公开跨本体P2暂不做，内部真实同本体动作中期训练更可行；Piper条件成熟后小规模P2，用每本体×动作语义独立投影。先保持**6×256**，不因视频多先上8×384/512|
|最小实验|同一个原始DINO和S0核心，P1预训练初始化vs随机；相同R1/R2微调、动作暴露与评估。前8步真实动作误差为主，Piper另报前10护栏。首轮上限天机13–22/Piper20–33H20 GPUh，≥5%稳定开发改善才扩展；本轮实际0GPUh|
|更划算的投入|优先补独立接触/失败/恢复：天机80–120训练episode；Piper150–250独立训练parent；另留新日期天机20–30、Piper30–50确认测试。大规模外域预训不能补出不可见接触|
|许可与边界|同事数据需用途同意；AgiBot/EgoDex/Ego4D等不能仅凭“开源”默认商用。Wuji公开线索多为仿真/retarget/少量quickstart，均不能冒充真实大规模命令数据。MuJoCo/仿真/重定向不进训练、标签、打分；模型标注不作测试真值|

**交付与审计限制。** 服务器仅写本报告；未训练、未用/查询GPU、未下载数据集或权重、未复制大文件、未改代码或同事目录、未连接上位机、未操作他人进程。为核画面仅CPU读少量源帧并在本地生成小缩略图；没有新增服务器数据缓存。只读探查为定向有限范围，不声称全盘穷尽；没有逐帧解码全量视频或重hash大文件。网页事实均附第一手链接；不确定的许可、公开包体积/动作schema、同机历史序列号和parent净小时均已明确保留为未核项。长时磁盘元数据遍历未完成的目录不填伪精确值，已停止本轮自有只读扫描，不影响任何同事任务。

