# E82-D：Piper 数据治理交付

日期：2026-11-12 BRT。完成时间：2026-11-12T10:41:53.657807-03:00。状态：**parent/split、净池、数值与窗口 CPU 包已交付；P3 独立示范质量和像素门槛尚未完成，未开放 P3 训练资格。**
CPU only，GPU 0 GPUh，未训练、未下载、未加载模型/权重；天机暂停，Piper P2=0。

## 1. 范围、版本与入口

T=/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu。
新目录 data/e82_piper_v1 和本报告开工时均不存在；源目录严格只读。只写本会话 manifest/、parent_graph/、split/、cache_config/、READY 接口、报告与指定追加日志；未写 labels_codex/ 或 code_e82/。
依据 E81 §1、§2.3–2.5、§3、§4、§5.1、§7.1，结合 E79、E80、E77。用户 Piper 专项合同优先于旧 AGENTS.md 中天机/仿真/旧架构冲突项。

- parent 入口：data/e82_piper_v1/READY_parent_split.json（当前 revision=3，status=parent_split_locked）。
- 数据入口：data/e82_piper_v1/READY_data.json。
- 最终读取：manifest/captures_final.jsonl、manifest/video_routes.jsonl、manifest/numeric_cache_v2/、manifest/window_cache_v2/。
- 配置/字段：cache_config/preprocess_final.json、manifest/SCHEMA.json、manifest/README.md。
- 文件 sha256 见 READY_data.json 与 manifest/DATA_FILE_SHA256.json；不把历史 proposal 或中间缓存当最终允许名单。

## 2. Parent 连通图与划分

索引 1,241 个有效 raw/快照/重标 HDF5 记录，656 个完整 timestamp+qpos 数组指纹；图共390个连通分量（包含独立元数据节点，不等于390条可训练录像）。
四类真实 ID：trajectory_id、recording_session_id、test_run_id、controller_job_id；另以 source HDF5/realpath、声明 SHA、全源 timestamp+raw qpos 指纹、同机绝对时间重叠建边。未用相同日期/相机/initial pose 伪造 session ID；缺失 session 明确 null。
自主 rollout→失败→接管→成功由真实 ID 全链并父。已知跨集组：
- 103/104 → 同一 dev 组件 pc_67b50a5790504d01056e。
- 110/111/112 → 同一 test 组件 pc_2034ef29bc3f01490e9c。
- 155/156 → 同一 test 组件 pc_de94492c558bdf6b4ffb。

历史 sealed_test 的15个组件包含55个 raw/快照等记录，全部 test；与 P1/P3 的组件交集为0。身份 token 跨split数量为0。
demo3→demo4 episodes63–67 有显式重标证据与相同数值指纹；132阶段片逐条映射原source索引；micro/fiveview的126条逐条映射到同一源parent，两个release声明的原raw哈希字典一致。它们均不新增计数/小时。旧快照、.views、crop/gaze/legacy 派生遵守同源继承；未证明的派生默认quarantine。legacy未默认纳入净池。

划分以历史146/9/18 proposal起步，完整组件以test>dev>train优先，额外P1组件固定seed保留约12.5%开发分区。全部划分在任何模型结果之前锁定。
- demo4：146 train /9 dev /18 historical test片段。
- demo6成功：21 train /6 dev /6 historical test；两条失败只留dev审计。
- seed=20261112；按sha256(seed:stratum:真实raw ID)排序，先3 test、再3 dev，其余train；分配前写SEED.json。真实ID、原source路径、sha256见 split/demo6_raw_ids_sha256.json。

|分层|train真实raw ID|dev|test|
|---|---|---|---|
|early|3,5,6,7,8,10,11,12,13,14|1,4,15|0,2,9|
|native|16,19,22,23,25,27,29,31,33,35,36|18,21,26|17,28,30|

raw ID不连续，不能把分层内local episode ID误当raw ID。采集session缺失者不按目录名杜撰整日session。

**首版修正记录：**首版图已有部分独立newdagger源路径节点，但未读取完整源身份元数据，发现后立即撤回READY并保留superseded_initial。补读全部173 processed实际源，增加49个完整raw记录后重新锁定；已有来源的split/component变化均为0，见 parent_graph/rebuild_delta.json。当前receipt只哈希稳定数据文件；初版/中间receipt不可用。

## 3. 去重净池、实际可用量

物理时间以30Hz源行计，一段三相机不乘3。不同池高度重叠，不可相加。

|池|去重条数|源帧数|净小时|独立parent|有效窗口|
|---|---:|---:|---:|---:|---:|
|P1候选训练净池|560|692,277|6.409972|245|109,029|
|P3候选：demo4 rebuild+demo6成功train|167|282,033|2.611417|150|33,888|
|P3已独立质量验收并放行|0|0|0|0|0|

P1达到4–8h目标；其中548条/244 parent有完整有效窗口，窗口覆盖688,225源帧=6.372454h；12条太短/不可组成完整窗口保留但不训练。P3候选161条/144 parent有完整窗口，覆盖281,337帧=2.604972h；6条太短不造尾帧。
P3的allow_p3只表示用户指定的候选范围，train_ready_p3=false；成功结局不能证明每段动作都正确。必须独立审核示范质量及像素可见性后，以新版本放行实际P3池。demo3本会话P3保持pending_E82_L，只进允许train的P1。

demo4回raw恢复停顿20,616帧=11.453333分钟（全部候选）；其中训练部分17,338帧=9.632222分钟。不是插值、复制尾帧或把拼接播放时间当真实时间。

P1来源分项（parent有交叉，分项parent不可相加）：
- demo3_dagger_realsense_v2_hdf5：92条，78,355帧，0.725509h。
- demo3_inference_realsense_v2_hdf5：140条，133,153帧，1.232898h。
- demo3_teleop_realsense_v2_hdf5：43条，112,105帧，1.038009h。
- demo4_dagger_realsense_v2_hdf5：81条，56,497帧，0.523120h。
- demo4_inference_realsense_v2_hdf5：112条，79,694帧，0.737907h。
- demo4_teleop_realsense_v2_hdf5：71条，140,094帧，1.297167h。
- demo6_teleop_realsense_v2_hdf5：21条，92,379帧，0.855361h。

## 4. 数值、时间与动作合同

14维固定：左6关节+左夹爪+右6关节+右夹爪，关节rad、夹爪m；左右从臂serial/can映射承接E77/E80原合同，未交换物理数组。
q完全保留原source canonical observation，qpos_raw另存；action的12关节来自既有下一source反馈，夹爪恢复原raw负值并保持signed absolute，不裁零。
167条训练processed来源逐行重核：demo4 provenance source_frame_index与action_target_source_frame_index，以及demo6 parquet原配对，目标索引=source+1、最大动作误差=0。不是新移动动作；不是motor command，没有previous-command anchor或伪命令历史。
H50取action[i:i+50]，对应feedback源i+1..i+50；12关节全部减同一当前实测q[i]作残差，夹爪absolute。历史为8条因果实测q+valid/age，缺失mask；dq为真实dt因果差分，12关节rad/s与双夹爪m/s分开存储。

原始30Hz不降采样；P1 anchor stride6（5/s），P3 stride8。O为源-6和0，W为源+10/+25/+50。缓存保留全部源时钟、逐相机曝光/接收/帧号、反馈bracket、原动作索引。
demo4全raw训练片的插值未来端最大5.847ms，全部训练来源最大19.783ms（demo3也审计）；明确等待最晚反馈host-receipt bracket端点后状态才可得，计入state age和decision_time。不能宣称零延迟状态。
O按每相机 exposure和receive均<=decision_time取last-available；最终已选图像age中位数约50ms，三路最大99.211/98.408/97.591ms，均<100ms。不从未来取“最近帧”。三相机无硬件同步，曝光是设备报告midpoint，光学精度未校准。重复曝光保留帧号/实际age和unique exposure统计。
W按物理目标时刻最近曝光，偏差<=16.668ms。完整历史、动作块、未来端点在同一真实连续段，状态bracket gap<=75ms，源dt必须(0,50ms]。最终另剔除1个相机索引越过允许池边界的P1窗口。
回raw已消除旧静止裁剪拼缝；不是让新窗口跨旧compressed行跳跃。所有窗口还检查所选O/W相机行均在允许source mask及同一连续段。

collection_status / conversion_status / parent_outcome / segment_demonstration_quality / task_outcome五字段分开保存，各有source/reviewer。COMPLETE、conversion passed或parent outcome均不能推导task success，更不能推导逐段quality。未知明确unknown/null。原父failure与成功接管段结果分开保留。未生成测试/开发模型标注。

## 5. 图像、配置、CPU缓存与训练统计

顶视448×252保比例，只有经核实的黑边才裁；不同相机/日期不套固定黑边参数。腕全图640×480→224×168，上下各28 edge pad到224×224，保留pixel-valid与原图仿射矩阵。native顶视回video_native读取1280×720，双腕回原生640×480；没有把224缓存放大冒称细节。
固定腕ROI仅保留E77候选左[290,280,400,420]、右[240,140,430,400]初始化，状态未批准/默认禁用，全腕回退保留；不使用未来图像选框。无主动gaze、无语言塔/成功信息输入O。task词表0=UNK、1=穿扎带；demo3任务待L确认。
6个训练capture、54张训练曝光生成PNG无损预热缓存，包含原生图、网络图、pixel-valid、crop与affine；CPU抽帧指纹保留。三路源视频路由全部存在；源数据不复制。
实看训练contact sheet确认解码、原生视野与三路内容；这是6个parent的格式抽查，**不能替代L的至少20个parent事件可见性审核**，也不能据此批准ROI或整段P3示范质量。

最终数值缓存numeric_cache_v2、window_cache_v2；旧中间缓存只为审计留存，README明确不可作为最终入口。最终审计独立验证相机可用时刻、池边界、原配对索引、跨缝、单位；见FINAL_NUMERIC_VALIDATION.json。
P3候选训练统计已CPU计算，仅144个有窗口的train parent、33,888窗口，parent等权、parent内窗口等权；关节残差与signed夹爪分开，std floor=.01rad/.001m。这是provisional candidate stats；若质量审核改变P3实际名单必须重拟合、重锁，两臂使用相同冻结统计。P1所有state/action/history结构mask，不使用动作统计条件。

硬预算100十进制GB：固定target LRU65GB、pixel15GB、模型15GB、index/log余量5GB；90%停增。本轮没有DINO特征/模型缓存或训练checkpoint。共享空闲低于2.5TiB暂停持久缓存；本轮结束仍高于该线。实际字节见READY_data.json，约0.25GB；不清理旧实验/同事文件。

若未来为P1/P3候选全部缓存固定目标，131,671个合并去重anchor×419,328字节≈55.213GB，低于共用65GB目标桶；这是预算估算，本轮实际目标特征写入0。结束共享空闲2.663TiB，高于2.5TiB暂停线。

## 6. 验收、只读核查与未完成

已通过：已知三组不拆；全部四类身份token跨split=0；历史test整组件均test；P1/P3与dev/test组件交集=0；原pair误差0；负夹爪保留；窗口不跨真实拼缝/池边界；五类结果分离；CPU总缓存<100GB。
1,625个源HDF5/元数据基线以(size,mtime_ns,ctime_ns,inode)在读取前记录、结束后重查，变化0、缺失0；读操作可能改变atime，故不将atime当内容修改。原位视频抽帧另存前后stat检查，变化0。
只读方式：h5py.File(path,'r')、Path.read_text/read_bytes/stat、ffmpeg只以源作-i、输出路径固定本会话自有目录。可重查manifest/source_stat_before.json与source_readonly_verification.json；源未写、未移动，未碰他人进程。并未全量重hash巨大原视频，统计核查不冒称完整像素/光学同步认证。

未完成/需后续：
1. E82-L的demo3任务一致性：本会话保持P3 pending，不因同prompt自动纳入。
2. P3逐段示范质量、至少20个parent像素事件可见性与固定ROI批准；当前P3_ready=0，不以最终成功或模型伪标签当独立质量真值。
3. 非枚举派生/跨未知目录的穷尽近重复人工认证：已知源链、132、micro/fiveview已映射，未知默认quarantine；本轮稀疏像素指纹不是全库近重复认证。
4. 无新独立确认测试；历史test只是历史探索/回归，不能重命名为新盲测。
5. GPU target特征、100步profile与训练均未运行，属于E82-M/后续授权工作。

本轮不需要用户新增授权或决策。待L结论/人工质量证据到达后，另立允许池版本；禁止悄悄更改本次划分/缓存/统计。READY_data.json明列P1数据资格与P3待质审状态。
