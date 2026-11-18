# E87-Y：H20 yam 双臂数据只读盘点

日期：2026-11-13 BRT。完成：2026-11-13T15:14:49.625908-03:00。ID：E87-Y-20261113T1447。状态：盘点完成，P1建议待决定；训练0、GPU0、未提特征、未写P1_ready。

## 1. 结论与范围

H20可见17个真实yam来源、1503条记录。序列与ID去重后520个片段版本，其中89个旧接管切片完全被完整父录像覆盖。净量为 **431个非冗余片段、389个录制父源、366235个30Hz名义帧、3.391065小时**。45个审核切片只来自3条长录像，不能算45次独立重置。389父源中67个缺原始ID，用完整状态/动作序列SHA256暂作代理；不保证等于全域独立重置数。

保守建议 **49条/30451帧/0.281954小时**：29条来源标成功的遥操作示范、20条全程接管且执行action=expert的成功恢复。另189条/1.386139h待结果或控制来源复核。49条全部三路短段解码通过，但未构造严格时间窗口/有效窗口数和正式parent划分；不是训练ready或已批准训练池。

用户确认Mac到yam需内网后，将范围收缩为H20，未再尝试yam。先前两次BatchMode SSH在认证前超时。原机episodes/rollouts/备份及其他用户目录未核，不计背景335/89条或仅有历史传输记录。授权背景写11月14日，实际客户端/UTC换算BRT为11月13日，按实际日期记账。

同事目录只读，不碰其进程/权限，不解码全片，不复制视频，不下载/安装，不查询/占用GPU。仅本项目新目录写索引/审计/小联系表。LIBERO、仿真、retarget未进入清单。用户本轮明确允许的项目外只读来源优先于旧AGENTS.md冲突限制，其余边界照守。E81/E79/E80/E77的父链、时间/动作合同、非模型测试真值约束保留；SigLIP2预算采用E86较新实测。

## 2. 全来源清单

均在 /mnt/workspace/colleague-f/dataset。下表是来源记录，跨行重复，不能累加为净量。小时=帧数/30/3600，三视角不乘3。长度分布单位为秒。完整字段见manifest.jsonl和audit/source_summary.json。

|来源|记录|帧数|名义小时|长度秒min/median/p95/max|抽检/通过|
|---|---:|---:|---:|---|---|
|insert_tube_dagger_full_mixed_v21_20260930|62|46448|0.43007|11.17/22.82/42.26/46.07|20/20|
|insert_tube_dagger_lerobot_incomplete_DO_NOT_TRAIN|1|335|0.00310|11.17/11.17/11.17/11.17|1/1|
|insert_tube_dagger_raw|100|98562|0.91261|11.17/31.55/48.10/57.53|20/20|
|insert_tube_dagger_reviewed_45eps_20261025_raw|45|34912|0.32326|18.73/25.80/31.48/36.43|20/20|
|insert_tube_dagger_yam_rlt_rec_20261020_raw|20|13725|0.12708|17.77/22.12/27.81/29.23|20/20|
|insert_tube_mixed_dagger_v21_20260930|122|35328|0.32711|0.63/6.73/22.44/27.20|20/20|
|insert_tubes_lerobot_v21_100eps|100|69288|0.64156|15.33/22.93/28.87/31.93|20/20|
|insert_tubes_pi05_sft_merged_265eps_20261025|265|216487|2.00451|11.17/26.10/41.86/57.53|20/20|
|insert_tubes_pi05_sft_yam_rlt_rec_20eps_20261020|20|13725|0.12708|17.77/22.12/27.81/29.23|20/20|
|insert_tubes_sft_mixed_v21_20261013|200|167850|1.55417|11.17/26.42/43.68/57.53|20/20|
|place_slippers_dagger_full_mixed_v21_20261008|145|140988|1.30544|21.97/31.80/43.95/48.80|20/20|
|place_slippers_dagger_raw_20261008|44|50170|0.46454|28.57/37.75/46.35/48.80|20/20|
|place_slippers_lerobot_v21_20261007|101|90818|0.84091|21.97/29.90/36.67/45.27|20/20|
|place_slippers_raw_20261007|101|90818|0.84091|21.97/29.90/36.67/45.27|20/20|
|yam_labeled_raw_archive_20261023|135|117040|1.08370|11.17/27.60/45.79/57.53|20/20|
|yam_recovery_21eps_20261027_raw|21|8760|0.08111|11.17/13.77/16.03/18.27|20/20|
|yam_recovery_pi05_sft_21eps_20261027|21|8760|0.08111|11.17/13.77/16.03/18.27|20/20|

insert_tube_dagger_raw有100条含metadata有效记录，另2个空目录；历史103中的第三个排除空ID当前不存在。slippers dagger有44条（另有source_manifest小文件），rlt recovery当前20条。task2原始/LeRobot目录均不存在，只剩preparation旧记录，计0条可用。failed_encode目录无有效episode清单，不计；incomplete_DO_NOT_TRAIN只有1条，该片有正常raw别名，不从不完整目录取训练数据。

### 2.1 净任务与控制条件

|类别|净片段|录制父源|帧数|小时|
|---|---:|---:|---:|---:|
|slippers|145|145|140988|1.305444|
|tubes|286|244|225247|2.085620|
|human_recovery_all_frames|20|20|13725|0.127083|
|intervention_marked_execution_origin_unverified|21|21|8760|0.081111|
|mixed_or_autonomous|189|147|183644|1.700407|
|teleop_candidate|201|201|160106|1.482463|

teleop_candidate只是来源/控制候选，不自动证明成功。189个混合/自主片段中31条接管为0，158条部分接管，不能称纯人工；其中含reviewed45。20条成功恢复有全intervene且action=expert来源证据。另21条recovery接管标记全真，但专家建议可能与执行动作不同、成功未指定，控制来源单列待核。未见H20独立纯rollouts目录，不表示原机没有rollout。

520片段版本已有来源success173、failure7、未确认340（含89覆盖子片）；作者/审核可靠性未独立验收。来源结果仅作训练池筛选，不作测试真值、阈值或checkpoint选型；COMPLETE、BUILD_COMPLETE、可解码不能推断成功。

## 3. 去重、对应与父链

dedup_map.jsonl保存520个canonical片段、1503条别名、父源、原标签与排除原因。本轮CPU重新计算完整float32状态+动作序列SHA256、raw NPY/JSON及LeRobot Parquet小文件全文件SHA256；结合原ID/源path/审核manifest/明确区间，不全量hash大视频，不按任务名判重复。

- merged265=mixed200+rlt20+reviewed45；与raw/归档/旧转换版按序列hash联结。merged全部train不覆盖原validation标签，4个validation父源从候选训练池保守排除。
- slippers LeRobot101对应raw101，dagger full145=同101+raw44。archive135按原ID、path及序列hash联结。67条base100缺现存原始ID，用序列hash canonical，不编造ID。
- mixed_dagger的89个expert连续切片逐条与raw父录像[start,end)切片hash一致（89/89），净新增小时0；证据audit/segment_slice_verification.json。
- reviewed45有10条metadata缺source_recording，审核manifest和source_directory可补回。45条来自20261023_155309_0c1480ea、20261025_130547_39563ef3、20261025_132855_b2bd416d三父源。按父合并区间，无重叠。初查399父源已纠正为389，过程版本保留在audit。

净名义3.391065h；顶视主机收帧跨度合计3.538515h，后者含gap，不等于可训练小时。431个片段的区间并集366235帧，无重叠重复累计。仅代表H20当前可见范围，不是yam全库净量或完整重置数认证。

## 4. 相机、时间、解码与场景目视

每来源固定seed抽min(20,N)条，三路ffprobe、CPU ffmpeg中段1秒严格解码、起/中/近尾各15帧抽查。共321条/963路：全部H.264、640×480、30/1；短段失败0，抽样容器帧数与manifest不符0。另确保建议49条每条三路通过同样检查，已命中来源抽样时复用证据。未做全流解码，未抽位置仍可能损坏。

抽样没有同一短窗≥3次完全相同缩略帧的疑似冻结，不等于全片排除冻结或重复曝光。431净片段相机时间戳重复1380、倒退0、间隔>50ms共15607、>100ms共7366，最大400.240ms；同索引三路最大skew317.484ms，逐episode/相机p95绝对skew中位数27.305ms。三路不能按同帧号假定同步，30Hz容器不等于30Hz新曝光。

源码camera/realsense.py读图后用time.time()*1000打戳，是主机收帧近似时钟，非硬件曝光时间。缺硬件时钟/统一触发/状态可得时间/逐session标定证据。后续P1必须按因果last-available/age、gap及连续旧图mask构窗，W按真实时距选择；无法满足的窗口隔离，不平移标签或动作。

### 4.1 旧转换时间字段1000倍问题

以下部分字段delta约33333而非33.33；旧脚本有timestamp再次乘1000代码证据。仅审计识别量纲，未改原数组。正式读取须版本化解释并回raw验证，不能机械全源除1000。建议池选raw/归档原始ms NPY。

|来源|含1000倍时间字段episode/总数|
|---|---:|
|insert_tube_dagger_full_mixed_v21_20260930|29/62|
|insert_tube_mixed_dagger_v21_20260930|89/122|
|insert_tubes_pi05_sft_merged_265eps_20261025|120/265|
|insert_tubes_pi05_sft_yam_rlt_rec_20eps_20261020|20/20|
|insert_tubes_sft_mixed_v21_20261013|100/200|
|place_slippers_dagger_full_mixed_v21_20261008|145/145|
|place_slippers_lerobot_v21_20261007|101/101|

top/left/right在转换schema映射cam_high/双腕，与Piper顶视+双腕结构近似。yam是640×480 4:3，机位/视野/镜头/遮挡/外参/尺度不同，不能套Piper黑边裁剪、ROI或标定；相机序列号/外参未逐session证实，腕视由schema与随臂近景支持。

### 4.2 联系表及目视

review/task_{slippers,tube_dagger,two_tubes}_{01,02}.jpg：每任务固定seed随机12个去重episode，各起/中/近终三时刻×三视角，每任务2图，合36episode/324张源帧缩略图；另4张任务族总览。联系表与索引共约2.66MB，低于500MB。

本会话助手逐张目视检查6张任务联系表；这是场景审计，不是独立人工成功真值或新语言训练标注。拖鞋为粉米色洞洞鞋、小白台面橙支架；试管为透明管/蓝盖、橙色多孔架。背景大体同一白桌/浅墙角，光照变化小，变化主要为物体初始摆放、架子位置/朝向和随臂腕视。夹爪、管盖、架子造成腕视局部遮挡。

粗估1个物理桌面环境、2类物体布置、3–4个机位/摆放配置，属采样启发判断，不是完整场景聚类数。可见操作者手/袖/局部躯干进入边缘；324帧中未见清晰脸，不能保证全片无人脸/隐私。所见主要是yam双臂，未确认第三台机器人。review/visual_review.json保存目视总结，不回流作训练标签。

## 5. 机器人、动作、语言

两臂各6关节+1夹爪，左6/左夹爪/右6/右夹爪，共14D；关节rad，夹爪接口约定归一化[0,1]、0闭1开。记录有轻微越界，不裁剪后冒充真值。适配器备注底层夹爪单位/校准仍应按驱动核实。Piper夹爪为signed absolute米，不能因同为14D共用动作头/尺度。

schema区分follower joint_pos/gripper_pos与commanded action。teleop loop先取当前obs/agent动作，同tick记录；DAgger有student/expert/intervene/executed action。不是Piper processed的t+1反馈合同。未逐session核总线发送时钟，不能只凭字段宣称完整因果发送合同；不做shift。P1只用视觉，屏蔽状态/动作。

URDF：/mnt/workspace/colleague-f/yam_model/yam.urdf，SHA256 c0967cf85c4904a142c3aa2bb7453af609354ee94997900040fd0b3e1b756bb8。每臂6个revolute、夹爪2个prismatic，行程约0.04695m；不等于记录归一化夹爪。源码路径/行号/hash见audit/robot_signal_evidence.json，只读文本，未运行机器人代码。

语言主要是task_name和短prompt：insert_tube_dagger、place_slippers、放鞋英文、依次插两管英文；123条raw含Insert the tube into the rack.。LeRobot tasks也是短任务句。yam_labels.json、archive labels.json、episode_labels.json含outcome/event_frame/split/note；reviewed45还有已有reason/arm/资格/成功判据/原区间。读到11种非空note，不是独立验收的逐帧细粒度语言。未生成新语言标签，冻结评估parent未补标签。

## 6. P1建议与成本

proposed_p1_yam_list.json顶层entries是49条建议，optional_outcome_review为189条待核，不是获准池。含canonical/parent/path/源结果/split、全序列及小文件SHA256、视频大小/mtime/首64KiB SHA256；后者明确不是全视频hash。文件自身hash见同名.sha256。

|建议层|条数|名义小时|处理|
|---|---:|---:|---|
|A：来源成功遥操作|29|0.154870|archive对应33条中排4个validation父|
|A：rlt成功全程人工恢复|20|0.127083|全intervene且执行action=expert|
|B：结果待核的非rollout候选|168|1.305028|101放鞋+67插管|
|B：recovery控制/结果待核|21|0.081111|intervene全真不能独立证明执行来源|
|排除/隔离|193|1.722972|189混合/自主+4validation；89覆盖子片另排重|

reviewed45虽源标成功，仍含自主/部分接管，保守首批不纳。失败、测试/标定/unknown、坏视频、不完整、重复、holdout不进本建议；未来若要恢复正确接管区间应另立审核，不用最终成功证明整段动作正确。E81允许真实失败用于P1，本轮按用户筛选目标保守提案，扩池另决定。

yam双臂/三视图/抓取放置与插入接触可能帮助P1，但刚性蓝盖管/大拖鞋缺少扎带软体、细锁孔和穿带末端接触。建议小量parent/任务均衡先导，仅P1，不作P2/P3，不混动作头，保持Piper配对预算/冻结确认集；不用yam世界loss或来源结果选checkpoint。

### 6.1 冻结SigLIP2（纯推算，未执行）

E86固定每相机帧2×4×768 fp16=12288B，三路完整30Hz源帧，非全patch缓存。features_v3/status.json新编码941285帧/4313.276s，保守有效吞吐约218.23相机帧/s（含解码/检查/写入）；纯encoder探针1575–2411帧/s，不能冒充全pipeline。下表GPUh含25%余量、不含排队，yam解码/共享I/O仍可能改变耗时。

|池|相机帧|特征GB|pipeline分钟|含25%GPUh|
|---|---:|---:|---:|---:|
|proposed_A|91353|1.123|6.98|0.145|
|optional_B|449109|5.519|34.30|0.715|
|all_unique_nonredundant|1098705|13.501|83.91|1.748|

A池纯encoder约38–58秒，pipeline约7分钟、含余量约8.7分钟。全净池含余量约1.75GPUh，不在本轮<1GPUh测试授权内，本轮未执行。已有Piper裸缓存约16GB；A增1.12GB，全净池上界13.50GB，必须与旧缓存/其他run合计遵守100GB，不能各建一桶。

全部建议源已在H20，无需从yam复制。新副本估算A源约0.547GB，B约1.929GB，所有净代表源约6.581GB；仅H20当前路径字节统计，不能当原机全库大小。本轮未复制原片、未申请传输。/mnt约98.44%用、共享空闲约1.75TB，不是项目配额。

## 7. 风险、未完成、需要决定

1. 域窄：同一桌面为主，与Piper穿带的机器人/夹爪/物体/接触尺度不同，收益未测。
2. 时间：主机收帧非曝光、旧字段1000倍问题、最多400ms gap/317ms skew；正式接入前须窗口连续性、因果age、端点有效性验证。本轮不报可训练窗口数。
3. 父链：67父缺原ID；389为可见录制父源含hash代理，不是全库重置认证。45审核片归3父；P1接触测试视频也算泄漏，继续保守排4个validation父。
4. 来源标签作者未独立验收，可能是模型或混合审核；仅训练筛选，不做测试真值/门槛/选型。189条仍待结果/控制复核。短段通过不证明全片完好。
5. 画面有手/袖/躯干，未全片人脸审计；用户训练授权不等于公开原片授权。
6. 扫描前后raw metadata mtime均一致，A视频size/mtime一致，不是目录写锁。H20来源未见yam-shared-workbench；原机仍写目录未核，后续接入须重新锁快照/hash。

需要用户决定是否接受49条进入下一轮P1补充候选，以及是否安排189条独立结果/控制复核。采用后仍须正式parent分区/时间合同/资源run；本轮不训练、不提特征、不写P1_ready。不需批准复制，现有H20可原位读。

## 8. 交付与验证

- data/yam_inventory_v1/manifest.jsonl：1503源记录与canonical/parent/split/排除字段。
- dedup_map.jsonl：520版本/别名，89覆盖切片、431净片段。
- proposed_p1_yam_list.json + .sha256：49建议、189待核，明确proposal状态。
- review/：10张小联系表、任务/来源索引和目视总结，非训练标注。
- audit/：CPU脚本/日志/状态/解码/时间/来源合同hash/成本/初版父链；最终数以本报告、unique_summary及validation为准。
- inventory_receipt.json：最终关键文件SHA256与完成状态，不是P1_ready。

Mac执行ssh -o BatchMode=yes -o ConnectTimeout=15 <gpu-server>。服务器解释器/mnt/workspace/dengjianfeng/venvs/e29/bin/python，CUDA_VISIBLE_DEVICES为空，仅numpy/pyarrow/PIL/OpenCV/ffprobe/ffmpeg CPU。批量scan_h20.py、video_check.py、candidate_check.py采用setsid nohup和自身JSONL断点/状态文件。其余refine、fix_parent_and_time、summarize、finalize为本次审计整理，不调用训练/机器人。

验证：1503记录均关联520唯一canonical，431净片段/389父，89/89区间hash吻合，49候选三路通过、holdout交集0，三任务各12样本，hash通过。输出约40MB，联系表2.66MB；无原片复制、无GPU/训练/特征缓存/P1_ready。EXPERIMENT_LOG及worklog按BRT追加。

