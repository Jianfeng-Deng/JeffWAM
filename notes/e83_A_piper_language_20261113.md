# E83-A Piper Tier A 双语长文本标注完成报告

日期：2026-11-13T00:54:38.867998-03:00（BRT，UTC−3）。执行者：codex_gpt-6.1-sol。**227/227 个目标 train/P1 parent 均已产出；待标注数为0。** 其中1条存在明确记录的左腕源视频不可读例外，因此226条具有三视角证据，1条具有两视角证据；不将缺失视角伪称已完成。

## 范围和交付

主文件：`/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/lang_codex/A/lang_A.jsonl`；断点：`progress.json`；完整样例：`samples.md`；图像证据：`thumbnails/`；统计：`quality_summary.json`。

|分层|完成parent|剩余|
|---|---:|---:|
|demo4|129|0|
|demo6early|10|0|
|demo6native|11|0|
|demo3|77|0|

150个P3候选训练parent（demo4 129、demo6early 10、demo6native 11）全部覆盖；demo3尚未进入P3，另补77个train/P1 same parent，诊断teleop7排除。逐条按revision3核对parent、连通分量、locked split及allow_p1；未给dev/test连通分量添加语言标签。候选P3身份不表示质量放行。

每parent只选一个连续代表片段；优先完整示范，再在候选链内取最长连续示范。若看图后选同parent更合适候选，保存原候选链与双语选择理由；未跨文件拼接。短片段保留真实范围，按用户授权减少关键帧和阶段数，缩减理由写在short_clip_exception_justification。

总计1017段双语过程和1328条双语关键帧描述；28条带阶段/关键帧缩减说明。置信度分布：{"high": 53, "medium": 164, "low": 10}。

## 文本与证据自检

先完成5条试标并逐句复核后批量。各执行者实际逐张看三视角均匀与事件帧，再写起始画面、祈使任务指令、3种改写、过程、关键帧；逐条追加并保存断点。起始描述只用起始约1秒证据；不从L2结局生成描述或指令。指令保留本段实际手分工和可见顺序，不补未录到的取带、换手、拉紧或放置，不写偶然滑脱和重试。过程如实记录真实重复/分离等动作，但不把后面结果带入前段。孔口遮挡时只描述靠近、交叠或伸出带段，未强断瞬时进入、锁止或拉紧。

程序检查覆盖唯一parent/ID、train/P1、双语字段与句数、3改写及最短英文≤15词、源帧范围、阶段连续覆盖及证据截止、关键帧范围、缩略图存在性、metadata和使用限制。全部条目在写入时检查；最终独立范围审计另存shards/worker1/final_scope_audit.json。最终同步各分片自检修订，清单见final_shard_sync.json。

下面按纯文本计算，去重率=规范化空白与大小写后的唯一文本数/文本条数；英文长度为词数，中文为汉字数；过程/关键帧按每段/每帧，改写按长度分别统计，不把证据路径和JSON标点计入长度。

|字段|文本条数|去重率|平均长度|最短–最长|
|---|---:|---:|---:|---:|
|scene_description_en|227|100.00%|136.44|106–169|
|task_instruction_en|227|69.16%|114.93|57–176|
|instruction_variants_short_en|227|40.09%|9.79|7–13|
|instruction_variants_medium_en|227|67.40%|33.85|19–45|
|instruction_variants_long_en|227|67.84%|55.68|34–81|
|process_narration_en|1017|100.00%|43.70|22–80|
|keyframe_captions_en|1328|100.00%|34.35|21–54|
|scene_description_zh|227|100.00%|177.99|137–243|
|task_instruction_zh|227|69.16%|143.62|72–227|
|instruction_variants_short_zh|227|40.97%|11.41|8–19|
|instruction_variants_medium_zh|227|66.96%|36.69|25–52|
|instruction_variants_long_zh|227|67.84%|68.00|44–104|
|process_narration_zh|1017|100.00%|54.36|28–105|
|keyframe_captions_zh|1328|99.85%|43.06|27–70|

所有完整起始画面描述均不同；过程和关键帧各自依据实际差异写作。任务指令和最短改写允许同任务、同实际分工的相同表达，未为提高去重率添加不实细节。复制式重复检查结合逐条画面核对；字符不同不等同事实正确，仍保留同labeler自检的局限。

## 源数据与限制

全程CPU只读原位抽帧，后台抽帧使用setsid/nohup；没有训练、GPU占用、模型下载或视频复制。缩略图合计0.949 GiB，小于5GB。源入口两个SHA256仍与开始一致，revision3未变。现存源stat guard覆盖371个文件，size/mtime/ctime/inode最终复核无变化；早期并发使用同名manifest的guard文件曾覆盖，不能将现存guard声称为全部源文件的独立初始快照，抽帧脚本每clip内部前后只读检查仍执行。未改动同事源目录或进程。

唯一缺视角：piper_4dc107335389f5711a0f65e0（demo3 episode142）左腕视频在所有现存别名路径均Permission denied，HDF5没有替代图像。使用实际可读top/right wrist两路，views中不列左腕，显式unavailable_views/evidence_gap_reason/annotation_completeness=two_view_evidence_exception，confidence=low；左腕描述写不可读，不编造。无需用户替我们决定抽帧，源权限限制如实保留。

视频事件只通过抽帧夹定区间；阶段边界不是逐帧接触检测。极短/近静止示范只提供所见姿态与准备目标，不能当作完整穿带动作示范。模型文本仅用于language_supervision_training，所有行均禁止test_ground_truth、threshold_source、model_checkpoint_selection。

## 现有短指令及术语

现有原文：**Pick up the zip tie and thread its pointed end through its own locking head to form a loop.**

demo4所查原始HDF5 attrs.prompt含此句；demo6所查raw prompt为空，其处理版meta/tasks.jsonl包含该固定句。逐源字段记录见existing_short_instructions.json。先读E82-D、E82-L/SPEC、E77及8份decisions_v2；沿用zip_tie→zip tie（扎带）、lock_head→locking head（锁头）、tail/strap（带身）、tip（尖端）。hand-over arm为接收臂、from_arm为移交臂；具体episode分工按自身画面核实，未把历史事件约定套成未见动作。

## 三个完整双语样例原文

# E83-A：三个完整双语样例原文

仅供 language_supervision_training；禁止测试真值、阈值、checkpoint 选择。元数据与全部证据帧路径以 lang_A.jsonl 为准。

## E83_A_d0e4ec936fbd786407be1324

parent=pc_20b529c4a05b47402a61；split=train；demo4_teleop_realsense_v2_hdf5/episode_3；confidence=high；source_frame=[0, 1992]

Longest continuous raw demonstration among the P3-candidate segments of this parent; no concatenation across source files.
选择此 parent 的 P3 候选片段中源时钟连续范围最长的一段示范；不跨文件拼接。

### scene_description

EN: The top camera shows a white table with rectangular cable covers and a transparent bottle-shaped container on the right. A long orange zip tie leans upward from the container, with its locking head at the raised end and its lower portion running down toward the container. A separate loop-shaped orange tie lies to the lower left of the container, and other loose orange segments are cut off by the bottom edge. The left wrist camera looks across the empty central tabletop toward black office chairs, with its open gripper jaws at the bottom. The right wrist camera centers the clear container and the upright tie between its open jaws, giving a closer view of the raised head but not a resolved view of the locking opening. Neither gripper is visibly holding the target at the start, and the target tip low in or behind the container cannot be located precisely.

ZH: 顶视相机显示白色桌面，桌上有长方形走线盖，右侧放着一个透明瓶状容器。一根长橙色扎带从容器处向上倾斜，锁头位于抬起的一端，下部带身延伸到容器附近。容器左下方另放着一根环状橙色扎带，还有散放带段被画面下缘截断。左侧腕部相机越过中央较空的桌面朝向黑色办公椅，张开的夹爪位于画面底部。右侧腕部相机把透明容器和竖起的扎带置于张开的夹爪之间，能较近地看到抬起的锁头，但看不清锁孔开口。起始画面未见任一夹爪持住目标，目标在容器内或后方的尖端位置无法精确确定。

### task_instruction

EN: Pick up the upright zip tie from the clear container and thread its tip through its own locking head to make a loop. Bring the right gripper around the strap beneath the head and close on it. Lift the tie from the container and present the head between the two grippers. Secure the head with the left gripper before letting go with the right gripper. Keep the head raised while the right gripper moves down the hanging strap to grasp near its tip. Turn the right wrist and lift the tip back beside the head, allowing the middle of the strap to curve below the hands. Align the pointed end with the head opening and advance it through while the left gripper maintains the head position. Take hold of the emerging strap and pull it farther through to preserve the closed loop. Bring the loop down to the table, open the grippers, and move clear of the released tie.

ZH: 从透明容器中取出竖起的扎带，把尖端穿过自身锁头形成环。把右夹爪移到锁头下方的带身两侧并夹紧。将扎带提离容器，把锁头送到两只夹爪之间。先用左夹爪稳稳夹住锁头，再松开右夹爪。保持锁头抬起，让右夹爪沿下垂带身移动，在尖端附近抓住。转动右腕，把尖端抬回锁头旁，让带身中部在双手下方弯曲。将尖端对准锁孔，在左夹爪维持锁头位置时向孔内送入。夹住穿出的一段，继续向外拉，以保留闭环。把环降至桌面，张开夹爪并离开释放的扎带。

### instruction_variants

1. EN: Make a loop by threading the zip tie through its locking head.

   ZH: 将扎带尖端穿过锁头形成环。

2. EN: Use the right gripper to lift the upright tie and hand its locking head to the left gripper. Grasp the free tip with the right gripper, feed it through the left-held head, and pull the strap through before placing the loop down.

   ZH: 用右夹爪提起竖立扎带，把锁头交给左夹爪。右夹爪抓取自由尖端，将其送入左手所持锁头并拉出带身，再放下闭环。

3. EN: Lift the tie from the clear container with the right gripper. Transfer its locking head to the left gripper. Use the right gripper to find and hold the free tip farther down the strap. Curve the tip back toward the held head and align it with the opening. Thread the tip through and pull the emerging strap while supporting the head. Lower and release the loop on the table.

   ZH: 右夹爪从透明容器中提起扎带。把锁头交给左夹爪。右夹爪沿带身向下找到并夹住自由尖端。将尖端弯回所持锁头处并对准孔口。支撑锁头，把尖端穿过并拉出露出的带段。将闭环降到桌面并释放。

### process_narration

**source_frame 0–221**

The right arm reaches toward the upright tie in the container and lifts it by the strap below the head. At frame 221 the right wrist camera shows the raised head above the jaws while the left camera still overlooks the central tabletop.

右臂接近容器中的竖立扎带，在锁头下方持住带身并提起。第221帧右腕画面显示锁头高于夹爪，左腕相机仍俯向桌面中央。

**source_frame 222–553**

The left gripper approaches the head and the hands meet above the table. By frame 553 the left gripper supports the head and the right gripper has moved lower along the hanging strap.

左夹爪靠近锁头，两只手在桌面上方会合。到第553帧，左夹爪支撑着锁头，右夹爪已沿下垂带身向下移。

**source_frame 554–1107**

The left arm keeps the head elevated while the right hand works down the strap. The right wrist view first shows the free end away from the head, then follows it upward as the right arm starts returning toward the left hand.

左臂保持锁头抬起，右手沿着带身向下操作。右腕画面先显示远离锁头的自由端，之后随着右臂开始向左手返回，视角跟随自由端上移。

**source_frame 1108–1328**

The right gripper brings the tip alongside the left-held head and bends the strap into a hanging curve. At frame 1328 the ends are close, but the sampled images do not yet show an unambiguous section protruding through the head.

右夹爪把尖端送到左手所持锁头旁，带身弯成下垂的弧线。第1328帧两端已靠近，但抽样画面中还没有明确可辨的穿出带段。

**source_frame 1329–1549**

At frame 1439 a narrow orange section is visible beyond the head, with the right gripper close above it. The right hand then shifts across the head, and at frame 1549 the left wrist view shows the head with strap extending to either side.

第1439帧锁头另一侧可见一段细橙色带段，右夹爪紧靠其上方。随后右手越过锁头调整位置，第1549帧左腕画面显示锁头两侧都有带身延伸。

**source_frame 1550–1771**

The right gripper withdraws from the head as the left gripper retains the curved tie. The hands move down toward the table; by frame 1771 the top view shows an additional loop-shaped tie at the lower left and the wrist views have returned toward the tabletop.

右夹爪离开锁头，左夹爪继续持着弯曲的扎带。双手向桌面下降；到第1771帧，顶视左下方可见新增的环状扎带，腕视也已重新朝向桌面。

**source_frame 1772–1992**

The grippers move away from the placed ties and return near the bottom of the wrist images. The final views show the clear container standing without the previously upright target and the loop-shaped ties lying on the table.

两只夹爪离开放置扎带的位置，回到腕视画面底部附近。末段画面中透明容器仍立在桌上，原先竖起的目标已不在容器内，环状扎带放在桌面上。

### keyframe_captions

**source_frame_index=0**

An upright orange tie leans out of the clear container to the right of the cable covers, with a separate loop-shaped tie lower left. Both wrist views show open grippers, and the target tip near the container is not distinguishable.

一根橙色扎带从走线盖右侧的透明容器中斜立而出，左下方另有环状扎带。两路腕视可见张开的夹爪，容器附近的目标尖端无法辨清。

**source_frame_index=221**

The right gripper holds the long tie below its raised head. The right wrist view exposes the head against the background chairs, while the top view shows the tie lifting from the container and the left wrist view remains mostly unobstructed.

右夹爪持在长扎带抬起的锁头下方。右腕画面中的锁头以背景椅子为衬，顶视可见扎带正提离容器，左腕画面此时大多未被遮挡。

**source_frame_index=443**

The grippers are close together around the locking head above the white table. The left wrist camera sees the head near its own jaws, while the opposite gripper obscures much of the right wrist foreground.

两只夹爪在白桌上方靠拢到锁头周围。左腕相机看到锁头靠近自身夹爪，右腕前景大部分被对侧夹爪挡住。

**source_frame_index=885**

The head remains at the left gripper while the right hand holds farther along the strap. The right wrist camera shows a free orange end projecting above its jaws, and the top view shows a long strap spanning the separated hands.

锁头仍位于左夹爪处，右手持在带身更远的位置。右腕相机可见自由橙色端伸出夹爪上方，顶视显示长带身跨在分开的双手之间。

**source_frame_index=1217**

The right-held tip is upright beside the head held by the left gripper. Both wrist views show the nearby opposing hand; the tip is still visibly separate above the right jaws at this sampled frame.

右手持有的尖端竖在左夹爪所持锁头旁。两路腕视都可见近处的对侧手，本帧尖端在右夹爪上方仍可辨为独立的一端。

**source_frame_index=1439**

The hands crowd around the locking head, with an orange section projecting beyond it. The left wrist view shows the curved strap and head more clearly, while the top view is partly covered by the right gripper.

双手聚在锁头附近，一段橙色带身越过锁头伸出。左腕画面较清楚地显示弯曲带身和锁头，顶视则被右夹爪部分挡住。

**source_frame_index=1660**

The right gripper is moving away from the left-held head. The left wrist view shows strap extending across the head and curving downward, whereas the right wrist view now mostly faces the table and nearby arm.

右夹爪正离开左手所持锁头。左腕画面可见带身横穿锁头并向下弯曲，右腕画面此时主要朝向桌面和附近机械臂。

**source_frame_index=1992**

The hands have withdrawn, and loop-shaped orange ties lie at the front of the table. The clear container no longer holds the upright target seen at the start, and both wrist views again show the tabletop and office background.

双手已经退出，橙色环状扎带放在桌面前部。透明容器中已没有起始时竖立的目标，两路腕视重新显示桌面与办公背景。

## E83_A_edf1bbc3281ed3e0e9f6ddc3

parent=pc_78b4c204fae581b72ab3；split=train；demo4_dagger_realsense_v2_hdf5/episode_5；confidence=medium；source_frame=[0, 2649]

Longest continuous raw demonstration among the P3-candidate segments of this parent; no concatenation across source files.
选择此 parent 的 P3 候选片段中源时钟连续范围最长的一段示范；不跨文件拼接。

### scene_description

EN: The clip starts with both robot grippers already raised close together over the white table. The left gripper holds an orange locking head with its attached strap hanging down, while the right gripper holds the other end near the head and the strap bends below the visible working area. In the top view the head is near the center between the jaws, with rectangular cable covers behind it and loose orange ties farther down the table. The left wrist camera shows the head directly ahead of its jaws, a curved strap extending to the right, the opposing gripper, and an office workspace with a brown divider in the background. The right wrist camera looks along a narrow orange strap toward the head beside the opposing gripper, whose housing obscures part of the contact area. Both grippers appear closed around their respective portions, but their inner contact surfaces and the exact relation between the tip and the locking opening cannot be fully seen at the start.

ZH: 片段开始时，两只机器人夹爪已经抬起，在白色桌面上方相互靠近。左夹爪持着橙色锁头及其下垂带身，右夹爪在锁头附近持着另一端，带身在可见操作区域下方弯曲。顶视中锁头位于夹爪之间的中央附近，后方是长方形走线盖，桌面更下方散放着橙色扎带。左侧腕部相机可见自身夹爪正前方的锁头、向右弯曲延伸的带身和对侧夹爪，背景是有棕色隔板的办公工作区。右侧腕部相机沿细长橙色带身望向对侧夹爪旁的锁头，夹爪外壳挡住了部分接触区域。两只夹爪看起来都已夹住各自的带段，但内侧接触面以及尖端与锁孔的准确关系在起始画面中无法完全看清。

### task_instruction

EN: Use the already held zip tie to form a closed loop by passing its tip through its own locking head. Maintain the left gripper's support of the locking head rather than reaching for another tie. Keep the right gripper near the free tip and leave enough of the pointed end exposed for insertion. Bring that tip toward the opening while allowing the strap to curve below the two hands. Use small right-wrist position and orientation adjustments to line up the tip with the passage through the head. Advance the tip through the opening while the left gripper keeps the head steady. Move the right gripper to the emerging section and draw the strap through without pulling the head out of the left gripper. Retain the resulting loop in the left gripper and move the right gripper away from the head.

ZH: 操作已经持住的这根扎带，将尖端穿过它自身的锁头以形成闭环。维持左夹爪对锁头的支撑，不必再去取另一根扎带。右夹爪持在自由尖端附近，留出足够的尖部用于穿入。将尖端送向孔口，让带身在两手下方弯曲。小幅调整右腕的位置和朝向，把尖端与锁头内的通路对齐。左夹爪稳住锁头，右夹爪沿孔口推进尖端。把右夹爪移到穿出的带段处并向外拉，不要把锁头拉离左夹爪。用左夹爪保持形成的环，让右夹爪离开锁头。

### instruction_variants

1. EN: Pass the held zip tie tip through its own locking head.

   ZH: 将手持扎带的尖端穿过自身锁头。

2. EN: Keep the locking head in the left gripper and align the right-held tip with its opening. Feed the tip through, grasp the emerging section, and draw it through while retaining the loop in the left hand.

   ZH: 左夹爪保持锁头，右手持尖端对准孔口。将尖端穿入，夹住露出的带段并向外拉，同时由左手保持闭环。

3. EN: Start with the locking head supported in the left gripper. Hold the free tip with the right gripper and curve the strap below the hands. Adjust the tip until it lines up with the opening in the head. Advance the tip through while stabilizing the head. Regrasp and pull the emerging strap, then withdraw the right gripper while the left gripper retains the loop.

   ZH: 从左夹爪已支撑锁头的状态开始。右夹爪持住自由尖端，使带身在双手下方弯曲。调整尖端，使其与锁孔对齐。稳住锁头，将尖端向孔内推进。重新抓住穿出的带段并向外拉，随后右夹爪退出，左夹爪保持闭环。

### process_narration

**source_frame 0–589**

Both grippers remain close to the head, with the left gripper supporting it and the right holding the other end. The tip is adjusted beside the head, but the grippers hide the exact contact through frame 589.

两只夹爪持续靠近锁头，左夹爪支撑锁头，右夹爪持着另一端。尖端在锁头旁调整，但到第589帧为止，准确接触位置仍被夹爪挡住。

**source_frame 590–900**

At frame 600 a short orange section is visible beyond the head, and the nearby right gripper changes its grip and position over the following sampled frames. The section projecting beyond the head becomes longer before the right gripper moves back across the head at frame 900.

第600帧可见一小段橙色带身越过锁头伸出，随后抽样帧中附近的右夹爪改变了抓取和位置。露出锁头的带段逐渐变长，之后第900帧右夹爪又移回锁头附近。

**source_frame 901–1324**

By frame 950 the strap hangs as a long, nearly straight length from the left-held head, and the earlier projecting section is no longer visible. The right gripper moves down and around the strap; the sampled views do not resolve whether it was deliberately withdrawn or lost during this movement.

到第950帧，左手所持锁头下方垂着较长、近乎直线的带身，先前伸出的带段已看不见。右夹爪沿带身向下并绕行，抽样画面不能分辨这一变化是主动抽回还是运动中失去连接。

**source_frame 1325–1700**

The right arm brings the free end back toward the left-held head for another alignment. At frames 1472 through 1700 the ends are again adjacent, but no clearly projecting section is visible beyond the head in these views.

右臂把自由端重新送回左手持有的锁头处对齐。第1472至1700帧两端再次靠近，但这些视角中没有清楚可见的带段从锁头另一侧伸出。

**source_frame 1701–1913**

The right gripper moves away from the head and down the hanging strap again. By frame 1913 the strap is extended between the raised left hand and the lower right hand, with the head still held on the left.

右夹爪再次离开锁头，沿下垂带身向下移动。到第1913帧，带身伸展在抬高的左手和较低的右手之间，锁头仍由左侧持住。

**source_frame 1914–2208**

The right gripper returns with the free end and brings it beside the head. At frame 2208 the tip-to-head junction is close to the gripper edges and remains partly occluded, so the visible proximity alone does not establish passage.

右夹爪带着自由端返回，把它送到锁头旁。第2208帧尖端与锁头的交汇处靠近夹爪边缘，仍被部分遮挡，因此仅凭靠近状态不能确认穿过。

**source_frame 2209–2400**

At frame 2300 an orange section is visible extending beyond the locking head in the left wrist view. The right gripper moves over that emerging section, and by frame 2400 the head has strap extending across both sides.

第2300帧左腕画面可见一段橙色带身越过锁头伸出。右夹爪移到这段露出的带身上方，到第2400帧锁头两侧都有带身延伸。

**source_frame 2401–2649**

The right hand draws away from the head, with a curved loop hanging below the left gripper. At frame 2649 the left wrist view shows the head and strap across it while the right arm is clear of the head; the clip ends with the tie still held rather than placed on the table.

右手离开锁头，左夹爪下方垂着弯曲的环。第2649帧左腕画面显示锁头及横穿其中的带身，右臂已经避开锁头；片段结束时扎带仍被持住，没有放到桌上。

### keyframe_captions

**source_frame_index=0**

The left gripper already holds the locking head and the right gripper brings the other end close alongside it. The top camera shows the crowded hands above the cable covers, while both wrist views partially obscure the contact region.

左夹爪已经持住锁头，右夹爪把另一端靠在它旁边。顶视显示走线盖上方拥挤的双手，两路腕视都部分遮住了接触区域。

**source_frame_index=600**

A short orange section projects beyond the head in the left wrist image, with the right gripper immediately beside it. The top and right wrist views show the long strap below the head and substantial occlusion by the hands.

左腕画面中一小段橙色带身伸出锁头另一侧，右夹爪紧邻其旁。顶视和右腕画面可见锁头下方的长带身，以及双手造成的大面积遮挡。

**source_frame_index=950**

The left hand holds the head high while a long orange strap hangs down almost straight. The earlier short projection is no longer visible, and the right wrist view shows the opposite arm and the descending strap rather than a resolved head junction.

左手在高处持着锁头，长橙色带身近乎直线地下垂。先前的小段露出部分已不可见，右腕画面显示对侧机械臂和向下延伸的带身，锁头连接处看不清。

**source_frame_index=1472**

The right-held end is back beside the head at the left gripper. The left wrist view shows the head with a curved strap arriving from the right, but the jaws partly hide the opening in both close views.

右手持有的一端已经回到左夹爪锁头旁。左腕画面显示锁头和从右侧弯来的带身，但两路近景中的孔口都被夹爪部分遮挡。

**source_frame_index=1913**

The grippers are separated vertically around a long hanging strap, with the left hand retaining the head. The left wrist camera sees the head and background workspace, and the right wrist camera looks along the strap past the opposing arm.

两只夹爪沿长下垂带身上下分开，左手继续持住锁头。左腕相机可见锁头和后方工作区，右腕相机沿带身望向对侧机械臂。

**source_frame_index=2208**

The right gripper again brings a broad orange end near the left-held head. The top and wrist views show close alignment, but the overlapping jaws prevent a clear view of passage at this frame.

右夹爪再次把较宽的橙色端部送到左手锁头附近。顶视和腕视显示两端紧靠对齐，但重叠夹爪使本帧无法清楚看见贯穿状态。

**source_frame_index=2300**

An orange segment now extends to the far side of the locking head in the left wrist view. The right gripper lies across the head, so the top image is more occluded than the wrist image of the emerging strap.

左腕画面中，橙色带段已经伸到锁头另一侧。右夹爪横在锁头附近，因此顶视比显示穿出带段的腕视受到更多遮挡。

**source_frame_index=2649**

The left gripper retains a small hanging loop and the right gripper is displaced to the right. The left wrist view shows the head with strap across it, while the right wrist view follows a loose projecting section away from the head.

左夹爪保持着一个下垂小环，右夹爪移到了右侧。左腕画面显示锁头及横穿其中的带身，右腕画面则沿着离开锁头的自由带段望去。

## E83_A_cf3d999d8bbb6e17a0e45e39

parent=pc_1db80a3342895e07d155；split=train；demo6_teleop_realsense_v2_hdf5/episode_16；confidence=high；source_frame=[0, 3448]

Longest continuous raw demonstration among the P3-candidate segments of this parent; no concatenation across source files.
选择此 parent 的 P3 候选片段中源时钟连续范围最长的一段示范；不跨文件拼接。

### scene_description

EN: The top camera shows a yellow slotted basket on a white table against a plain wall, with several orange straps lying on the tabletop to the right. Long orange zip ties stand in the basket with their locking heads uppermost and their straps dropping toward its base. The visible upright ties are spaced apart above the basket, although their tips are difficult to separate from the slots and lower rim. The left wrist camera initially sees mostly the wall and its own open gripper jaws, with the basket outside most of that view. The right wrist camera shows raised locking heads and curved upright straps above a small visible portion of the basket, framed by its open jaws. Neither gripper visibly holds a target yet, and most of the arm bodies are outside the top view; the head openings and the exact placement of the lower tips are too small or occluded to specify.

ZH: 顶视相机显示素墙前白桌上的黄色镂空筐，桌面右侧还散放着橙色带身。长橙色扎带竖在筐内，锁头位于上端，带身向筐底垂下。筐上方可见的竖立扎带彼此分开，但尖端与筐底孔条、下部筐沿不易区分。左侧腕部相机初始主要看到墙面和自身张开的夹爪，筐大多不在该视野内。右侧腕部相机可见抬起的锁头、弯曲竖立的带身和少量筐体，张开的夹爪位于两侧前景。尚未见夹爪持住目标，两臂主体大多在顶视画面外，锁孔及下部尖端的具体位置过小或被遮挡，无法明确说明。

### task_instruction

EN: Take the upright zip ties in the basket one at a time and thread each tip through its own locking head to make a closed loop. Approach the selected strap with the right gripper and grasp it beneath the head. Lift that tie and bring its head toward the left gripper above the basket. Close the left gripper around the head, then release the right gripper's initial hold. Move the right gripper down the hanging strap and grasp near the free tip, leaving the pointed end protruding. Bend the strap and lift the tip back toward the head supported by the left hand. Adjust the tip direction to the opening and feed it through while holding the head stable. Grip the emerging strap and pull it farther through, then lower and release the loop into the basket. Continue with the other upright ties using the same head transfer and tip-handling order.

ZH: 逐根操作筐中竖起的扎带，把每根尖端穿过自身锁头形成闭环。用右夹爪接近选中的带身，在锁头下方抓住。提起这根扎带，把锁头送到筐上方的左夹爪附近。左夹爪夹住锁头后，再松开右夹爪的初始抓持。右夹爪沿下垂带身向下移动，在自由尖端附近抓取并留出尖部。弯曲带身，把尖端抬回左手所持锁头处。调整尖端方向对准孔口，在稳住锁头的同时向内送入。夹住穿出的带身并继续向外拉，随后把环降低并释放到筐内。按相同的锁头交接和尖端操作顺序继续处理其余竖立扎带。

### instruction_variants

1. EN: Thread the zip ties into loops, one at a time.

   ZH: 逐根穿扎带，形成闭环。

2. EN: Take a tie from the basket with the right gripper and hand its head to the left gripper. Use the right hand to return the tip to the head, thread and pull it through, and release the loop into the basket before taking another tie.

   ZH: 右夹爪从筐中取扎带，把锁头交给左夹爪。右手将尖端弯回锁头处并穿入拉出，把环放回筐内后再取下一根。

3. EN: Choose an upright tie with the right gripper and lift it above the basket. Transfer its locking head to the left gripper. Hold the free tip with the right gripper and curve the strap back toward the head. Align the tip with the opening and feed it through. Pull the emerging strap with the right hand while the left hand holds the head. Lower the loop into the basket and repeat for the remaining upright ties.

   ZH: 右夹爪选取竖立扎带并提到筐上方。将锁头交给左夹爪。右夹爪持住自由尖端，把带身向锁头处弯回。尖端对准孔口后向内送入。左手持住锁头，右手拉出穿出的带身。把环降到筐内，再处理其余竖立扎带。

### process_narration

**source_frame 0–360**

The right gripper approaches the upright ties and raises one by the strap below its head. The left gripper meets it above the basket, and by frame 360 the left wrist view shows the head held at the left jaws.

右夹爪靠近竖立扎带，在锁头下方持住一根带身并提起。左夹爪在筐上方与其会合，到第360帧左腕画面可见锁头被左夹爪持住。

**source_frame 361–766**

The right arm moves down the strap and lifts the free tip back toward the head, forming a hanging curve. At frame 766 the tip is alongside the head but still visibly offset in the left wrist image, with the right jaws below it.

右臂沿带身向下移动，再把自由尖端抬回锁头附近，带身形成下垂弧线。第766帧左腕画面中尖端位于锁头旁但仍有可见偏移，右夹爪在其下方。

**source_frame 767–1149**

The hands close the distance around the head, and at frame 958 a short orange section is visible beyond it. By frame 1149 the left wrist view shows a longer emerging section while the top view shows a curved loop below the left-held head.

双手在锁头周围靠近，第958帧可见一小段橙色带身伸到锁头另一侧。到第1149帧，左腕画面显示更长的露出带段，顶视显示左手锁头下方弯曲的环。

**source_frame 1150–1532**

The hands lower from the first loop and return toward the upright ties in the basket. At frame 1341 the right wrist view shows a new raised head near the right jaws, and by frame 1532 the head is supported by the left hand with the right hand lower on its strap.

双手从第一个环的位置降低，并回到筐中的竖立扎带附近。第1341帧右腕画面显示新的抬起锁头靠近右夹爪，到第1532帧锁头由左手支撑，右手位于带身更下方。

**source_frame 1533–1916**

The right hand raises this tie's free end and aligns it beside the left-held head. At frame 1916 the left wrist image shows strap extending beyond the head and the top view shows a hanging loop between the two hands.

右手抬起这根扎带的自由端，与左手锁头并排对齐。第1916帧左腕画面可见带身越过锁头伸出，顶视显示双手之间下垂的环。

**source_frame 1917–2299**

The right hand shifts across the head and then moves away as the left hand lowers the loop toward the basket. At frame 2299 the arms are back near the remaining upright tie, with open jaws and the basket visible in both wrist views.

右手越过锁头调整位置后退出，左手将环降向筐内。第2299帧双臂已回到剩余竖立扎带附近，两路腕视可见张开的夹爪和筐。

**source_frame 2300–2682**

The remaining upright tie is lifted and the head is transferred toward the left gripper. The right hand works down the strap; by frame 2682 a free orange end projects above the right jaws, separate from the left-held head.

剩余竖立扎带被提起，锁头向左夹爪交接。右手沿带身向下操作，到第2682帧一段橙色自由端伸出右夹爪上方，与左手锁头分开。

**source_frame 2683–3065**

The right gripper bends the tip back toward the head and adjusts it alongside the opening. At frame 3065 an orange section is visible beyond the head in both wrist views, while the top camera shows the broad curved strap below the hands.

右夹爪把尖端弯回锁头处，在孔口旁调整。第3065帧两路腕视都可见橙色带段伸到锁头另一侧，顶视显示双手下方较宽的弯曲带身。

**source_frame 3066–3448**

The right gripper withdraws from the head and the left hand retains a small loop at frame 3256. The arms then lower and move out of the top view, leaving loop-shaped straps in the basket and the loose straps still on the table to its right.

右夹爪离开锁头，第3256帧左手保持着一个小环。随后两臂降低并移出顶视，筐内留有环状带身，筐右侧桌面上的散放带身仍在。

### keyframe_captions

**source_frame_index=0**

Upright orange ties stand above a yellow slotted basket, with loose orange straps lying to its right in the top view. The left wrist sees mostly wall and open jaws, and the right wrist sees raised heads above the basket with the lower tips obscured.

橙色扎带竖在黄色镂空筐上方，顶视中筐右侧散放着橙色带身。左腕主要拍到墙面和张开的夹爪，右腕可见筐上方抬起的锁头，下部尖端被挡住。

**source_frame_index=270**

The right gripper holds a selected tie below its head and the left gripper approaches from the other side. Both wrist views show the raised head and strap near the opposite hand, with the basket below in the top image.

右夹爪在锁头下方持住选中的扎带，左夹爪从另一侧靠近。两路腕视可见抬起的锁头及靠近对侧手的带身，顶视中筐位于下方。

**source_frame_index=766**

The left gripper holds the head while the right-held tip stands alongside it, leaving a visible offset in the left wrist view. The top image shows the curved strap and the right wrist view looks along a narrow strap past the opposing arm.

左夹爪持住锁头，右手尖端竖在其旁，左腕画面中两者仍有可见偏移。顶视显示弯曲带身，右腕沿细长带段望向对侧机械臂。

**source_frame_index=1149**

A short length of strap projects through the head, with a hanging loop below the left hand. The left wrist view shows the emerging section clearly, while the right wrist view is displaced toward the wall and nearby gripper.

一段带身穿过锁头伸出，左手下方垂着环。左腕画面清楚显示露出的带段，右腕视角则转向墙面和附近夹爪。

**source_frame_index=1532**

The head of another raised tie is supported by the left gripper and the right hand holds farther down the strap. The left wrist view centers the head, and the right wrist view follows the narrow strap toward the opposite jaws.

另一根抬起扎带的锁头由左夹爪支撑，右手持在带身更下方。左腕画面以锁头为前景中央，右腕沿细长带身望向对侧夹爪。

**source_frame_index=1916**

The second working tie forms a hanging curve, and an orange section projects beyond its head in the left wrist view. The right gripper lies close to the head, partly crowding the top and right wrist views.

第二根正在操作的扎带形成下垂弧线，左腕画面可见橙色带段伸到锁头另一侧。右夹爪紧靠锁头，部分占据顶视和右腕视野。

**source_frame_index=2299**

The grippers have moved back toward the remaining upright tie above the basket. The left wrist image sees its raised head and lower strap near open jaws, while the right wrist image sees the opposite arm beside the basket.

双夹爪已回到筐上方剩余的竖立扎带附近。左腕画面可见张开夹爪旁抬起的锁头和下部带身，右腕画面可见筐旁的对侧机械臂。

**source_frame_index=2682**

The left hand holds the last selected head high and the right hand holds a free end farther down the strap. The right wrist view shows the tip projecting upward above its jaws, and the left wrist view keeps the held head close to the foreground.

左手在高处持住最后选中的锁头，右手在带身更下方持着自由端。右腕画面可见尖端向上伸出夹爪，左腕画面保留近前的锁头。

**source_frame_index=3065**

The right gripper is close across the locking head and a strap section extends beyond the head. Both wrist views expose the emerging section, while the top view shows the curved orange strap dropping below the hands.

右夹爪紧靠并横在锁头附近，一段带身越过锁头伸出。两路腕视都露出穿出的带段，顶视可见橙色弯曲带身垂在双手下方。

**source_frame_index=3448**

The basket contains loop-shaped orange straps and no upright ties are visible above it. The arms are outside the top view, loose straps remain on the table to the right, and the wrist views show open jaws near the basket and wall.

筐内有橙色环状带身，筐上方已看不到竖立扎带。机械臂移出顶视，筐右侧桌面上仍有散放带身，腕视显示筐和墙面附近张开的夹爪。



## 完成状态

227个目标parent均已有语言记录，未完成标注0；需用户决定事项0。保留的材料限制为1条源左腕不可读和明确列出的短片段，不以捏造文字填充。前期阶段性报告与断点历史保存在20261112报告，不代表本次最终完成数。

主JSONL SHA256：`afde8b79c5910f0af14f27380f3055d03e8c80c005074caf5903938bea7be95a`。

最终独立审计补记（BRT 2026-11-13T00:54:48−03:00）：227/227，missing/extra/duplicate/errors均0；227个只读HDF5头及13,534条证据路径、源帧范围、容器帧范围全部通过。均匀与detail采样索引合并后无未索引证据。主文件SHA256与quality_summary一致。
