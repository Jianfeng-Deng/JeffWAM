> 历史阶段记录。最终227/227完成报告见 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e83_A_piper_language_20261113.md`（2026-11-13T00:54:38.867998-03:00）；下文早期9条等数字均为当时快照。

# E83-A Piper Tier A 长文本语言标注（部分交付）

日期：2026-11-12T15:07:53.041355-03:00。执行者：codex_gpt-6.1-sol。**已完成9个独立train parent；全量未完成。**

输出为实际看图后的双语长文本，而非L1/L2/L3结构化结局标签。只允许 language_supervision_training，forbidden_uses=[test_ground_truth, threshold_source, model_checkpoint_selection]。未改变P3质量放行状态，未把最终结局加入起始画面或任务指令。

## 范围、划分与代表片段

先读 E82-D、E82-L/SPEC、E77 Piper相关部分和8份decisions_v2事件文本。revision3入口和其9项receipt在开始/结束均核验；结束另核验captures/video_routes的数据receipt。源→canonical→parent/component→locked_split逐条核对，全部train且allow_p1；dev/test连通分量抽帧/语言标注均为0。

P3候选共150个parent（demo4=129、demo6early=10、demo6native=11），每parent选候选示范中最长连续raw片段，不跨文件拼接。parent_candidate_segments保留链上候选、原范围和选择理由；片段首帧不冒称实际物理接管时刻。demo3未进入当前P3；通过源路径/duplicate_aliases与L1 same连接、去除已在P3的parent后，补充77个独立train/P1 parent，优先完整teleop再最长连续片段。诊断teleop7排除。same不等于示范质量合格。

|分层|已完成|候选parent|未完成|
|---|---:|---:|---:|
|demo4|4|129|125|
|demo6early|2|10|8|
|demo6native|2|11|9|
|demo3|1|77|76|

首批5条：demo4 teleop1/3、dagger5、demo6 raw3 early/raw16 native；逐句自检后才做batch001（demo4 teleop5、demo6 raw5 early/raw19 native），最后做demo3 teleop8。每条append+fsync JSONL并原子更新progress，未完成ID全量列在progress.json。

## 现有短指令与命名约定

现有短指令原文：**Pick up the zip tie and thread its pointed end through its own locking head to form a loop.**

demo4三条试标原始HDF5 attrs.prompt含该句；所查demo6 raw3/raw16 attrs.prompt为空，处理版 demo6_edges_nextframe_signedgrip_20261030/{early_proxy_success,native_rgb_success}/meta/tasks.jsonl 的 task 则为该句。demo4处理版meta/tasks.jsonl也相同。证据分列在existing_short_instructions.json，不能把处理版固定句冒充demo6原始prompt。

沿用decisions_v2对象名：zip_tie→zip tie（扎带）、lock_head→locking head（锁头）、tail→tail/strap（带身），将pointed tip（尖端）与带身区分。handover事件的arm是接收臂，from_arm是移交臂；8份参考文本均为right→left object_transfer，仅作为约定，未读取这些历史评估episode的图像。新episode分工由自身三视角帧核实。

## 试标与事实核对

5条试标分别包含6句起始描述、8–9句操作指令、3种长度改写、7–10段过程、8–10张关键帧；详见pilot_review.json。起始画面第二遍复读拼图pilot_start_second_pass.jpg；批量起始复读batch_001_start_second_pass.jpg；demo3首张拼图也复读。各条review_notes记逐句核对所据帧与保留不确定项。这是同一labeler自检，非独立人工一致性。

事实修订：接管片段不补写未录到的取带/交接；穿出带段之后不再可见时，只描述可见变化，不编造滑脱机制；靠近孔口不等同穿入；原来已在桌面/筐底的环不当成本次结果。early3首次允许源帧是28，不读取其P1外的raw0作起始监督。帧号误抄4418被存在性检查拦截，提交前改成真实4318。

场景逐句仅使用起始约1秒内的证据；任务指令按本段实际分工与顺序写成应执行的动作，未写成功/失败、偶然滑脱/重试。过程可以写重复对齐和连接处可见变化，但每段证据的最大帧不超过该段结束；keyframe只描写该帧及此前可知的对象身份。

## 质量统计与重复检查

9条包含73段过程、80个关键帧。直接审阅证据为224个源时刻、672张view帧；不同相机不当作独立episode。完整原视野JPG保留原生尺寸；拼图仅用于审阅，未从低分辨率缓存放大冒充细节。

去重率=规范化空白/大小写后的唯一完整文本数÷文本数；重复比例=1−去重率。英文长度为regex词数，中文长度为汉字数；过程/关键帧按单段/单帧统计，variants按3种长度分别统计。

|字段|条数|去重率|平均长度|最短–最长|
|---|---:|---:|---:|---:|
|scene_description_en|9|100.0%|151.8|137–165|
|task_instruction_en|9|100.0%|158.6|140–176|
|instruction_variants_short_en|9|100.0%|10.1|9–13|
|instruction_variants_medium_en|9|100.0%|43.0|36–45|
|instruction_variants_long_en|9|100.0%|71.9|64–80|
|process_narration_en|73|100.0%|42.8|33–57|
|keyframe_captions_en|80|100.0%|40.0|31–54|
|scene_description_zh|9|100.0%|209.7|188–233|
|task_instruction_zh|9|100.0%|202.3|172–227|
|instruction_variants_short_zh|9|100.0%|12.2|9–14|
|instruction_variants_medium_zh|9|100.0%|45.0|41–49|
|instruction_variants_long_zh|9|100.0%|85.6|76–102|
|process_narration_zh|73|100.0%|52.8|40–75|
|keyframe_captions_zh|80|100.0%|50.3|39–70|

英文场景最相似完整文本对的word-trigram Jaccard为0.156，对应E83_A_cf3d999d8bbb6e17a0e45e39 / E83_A_554224873da435ced4c978c5；该数仅作本批重复诊断，不作模型/评估阈值。

逐条抽查全部9条场景：瓶内单根/多根扇形、桌面已有环、开启的走线盖、黄筐底部预存交叠带身、起始已持双端的接管状态均在相应文本中体现。demo6墙面/黄筐/双臂分工相似是实景相似；没有给未看过的episode套写文本。短指令和同一任务步骤可相似。精确去重与近似度不代表视觉事实已被独立验真。

## 资源、命令与只读核查

CPU OpenCV/h5py/Pillow，CUDA_VISIBLE_DEVICES为空，OMP/OPENBLAS/MKL=1，cv2线程=1；未调用GPU/模型推理/训练。GPU=0 GPUh；未下载模型或外部数据，未复制源视频。仅把本任务自生成的少量审阅拼图临时传回会话用于实际看图，源视频/HDF5始终原位只读。thumbnails总量54,055,607 bytes（54.06 MB），低于5GB。

原位解码前后及最终复查36个源HDF5/视频的size/mtime/ctime/inode，变化0；不以可能受读取影响的atime判内容修改。见source_stat_*.json与quality_summary.json。未触碰同事进程、未写同事目录，未改E82 READY/labels/动作数组。

命令：`python3 notes/e83_A_bootstrap_20261112.py`；`setsid nohup env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python -u extract.py <manifest.json> > <extract.log> 2>&1 < /dev/null`；`python3 commit.py <authored_batch.json>`；`python3 finalize.py`。抽帧后台脚本只抽帧，不自动为未审帧生成文字。

## 交付、未完成与继续方式

目录 data/e82_piper_v1/lang_codex/A/：lang_A.jsonl、progress.json、samples.md、thumbnails/，另有candidate/demo3/batch manifests、pilot_review.json、quality_summary.json、provenance.json、源只读记录和本任务脚本。下文放3条完整双语文本原文，含场景、详细指令、3条改写、全部过程段和关键帧说明。

**未完成：142个P3候选parent、76个demo3补充parent，均未生成语言文本，不冒称全量完成。** 当前状态partial_reviewed_checkpoint，后台没有继续标注任务。续跑从progress.json和JSONL双重去重；按demo4/demo6early/demo6native交错取pending代表片段，demo3最后；有frames.json不等于已经看图，必须复读实际图片再写文字。

本批只做少量长示范，不能把长度/可见性/分工比例外推为全库分布；源帧之间的短时事件仍可能漏看。逐帧人工复核、独立事实验真与P3逐段动作质量审批均未完成。无需新增GPU、下载或训练授权；三个样例供组长审阅风格与术语，继续标注无需另改冻结划分。

## 三个完整样例原文

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



## 短片规范冲突补记

142个未完成P3 parent中有5个最长候选示范仅2–4个允许源观测（具体ID见short_clip_exceptions.json）。这些不可能产生6–10个不同关键帧，其中2–3帧者也不能硬切4个有意义阶段；未补帧、未重复帧、未编造动作。同parent有58–236帧的P1 inference成员，但不是已证实的完整遥操作示范，不能仅凭长度改选。后续需实际看图核实替代片段；若没有合适主片段，需用户决定允许短片降低关键帧/阶段数量，或继续保留未标。此问题仅影响该5个短parent，其余续标不需新增资源或划分授权。


续标授权补记（2026-11-12T15:45:10.844469-03:00）：用户允许无合适长片段时减少关键帧和阶段；先看同parent的train/P1较长备选，仍不合适则如实记录理由并仅标实际不同帧。batch003的3个候选已抽帧，尚未看图和写入语言标注，不计完成。

续标实际看图发现（2026-11-12T15:48:15.317872-03:00）：batch003 demo4 dagger1在0–736抽样三视图中主要是右手持带、容器附近调整和提起，未见左手交接或穿入。最长候选不必然是完整示范，须回查其7个train/P1链成员。记录representative_review.jsonl；尚未编写该条语言监督，完成数仍9。


### 持续标注检查点 2026-11-12T15:56:38.774908-03:00

{
  "timestamp_brt": "2026-11-12T15:56:38.774908-03:00",
  "completed": 12,
  "by_stratum": {
    "demo4": 5,
    "demo6early": 3,
    "demo6native": 3,
    "demo3": 1
  },
  "pending_p3": 139,
  "pending_demo3": 76,
  "process_stages": 89,
  "keyframes": 102,
  "fields": {
    "scene_description_en": {
      "unique_fraction": 1.0,
      "mean_characters": 885.75
    },
    "scene_description_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 225.75
    },
    "task_instruction_en": {
      "unique_fraction": 1.0,
      "mean_characters": 850.3333333333334
    },
    "task_instruction_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 213.75
    },
    "instruction_variants_en": {
      "unique_fraction": 1.0,
      "mean_characters": 671.0
    },
    "instruction_variants_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 168.5
    },
    "process_narration": {
      "unique_fraction": 1.0,
      "mean_characters": 4140.416666666667
    },
    "keyframe_captions": {
      "unique_fraction": 1.0,
      "mean_characters": 19089.333333333332
    }
  }
}

本批新增demo6 early episode6、native episode22及demo4 dagger3。逐条实际核看三视角；dagger3替代同parent较长但仅取带的dagger1，明确从操作途中开始。当前全部字段的逐条完整文本去重率见上；任务同源术语相似是预期现象，画面描述保留筐底/桌面散带、头部排列和初始抓持差异。统计为当前已完成范围，不代表全部完成。


### 持续标注检查点 2026-11-12T15:58:04.974353-03:00

{
  "timestamp_brt": "2026-11-12T15:58:04.974353-03:00",
  "completed": 13,
  "by_stratum": {
    "demo4": 6,
    "demo6early": 3,
    "demo6native": 3,
    "demo3": 1
  },
  "pending_p3": 138,
  "pending_demo3": 76,
  "process_stages": 93,
  "keyframes": 108,
  "fields": {
    "scene_description_en": {
      "unique_fraction": 1.0,
      "mean_characters": 890.2307692307693
    },
    "scene_description_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 227.07692307692307
    },
    "task_instruction_en": {
      "unique_fraction": 1.0,
      "mean_characters": 847.5384615384615
    },
    "task_instruction_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 212.76923076923077
    },
    "instruction_variants_en": {
      "unique_fraction": 1.0,
      "mean_characters": 665.6923076923077
    },
    "instruction_variants_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 167.46153846153845
    },
    "process_narration": {
      "unique_fraction": 1.0,
      "mean_characters": 4014.153846153846
    },
    "keyframe_captions": {
      "unique_fraction": 1.0,
      "mean_characters": 18674.846153846152
    }
  }
}

本批新增demo6 early episode6、native episode22及demo4 dagger3。逐条实际核看三视角；dagger3替代同parent较长但仅取带的dagger1，明确从操作途中开始。当前全部字段的逐条完整文本去重率见上；任务同源术语相似是预期现象，画面描述保留筐底/桌面散带、头部排列和初始抓持差异。统计为当前已完成范围，不代表全部完成。


### 持续标注检查点 2026-11-12T23:35:14.583946-03:00

{
  "timestamp_brt": "2026-11-12T23:35:14.583946-03:00",
  "completed": 16,
  "by_stratum": {
    "demo4": 6,
    "demo6early": 4,
    "demo6native": 4,
    "demo3": 2
  },
  "pending_p3": 136,
  "pending_demo3": 75,
  "process_stages": 106,
  "keyframes": 126,
  "fields": {
    "scene_description_en": {
      "unique_fraction": 1.0,
      "mean_characters": 876.625
    },
    "scene_description_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 223.6875
    },
    "task_instruction_en": {
      "unique_fraction": 1.0,
      "mean_characters": 837.1875
    },
    "task_instruction_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 211.125
    },
    "instruction_variants_en": {
      "unique_fraction": 1.0,
      "mean_characters": 652.875
    },
    "instruction_variants_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 164.6875
    },
    "process_narration": {
      "unique_fraction": 1.0,
      "mean_characters": 3822.5625
    },
    "keyframe_captions": {
      "unique_fraction": 1.0,
      "mean_characters": 17646.4375
    }
  }
}

继续按实际三视角画面逐条标注；短片段按可见动作选择阶段，不补写未观察流程。当前全部字段的逐条完整文本去重率见上；任务同源术语相似是预期现象，画面描述保留筐底/桌面散带、头部排列和初始抓持差异。统计为当前已完成范围，不代表全部完成。


### 持续标注检查点 2026-11-12T23:40:36.514912-03:00

{
  "timestamp_brt": "2026-11-12T23:40:36.514912-03:00",
  "completed": 20,
  "by_stratum": {
    "demo4": 7,
    "demo6early": 5,
    "demo6native": 5,
    "demo3": 3
  },
  "pending_p3": 133,
  "pending_demo3": 74,
  "process_stages": 125,
  "keyframes": 150,
  "fields": {
    "scene_description_en": {
      "unique_fraction": 1.0,
      "mean_characters": 879.6
    },
    "scene_description_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 225.5
    },
    "task_instruction_en": {
      "unique_fraction": 1.0,
      "mean_characters": 821.3
    },
    "task_instruction_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 206.8
    },
    "instruction_variants_en": {
      "unique_fraction": 1.0,
      "mean_characters": 642.85
    },
    "instruction_variants_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 162.75
    },
    "process_narration": {
      "unique_fraction": 1.0,
      "mean_characters": 3660.0
    },
    "keyframe_captions": {
      "unique_fraction": 1.0,
      "mean_characters": 16786.25
    }
  }
}

继续按实际三视角画面逐条标注；短片段按可见动作选择阶段，不补写未观察流程。当前全部字段的逐条完整文本去重率见上；任务同源术语相似是预期现象，画面描述保留筐底/桌面散带、头部排列和初始抓持差异。统计为当前已完成范围，不代表全部完成。


### 持续标注检查点 2026-11-13T00:06:55.990032-03:00

{
  "timestamp_brt": "2026-11-13T00:06:55.990032-03:00",
  "completed": 73,
  "by_stratum": {
    "demo4": 37,
    "demo6early": 10,
    "demo6native": 11,
    "demo3": 15
  },
  "pending_p3": 92,
  "pending_demo3": 62,
  "process_stages": 360,
  "keyframes": 466,
  "fields": {
    "scene_description_en": {
      "unique_fraction": 1.0,
      "mean_characters": 839.1232876712329
    },
    "scene_description_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 213.63013698630138
    },
    "task_instruction_en": {
      "unique_fraction": 0.7671232876712328,
      "mean_characters": 707.6986301369863
    },
    "task_instruction_zh": {
      "unique_fraction": 0.7671232876712328,
      "mean_characters": 176.9041095890411
    },
    "instruction_variants_en": {
      "unique_fraction": 0.7671232876712328,
      "mean_characters": 577.4109589041096
    },
    "instruction_variants_zh": {
      "unique_fraction": 0.7671232876712328,
      "mean_characters": 148.73972602739727
    },
    "process_narration": {
      "unique_fraction": 1.0,
      "mean_characters": 2931.5068493150684
    },
    "keyframe_captions": {
      "unique_fraction": 1.0,
      "mean_characters": 14265.534246575342
    }
  }
}

继续按实际三视角画面逐条标注；短片段按可见动作选择阶段，不补写未观察流程。当前全部字段的逐条完整文本去重率见上；任务同源术语相似是预期现象，画面描述保留筐底/桌面散带、头部排列和初始抓持差异。统计为当前已完成范围，不代表全部完成。


### 持续标注检查点 2026-11-13T00:15:08.985175-03:00

{
  "timestamp_brt": "2026-11-13T00:15:08.985175-03:00",
  "completed": 104,
  "by_stratum": {
    "demo4": 53,
    "demo6early": 10,
    "demo6native": 11,
    "demo3": 30
  },
  "pending_p3": 76,
  "pending_demo3": 47,
  "process_stages": 502,
  "keyframes": 651,
  "fields": {
    "scene_description_en": {
      "unique_fraction": 1.0,
      "mean_characters": 820.0192307692307
    },
    "scene_description_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 205.35576923076923
    },
    "task_instruction_en": {
      "unique_fraction": 0.7403846153846154,
      "mean_characters": 689.0865384615385
    },
    "task_instruction_zh": {
      "unique_fraction": 0.7307692307692307,
      "mean_characters": 171.57692307692307
    },
    "instruction_variants_en": {
      "unique_fraction": 0.7211538461538461,
      "mean_characters": 565.1730769230769
    },
    "instruction_variants_zh": {
      "unique_fraction": 0.7211538461538461,
      "mean_characters": 146.26923076923077
    },
    "process_narration": {
      "unique_fraction": 1.0,
      "mean_characters": 2856.076923076923
    },
    "keyframe_captions": {
      "unique_fraction": 1.0,
      "mean_characters": 13987.336538461539
    }
  }
}

继续按实际三视角画面逐条标注；短片段按可见动作选择阶段，不补写未观察流程。当前全部字段的逐条完整文本去重率见上；任务同源术语相似是预期现象，画面描述保留筐底/桌面散带、头部排列和初始抓持差异。统计为当前已完成范围，不代表全部完成。


### 持续标注检查点 2026-11-13T00:22:17.380847-03:00

{
  "timestamp_brt": "2026-11-13T00:22:17.380847-03:00",
  "completed": 129,
  "by_stratum": {
    "demo4": 65,
    "demo6early": 10,
    "demo6native": 11,
    "demo3": 43
  },
  "pending_p3": 64,
  "pending_demo3": 34,
  "process_stages": 616,
  "keyframes": 801,
  "fields": {
    "scene_description_en": {
      "unique_fraction": 1.0,
      "mean_characters": 810.8217054263566
    },
    "scene_description_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 201.26356589147287
    },
    "task_instruction_en": {
      "unique_fraction": 0.7054263565891473,
      "mean_characters": 681.3023255813954
    },
    "task_instruction_zh": {
      "unique_fraction": 0.6976744186046512,
      "mean_characters": 169.4496124031008
    },
    "instruction_variants_en": {
      "unique_fraction": 0.6821705426356589,
      "mean_characters": 559.3178294573644
    },
    "instruction_variants_zh": {
      "unique_fraction": 0.6821705426356589,
      "mean_characters": 145.02325581395348
    },
    "process_narration": {
      "unique_fraction": 1.0,
      "mean_characters": 2817.6356589147285
    },
    "keyframe_captions": {
      "unique_fraction": 1.0,
      "mean_characters": 13866.968992248061
    }
  }
}

继续按实际三视角画面逐条标注；短片段按可见动作选择阶段，不补写未观察流程。当前全部字段的逐条完整文本去重率见上；任务同源术语相似是预期现象，画面描述保留筐底/桌面散带、头部排列和初始抓持差异。统计为当前已完成范围，不代表全部完成。


### 持续标注检查点 2026-11-13T00:30:07.105567-03:00

{
  "timestamp_brt": "2026-11-13T00:30:07.105567-03:00",
  "completed": 152,
  "by_stratum": {
    "demo4": 77,
    "demo6early": 10,
    "demo6native": 11,
    "demo3": 54
  },
  "pending_p3": 52,
  "pending_demo3": 23,
  "process_stages": 710,
  "keyframes": 935,
  "fields": {
    "scene_description_en": {
      "unique_fraction": 1.0,
      "mean_characters": 802.6842105263158
    },
    "scene_description_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 198.2434210526316
    },
    "task_instruction_en": {
      "unique_fraction": 0.6973684210526315,
      "mean_characters": 672.1513157894736
    },
    "task_instruction_zh": {
      "unique_fraction": 0.6907894736842105,
      "mean_characters": 167.06578947368422
    },
    "instruction_variants_en": {
      "unique_fraction": 0.6710526315789473,
      "mean_characters": 555.1644736842105
    },
    "instruction_variants_zh": {
      "unique_fraction": 0.6710526315789473,
      "mean_characters": 144.1315789473684
    },
    "process_narration": {
      "unique_fraction": 1.0,
      "mean_characters": 2758.190789473684
    },
    "keyframe_captions": {
      "unique_fraction": 1.0,
      "mean_characters": 13728.888157894737
    }
  }
}

继续按实际三视角画面逐条标注；短片段按可见动作选择阶段，不补写未观察流程。当前全部字段的逐条完整文本去重率见上；任务同源术语相似是预期现象，画面描述保留筐底/桌面散带、头部排列和初始抓持差异。统计为当前已完成范围，不代表全部完成。


### 持续标注检查点 2026-11-13T00:44:31.385022-03:00

{
  "timestamp_brt": "2026-11-13T00:44:31.385022-03:00",
  "completed": 200,
  "by_stratum": {
    "demo4": 110,
    "demo6early": 10,
    "demo6native": 11,
    "demo3": 69
  },
  "pending_p3": 19,
  "pending_demo3": 8,
  "process_stages": 910,
  "keyframes": 1189,
  "fields": {
    "scene_description_en": {
      "unique_fraction": 1.0,
      "mean_characters": 793.155
    },
    "scene_description_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 193.855
    },
    "task_instruction_en": {
      "unique_fraction": 0.69,
      "mean_characters": 649.515
    },
    "task_instruction_zh": {
      "unique_fraction": 0.685,
      "mean_characters": 160.99
    },
    "instruction_variants_en": {
      "unique_fraction": 0.67,
      "mean_characters": 545.875
    },
    "instruction_variants_zh": {
      "unique_fraction": 0.67,
      "mean_characters": 141.73
    },
    "process_narration": {
      "unique_fraction": 1.0,
      "mean_characters": 2644.66
    },
    "keyframe_captions": {
      "unique_fraction": 1.0,
      "mean_characters": 13257.98
    }
  }
}

继续按实际三视角画面逐条标注；短片段按可见动作选择阶段，不补写未观察流程。当前全部字段的逐条完整文本去重率见上；任务同源术语相似是预期现象，画面描述保留筐底/桌面散带、头部排列和初始抓持差异。统计为当前已完成范围，不代表全部完成。


### 持续标注检查点 2026-11-13T00:50:29.899163-03:00

{
  "timestamp_brt": "2026-11-13T00:50:29.899163-03:00",
  "completed": 221,
  "by_stratum": {
    "demo4": 126,
    "demo6early": 10,
    "demo6native": 11,
    "demo3": 74
  },
  "pending_p3": 3,
  "pending_demo3": 3,
  "process_stages": 997,
  "keyframes": 1300,
  "fields": {
    "scene_description_en": {
      "unique_fraction": 1.0,
      "mean_characters": 788.579185520362
    },
    "scene_description_zh": {
      "unique_fraction": 1.0,
      "mean_characters": 192.20361990950227
    },
    "task_instruction_en": {
      "unique_fraction": 0.6923076923076923,
      "mean_characters": 645.1266968325792
    },
    "task_instruction_zh": {
      "unique_fraction": 0.6877828054298643,
      "mean_characters": 159.44796380090497
    },
    "instruction_variants_en": {
      "unique_fraction": 0.669683257918552,
      "mean_characters": 544.6153846153846
    },
    "instruction_variants_zh": {
      "unique_fraction": 0.669683257918552,
      "mean_characters": 141.21719457013575
    },
    "process_narration": {
      "unique_fraction": 1.0,
      "mean_characters": 2606.868778280543
    },
    "keyframe_captions": {
      "unique_fraction": 1.0,
      "mean_characters": 13117.10407239819
    }
  }
}

继续按实际三视角画面逐条标注；短片段按可见动作选择阶段，不补写未观察流程。当前全部字段的逐条完整文本去重率见上；任务同源术语相似是预期现象，画面描述保留筐底/桌面散带、头部排列和初始抓持差异。统计为当前已完成范围，不代表全部完成。
