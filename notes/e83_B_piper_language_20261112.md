# E83-B Piper 双语语言训练标注


日期：2026-11-12 BRT；完成：2026-11-12T15:12:00.475222-03:00；labeler=`codex_gpt-6.1-sol`。状态：**94条完成，1条诊断隔离；证据限制明确保留。**


## 范围与交付

按 READY_data.json 与 READY_parent_split.json revision3 的稳定入口推导：P1 train 连通 parent 245，P3 候选 parent 150，Tier B 差集95，A/B候选组件交集0。roster.json先于解码保存；每个parent选择三路相机声明完整优先、再最长的P1原始clip，平局按canonical_id确定，不按L2结局筛选。全部94条均为train，dev/test语言标注及抽帧为0；没有覆盖E82文件。

唯一隔离：E83B_061，demo3 teleop episode_7，原始短指令明确为“采集链路诊断片段，不用于训练”，L1亦为unclear。roster保留，exclusions.json记录原因，不生成扎带任务语言。因此94/94有效目标完成，95个roster全部有处理结论。

主文件为 `data/e82_piper_v1/lang_codex/B/lang_B.jsonl`；每parent一条。另有progress.json、roster.json、samples.md、thumbnails/、FINAL_QA.json、provenance.json、pilot_review.json、existing_language_and_conventions.json及脚本/版本审计文件。**训练读取只指定lang_B.jsonl，不通配导入pre_final_qa、批次输入或审计副本。**

## 已有短指令、任务与事件约定

现有短指令原文：

> Pick up the zip tie and thread its pointed end through its own locking head to form a loop.


已直接核查processed_sources.jsonl中train成员：demo4共146条、demo6共21条的tasks字段均为此句；还核查了现有processed meta/tasks.jsonl词表，路径及原文保存在existing_language_and_conventions.json。允许原始来源元数据538条prompt同句，21条demo6原始prompt为空，1条诊断指令；不可把原始prompt为空误说成processed没有任务文本。

已先读E82-D、E82-L、labels_codex/SPEC.md任务定义及E77的Piper来源、对象、双臂及事件章节。人工data/piper_events_20261106/decisions_v2八文件仅作术语/流程约定依据，未迁移其历史留出episode标签或读取其评估视频。旧object键zip_tie/lock_head/tail分别对应zip tie/locking head/tip或free strap；tail不自动等同精确尖端像素。标准分工为右手取整带、送至操作区、向左手交接锁头，右手取自由带身、对齐、穿入与拉出；真实clip已从中途开始时按当前持握关系写。人工stride2事件frame未被当成本轮raw source_frame_index。

## 视觉证据、因果性与首5条试标

CPU OpenCV原位只读解码，h5py以r模式读取；OMP/MKL/OPENBLAS限制1线程，CUDA_VISIBLE_DEVICES为空。不加载策略、VLM或权重，不训练，GPU 0 GPUh，不下载模型或原数据，不复制视频。服务器端批量抽帧用setsid nohup独立后台脚本，缩略图传至本机会话用于实际目视。

每clip采用raw 0基源帧0、约1秒帧30、12%/25%/40%/55%/70%/85%及末帧，共9个时刻；三个相机分别保留原尺寸JPEG，再拼成带ID/帧号/视角的审阅图。scene_description只用frame0，逐句对照首帧，不引入后续事实。task_instruction是8句祈使说明；结局、偶然滑脱及重试只在retrospective_summary叙述。summary只概括选定clip，不借用parent链后来成功或L2标签。

先完成E83B_001–005试标，再作同labeler第二遍视觉复读，保存pilot_review.json后提交其余批次；每批立刻追加JSONL，逐条fsync并原子更新progress.json。首5条发现2号只录到接近及容器倾倒、3号左腕不可用、4/5号结束于自由带身抓取，因此删去无法证实的完成断言并明确遮挡。该自检不等于独立人工一致性。

E83B_003的左腕路线元数据虽有路径，但实际解码没有有效左腕图；最终views由实际evidence_frames导出，仅top/right，missing_views显式保留cam_left_wrist。缺视角占位图不当作证据，不把黑图送作画面事实。

额外按原图尺寸复查E83B_011、018、041首帧三视图：分别对应容器居中、带身已横过锁头、尖端仍在锁头旁的不同起始状态；复核图为各自qa_start_native.jpg。最终校正了晚阶段clip的中/长改写为继续夹取和拉出已露出带身，避免重复穿入。

## 最终数量与质量统计

94条置信度：high=9，medium=81，low=4。94个不同parent/component；所有schema、帧范围、证据路径、双语字段、用途限制与句数检查通过。场景4–8句、指令6–12句（实际8句）、回顾2–3句，三条改写分别1/3/6句；最简英文14词，全部≤15词。

去重保留率定义为unique_texts / records；重复比例为1减去该值。英文长度按空白词数，中文长度按汉字数（不含标点）。下表显示全字段精确文本统计，不能把高去重率解释为语义独立性。

|字段|独特文本/条数|去重保留率|平均长度|范围|单位|
|---|---:|---:|---:|---:|---|

|scene_description_en|94/94|100.00%|78.8|68–110|词|

|scene_description_zh|94/94|100.00%|96.6|83–147|汉字|

|task_instruction_en|8/94|8.51%|147.4|109–155|词|

|task_instruction_zh|8/94|8.51%|178.1|130–186|汉字|

|retrospective_summary_en|94/94|100.00%|43.4|23–63|词|

|retrospective_summary_zh|94/94|100.00%|51.1|28–76|汉字|

|instruction_variants_short_en|1/94|1.06%|14.0|14–14|词|

|instruction_variants_short_zh|1/94|1.06%|15.0|15–15|汉字|

|instruction_variants_medium_en|3/94|3.19%|49.7|38–53|词|

|instruction_variants_medium_zh|3/94|3.19%|51.6|45–54|汉字|

|instruction_variants_long_en|3/94|3.19%|72.4|55–76|词|

|instruction_variants_long_zh|3/94|3.19%|84.4|74–87|汉字|


画面段落逐条据图写出，英文及中文精确重复均为0。指令只有8个完整英文文本，确有较高重复：这是同一任务、相同分工步骤的共享规范语言，不能宣称每条指令独一无二。短改写刻意保留同一任务句；中/长改写分当前持握状态，完成步骤的措辞相似。未通过加入ID、随机同义词或无证细节伪造多样性。

另计算英文场景的word five-gram Jaccard并抽看最高相似对，检查重复是否掩盖视觉差异。最高几对如下（该数值仅文字审计，不作评估阈值）：

|ID对|5-gram Jaccard|实际差异核查|
|---|---:|---|

|E83B_030 / E83B_095|0.1484|30/95：30号有前部多个散环，95号左腕散环与背景坐人不同；二者均为独立看图段落。|

|E83B_080 / E83B_087|0.1250|80/87：80号容器为多带束、后盖板抬起；87号单带、前部带长斜延伸段的环。|

|E83B_069 / E83B_092|0.1206|69/92：69号背景为单根备用带及前缘散带；92号备用带分叉、前部环位置不同。|

## 限制、未完成项与消费建议

以下13条没有完整录到双臂分工与后续操作，不能宣称episode全流程已实证：E83B_002, E83B_010, E83B_015, E83B_017, E83B_020, E83B_030, E83B_042, E83B_045, E83B_058, E83B_066, E83B_067, E83B_078, E83B_083。部分只到抓取/交接、部分无动作、部分交接未保持。visual_review的full_episode_hand_sequence_verified=false和文字不确定性保留；其未录到的完成步骤依据任务定义及人工约定作祈使性补全，不是事实描述。若要求指令每个步骤均由该clip实证，应先筛除这些记录的指令目标或人工补核；画面描述和事后回顾仍有各自实际证据。

其余记录的已见分工也不代表所有对齐、贯穿、拉紧或放下细节均清晰。每条unobserved_completion_steps_are_prescriptive=true，不能拿“应该做”的指令当动作事件真值。没有把接近孔口当已穿入；回顾中只见横过锁头但孔内接触被挡时按几何现象描述。长录制003/035/044/063等为稀疏审阅，不能排除帧间短暂滑脱、人的介入或另一次尝试，也未生成逐阶段事件时间真值。

本任务为Tier B，不生成Tier A的process_narration或keyframe_captions，不重复Tier A组件。未进行独立人工验真、逐帧审核或训练效果评估；不批准P3质量门槛。诊断61号无语言输出是有意隔离，不是遗漏。

需组长判断的仅是上述13条“规范补全指令”是否进入所需训练子集；所有证据与标志已保存，不需为本轮交付再授权训练。

所有记录固定intended_uses=[language_supervision_training]及forbidden_uses=[test_ground_truth, threshold_source, model_checkpoint_selection]。模型生成的段落只作为训练目标，不作测试真值、阈值或checkpoint选型依据。

## 资源、命令与只读验证

有效94条证据图像（frame×view）2529张；包括隔离61号在内缩略图总计128,556,209 bytes（128.56 MB），远低于5GB。对94条使用的376个源路径比较size/mtime_ns/ctime_ns/inode，变化0；读取可能改变atime，未把atime作为内容修改。revision3引用9文件SHA256及READY_data哈希在最终再次一致。没有修改同事目录或接触其进程。

代表命令（均在服务器自有B目录执行）：

```sh
python3 /tmp/e83b_prepare.py
setsid nohup /mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python extract_cpu.py 0 5 > pilot_extraction.log 2>&1 < /dev/null &
setsid nohup /mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python extract_cpu.py 5 95 > batch_extraction.log 2>&1 < /dev/null &
python3 commit.py pilot.json
python3 commit.py batch02.json  # 后续逐批至batch16.json
python3 finalize.py
python3 report.py
```


复跑标注提交会按completed ID跳过；抽帧按已有index.json跳过；不应重做已完成ID。pre_final_qa保留首次校验前版本，qa_revision_log保存最后改写原因。

## 三个样例原文

以下原样摘取主JSONL的全部语言字段，中英文均保留；证据索引与完整元数据见主文件。

### E83B_011

parent/component=`pc_1a6628eef51912393d0f`；split=train；tier=B；confidence=medium；source_frame_index=0。

source_path：`/mnt/workspace/colleague-a/Datasets/piper/piper_raw_incremental/raw/demo3_teleop_realsense_v2_hdf5/aloha_mobile_dummy/episode_28.hdf5`

证据：[首帧三视图](/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/lang_codex/B/thumbnails/E83B_011/qa_start_native.jpg)；[完整抽帧时间线](/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/lang_codex/B/thumbnails/E83B_011/sheet.jpg)。

**scene_description_en**

A single orange tie rises from a clear container positioned directly over the central cable-cover area in the top view. Its locking head leans to the right above the rim, while the lower strap descends behind the clear wall and the exact tip is difficult to pick out. The left wrist camera sees the cable covers and separated empty fingers against an office background. The right wrist view is offset from the container, which is only partly visible near the left edge, and its gripper fingers are also open.

**scene_description_zh**

顶视中，一根橙色扎带从直接放在中央走线盖区域的透明容器伸出。锁头高于容器口并向右倾，下部带身落在透明壁后，准确尖端难以辨出。左腕画面可见走线盖及分开的空夹指，后方是办公环境。右腕视野与容器错开，容器只在左缘部分可见，其夹指也张开着。

**task_instruction_en**

Thread the tip of the zip tie through its own locking head to make a closed loop. Use the right gripper to take one zip tie from the clear container and lift it into the shared workspace. Present the locking-head end to the left gripper, close the left fingers around that end, and release the right grip only after the left provides support. Move the right gripper along the hanging strap and take the free end with the tip projecting beyond the fingers. Curve the free strap back toward the locking head without twisting the entry section. Adjust the right wrist and the left holding pose until the tip points into the locking-head opening. Advance the tip through the opening while the left gripper stabilizes the head, then draw the emerging strap through to establish the loop. Keep the loop supported between the grippers and lower it onto the clear tabletop before opening the fingers.

**task_instruction_zh**

让扎带尖端穿过自身锁头，形成闭环。 用右夹爪从透明容器取出一根扎带，并提到双臂共同操作区。 将锁头端送给左夹爪，左夹指闭合固定后再松开右夹爪。 让右夹爪沿垂下的带身移动并夹取自由端，使尖端伸出夹指。 将自由带身弯回锁头方向，避免扭转准备穿入的部分。 调整右腕与左手持握姿态，使尖端对准锁头入口。 在左夹爪固定锁头的同时推进尖端穿过入口，再将露出的带身拉过锁头以建立闭环。 用夹爪托住闭环，将其降低至空桌面后再张开夹指。

**instruction_variants 1 (short) en**

Thread the zip tie’s tip through its own locking head to form a loop.

**instruction_variants 1 zh**

将扎带尖端穿过自身锁头形成闭环。

**instruction_variants 2 (medium) en**

Use the right gripper to bring the tie to the left gripper and transfer support of the locking head to the left. Take the free strap with the right gripper, align its tip with the head, and feed it through. Pull out the emerging section and support the loop before placing it down.

**instruction_variants 2 zh**

用右夹爪将扎带送向左夹爪，将锁头交给左手固定。右夹爪取自由带身，将尖端对准锁头并穿入。拉出露出的部分，托住闭环后再放下。

**instruction_variants 3 (long) en**

Bring the tie into the shared workspace with the right gripper. Let the left gripper secure the locking head before releasing the right grip. Use the right gripper to take the free strap behind the tip. Curve the strap toward the head and align the tip with the opening. Feed the tip through and draw the emerging section out to form a loop. Keep the loop supported while lowering it onto the table and releasing it.

**instruction_variants 3 zh**

用右夹爪将扎带带到共同操作区。左夹爪固定锁头后再松开右夹爪。右夹爪夹住尖端后方的自由带身。将带身弯向锁头，使尖端对准入口。将尖端穿入并拉出露出的部分形成闭环。托住闭环降低至桌面，再松开夹爪。

**retrospective_summary_en**

The right gripper takes the tie and transfers the locking-head end to the left gripper before gathering the free strap. The hands bend and bring the strap across the head, then lower and release the loop-shaped tie near the front of the table.

**retrospective_summary_zh**

右夹爪取带，将锁头端交给左夹爪后再抓取自由带身。双手将带身弯回并送过锁头区域，随后降低并把呈环状的扎带释放到桌前部。

### E83B_018

parent/component=`pc_353ab4e03367f6c543ad`；split=train；tier=B；confidence=medium；source_frame_index=0。

source_path：`/mnt/workspace/colleague-a/Datasets/piper/piper_raw_incremental/raw/demo3_inference_realsense_v2_hdf5/aloha_mobile_dummy/episode_115.hdf5`

证据：[首帧三视图](/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/lang_codex/B/thumbnails/E83B_018/qa_start_native.jpg)；[完整抽帧时间线](/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/lang_codex/B/thumbnails/E83B_018/sheet.jpg)。

**scene_description_en**

The grippers face each other over the cable covers, with the left holding an orange locking head. A loop hangs below them in the top view, while the left wrist camera shows a short tip projecting left across the head and a curved strap extending right. The right wrist view shows the rounded loop around the opposing gripper, which obscures the exact contact. A clear container and loose loop lie on the white table behind the hands, with a desk divider farther back.

**scene_description_zh**

两夹爪在走线盖上方面对面，左手夹着橙色锁头。顶视可见下垂的环，左腕近景中短尖端向左露出锁头，弯曲带身伸向右侧。右腕画面可见环绕对侧夹爪的圆弧，准确接触点被挡。双手后方白桌上有透明容器和散放的环，远处有隔板。

**task_instruction_en**

Thread the tip of the zip tie through its own locking head to make a closed loop. Keep the left gripper on the locking head and the right gripper beside the strap already crossing the head. Check the exposed strap on the far side of the locking head using the wrist views. Grip that exposed section with the right gripper without displacing the left-held head. Draw the strap farther through the head while supporting the hanging loop. Keep the loop clear of the gripper fingers as the arms separate. Lower the supported tie onto an empty area of the tabletop. Open the grippers after the table supports the tie.

**task_instruction_zh**

让扎带尖端穿过自身锁头，形成闭环。 让左夹爪稳住锁头，右夹爪保持在已经横过锁头的带身旁。 通过腕部视角检查锁头另一侧露出的带身。 用右夹爪夹住露出部分，避免推动左手固定的锁头。 托住下垂的环，将带身继续拉过锁头。 双臂分开时让环避开夹指。 托住扎带，将其降低至桌面的空处。 桌面承托扎带后再张开夹爪。

**instruction_variants 1 (short) en**

Thread the zip tie’s tip through its own locking head to form a loop.

**instruction_variants 1 zh**

将扎带尖端穿过自身锁头形成闭环。

**instruction_variants 2 (medium) en**

Keep the left gripper on the locking head and draw the exposed strap through with the right gripper. Maintain support of the loop while lowering the tie to the table. Release it only after the tabletop supports it.

**instruction_variants 2 zh**

左夹爪稳住锁头，右夹爪将露出的带身继续拉过锁头。降低扎带到桌面时保持对环的承托。桌面托住后再释放。

**instruction_variants 3 (long) en**

Keep the locking head steady in the left gripper. Locate the strap already projecting beyond the head. Close the right gripper on the exposed section without pushing the head. Draw that section through while supporting the loop. Lower the tie to a clear area of the table. Open the grippers once the table supports it.

**instruction_variants 3 zh**

左夹爪保持锁头稳定。找到已经露出锁头的带身。在不推动锁头的情况下用右夹爪夹住露出部分。托住环并把该段继续拉过锁头。将扎带降低至桌面的空处。桌面托住后再张开夹爪。

**retrospective_summary_en**

The right hand opens, moves above the head, and closes near the projecting strap while the left continues holding the head. It makes several small pulling or positioning movements, and the clip ends with the loop still held between the arms.

**retrospective_summary_zh**

右手张开并移到锁头上方，在露出的带身附近闭合，左手继续固定锁头。右手做了数次小幅拉动或定位，片尾环仍由双臂持握。

### E83B_041

parent/component=`pc_68813839aa10ec8cdbc4`；split=train；tier=B；confidence=medium；source_frame_index=0。

source_path：`/mnt/workspace/colleague-a/Datasets/piper/piper_raw_incremental/raw/demo3_dagger_realsense_v2_hdf5/aloha_mobile_dummy/episode_52.hdf5`

证据：[首帧三视图](/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/lang_codex/B/thumbnails/E83B_041/qa_start_native.jpg)；[完整抽帧时间线](/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/lang_codex/B/thumbnails/E83B_041/sheet.jpg)。

**scene_description_en**

The left gripper holds an orange locking head while the right holds a returned strap with its free end pointing upward beside the head. In the top view the arms nearly meet above loose loops and a clear container farther back. The left wrist camera shows the tip diagonally above the head, with a separate standing tie and desk divider behind. The right wrist camera shows a rounded strap arc against the other gripper, obscuring the exact finger contact and hole.

**scene_description_zh**

左夹爪持橙色锁头，右手夹着折回带身，自由端在锁头旁向上指。顶视中双臂在散环上方几乎相接，后方有透明容器。左腕画面里尖端斜伸在锁头上方，背景另有竖立扎带及隔板。右腕相机看到对侧夹爪前的圆弧带身，准确夹持点及锁孔被挡。

**task_instruction_en**

Thread the tip of the zip tie through its own locking head to make a closed loop. Maintain the left gripper on the locking head and the right gripper on the free strap. Hold the locking head steady with the left arm while leaving its opening accessible. Position the right grip behind the tip so that the exposed end can enter the opening. Curve the free strap back toward the locking head without twisting the entry section. Adjust the right wrist and the left holding pose until the tip points into the locking-head opening. Advance the tip through the opening while the left gripper stabilizes the head, then draw the emerging strap through to establish the loop. Keep the loop supported between the grippers and lower it onto the clear tabletop before opening the fingers.

**task_instruction_zh**

让扎带尖端穿过自身锁头，形成闭环。 保持左夹爪持锁头、右夹爪持自由带身的分工。 用左臂稳住锁头，并让锁孔保持可接近。 将右夹爪的夹持点放在尖端后方，留出能够进入锁孔的端部。 将自由带身弯回锁头方向，避免扭转准备穿入的部分。 调整右腕与左手持握姿态，使尖端对准锁头入口。 在左夹爪固定锁头的同时推进尖端穿过入口，再将露出的带身拉过锁头以建立闭环。 用夹爪托住闭环，将其降低至空桌面后再张开夹指。

**instruction_variants 1 (short) en**

Thread the zip tie’s tip through its own locking head to form a loop.

**instruction_variants 1 zh**

将扎带尖端穿过自身锁头形成闭环。

**instruction_variants 2 (medium) en**

Keep the left gripper on the locking head and guide the free strap with the right gripper. Align the tip with the opening, pass it through, and pull the emerging section to form a loop. Support the loop before setting it down.

**instruction_variants 2 zh**

左夹爪稳住锁头，右夹爪引导自由带身。将尖端对准入口并穿过，拉出露出的部分形成闭环。托住闭环后再放下。

**instruction_variants 3 (long) en**

Stabilize the locking head with the left gripper. Hold the free strap near its tip with the right gripper. Bend the strap back toward the head and line up the exposed tip with the opening. Feed the tip through while keeping the head steady. Pull the emerging section through to form a closed loop. Support the tie as you lower and release it onto the tabletop.

**instruction_variants 3 zh**

左夹爪固定锁头。右夹爪夹住靠近尖端的自由带身。将带身弯回锁头，并使露出的尖端对准入口。在稳住锁头的同时送入尖端。将露出部分拉过锁头以形成闭环。托住扎带降低至桌面，再松开夹爪。

**retrospective_summary_en**

The right hand moves down the strap and returns its end toward the head repeatedly. Later views show the strap crossing the head and a hanging loop, with the right hand moving away while the left still supports the tie at the end.

**retrospective_summary_zh**

右手沿带身向下退开，又多次把端部送回锁头。后段可见带身横过锁头及下垂的环，片尾右手移开，左手仍承托扎带。
