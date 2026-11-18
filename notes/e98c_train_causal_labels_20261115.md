# E98-C 训练因果首遍盲标

创建 BRT：2026-11-15T12:31:37.174450-03:00
状态：初始化，未完成标注。GPUh=0。

## 协议与隔离
仅自动导出训练查询键；不读取回顾式答案和逐 parent 说明。每 parent 按 source_frame 递增，先阶段后21谓词，落盘锁定后才曝光未来；P073固定unknown。dev答案不读。永久排除 piper_4dc107335389f5711a0f65e0。

## 覆盖统计
待导出。

## 对照与独立复标
全部锁定后才允许对照。待完成；非人工，同模型家族。

## 资源与限制
纯CPU；新增落盘≤10GB；源和旧标签只读；不写E103-D目录。

## 查询导出与分片
60 parent、4410时刻、92610条计划谓词；A/B/C字典序各20parent。导出SHA256：79ef6465a3fa36eff7811f84429dc6c4ec0831bde4f6f47fe1ba9c0daa8f71d1。查询SHA256：705ea296e34c959130599fe26ab59c650d0211818bbcd010f2a3722e3fa060d0。初版导出因source_frame空值中止，未输出答案；修正为首个非空source/frame键后成功。源帧不平移。门控SHA256：f8fd045e1f5c5c5f01f6641243897b8c93735d85d2adf78128578abe649f60fe。每点三相机t-60,t-30,t；exposure及lock事件哈希链，锁定文件独占创建，fsync后方可下一点。图像导出不等于实际观看，visually_reviewed为标注者声明。

## 审计与发布门控验证
已验证不完整时compare_after_lock.py返回REFUSED_PRIMARY_INCOMPLETE，retrospective_files_opened=false；--publish同样拒绝且未创建根标签文件。10%复标查询预先固定seed=98，共441时刻，只保存查询键，未开始复标。比较脚本尚未打开回顾答案。其时间统计预置仅为下一次观测到的因果yes间隔，不冒充同目标事件提前量；事件身份匹配仍须全量锁定后处理。

### 因果锁定检查点 2026-11-15T12:44:54.362392-03:00
有效时刻 29/4410，完整parent 0/60，谓词 609，审计错误 0，新增目录字节 10509717；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T12:46:54.816832-03:00
有效时刻 39/4410，完整parent 1/60，谓词 819，审计错误 0，新增目录字节 14095527；GPUh=0。尚未进行回顾对照或独立复标。

## 门控v2（不修改既有锁）
gate_v2.py保留同一逐点锁定顺序与哈希链。若同一parent前一锁距离≤30帧，仅显示当前三机，历史来自此前已经看过且锁定的帧；其余点仍显示t-60,t-30,t。每次记录history_policy与门控脚本SHA。此优化不开放未来帧。AUDIT_HOLD存在则拒绝新曝光/锁定。原gate.py与原审计脚本保留哈希版本；审计临时文件加入pid避免并发碰撞。

### 因果锁定检查点 2026-11-15T12:47:55.066174-03:00
有效时刻 48/4410，完整parent 2/60，谓词 1008，审计错误 0，新增目录字节 16548104；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T12:48:55.358004-03:00
有效时刻 57/4410，完整parent 3/60，谓词 1197，审计错误 0，新增目录字节 18721778；GPUh=0。尚未进行回顾对照或独立复标。

## 源身份核验完成 2026-11-15T12:51:24.185954-03:00
60不同train parent、180视频全部原位SHA256与帧数核验通过，查询帧均在合法范围，三机帧数相等。源读取总量55,292,895,938字节（非新增落盘）；源视频复制0，GPUh=0。source_identity.json SHA256=4034526ca9e2e60df09cb5c766f099dcf0701352c292b887caeb842dd6fbe795。未显示未来图像；hash读取压缩字节不等于图像曝光。HDF原始行/视频索引契约沿用E82，不声称本轮重复逐帧核验同步。

## 一致率实现核验
四值一致率/κ、yes/no双方二值子集过滤、回顾yes但因果no/unknown分母、退化/空集合返回null已用合成符号对核验通过。未读取真实旧答案；真实一致率仍待全部首遍锁定。验证记录statistics_unit_validation.json。

## 显示几何核验与修正
180路中172路为640×480、8路为1280×720。原gate/v2将所有帧缩放至640×480，会使后者纵横比变形但未裁剪或平移源帧；gate_v3已改为保持纵横比、居中黑边。此前已锁定的受影响时刻计数：{"piper_ac3321cebed909757be43c3e": 42, "piper_5293db5f66779e270bae10a6": 61}；保留原答案不重写。受影响canonical与版本SHA记录aspect_audit_initial_locked.json。该显示局限须在最终报告单列，若之后发现答案错误另写later_note。

## 执行偏差记录
初始化读取完整阶段/谓词JSON时，首个原始命令输出1491行并被截断，违反单次≤200行输出约束；未包含训练禁区答案。后续读取均精简字段并只输出摘要。
几何修正前已显示的原图未被替换，旧锁不重写；v3后续保持纵横比。最终须按exposure代码SHA重计受影响数量，不能把修正前检查点103时刻当最终数字。

## 延迟分析预置（未读取回顾答案）
event_offsets_after_lock.py已创建并验证在主标不全时拒绝读取旧答案。完成后将只在明确cycle/attempt上下文内对齐首次因果yes采样点，记录回顾yes且因果no/unknown的有符号帧偏移；缺上下文或无后续正证据保留缺测/截断。它是可观察证据的操作性时间参考，不冒充物理事件真值；同上下文重复动态事件仍需说明。所有真实数值仍未计算。

### 因果锁定检查点 2026-11-15T13:38:17.740725-03:00
有效时刻 570/4410，完整parent 4/60，谓词 11970，审计错误 0，新增目录字节 114594315；GPUh=0。尚未进行回顾对照或独立复标。

## 未曝光parent工作量均衡 2026-11-15T13:39:46.170904-03:00
仅迁移从未有exposure的完整parent；保留每分片现有曝光keys前缀，未修改任何锁/图像/答案，60-parent查询清单不变。详情及前后assignment SHA见assignment_rebalance_v1.json。新统计：[{"shard": "a", "old_parents": 20, "new_parents": 25, "old_moments": 1156, "new_moments": 1470, "before_sha256": "0c4317407d0d37cb826d3693551edb84e63769e6192b18c43c490fb990fbeb0d", "after_sha256": "8d34ee62ba2474fab4367d91399c625e587f1a0dd6323dafb8a3544d843152a1", "existing_exposure_prefix_unchanged": true}, {"shard": "b", "old_parents": 20, "new_parents": 18, "old_moments": 1698, "new_moments": 1470, "before_sha256": "0e4da31acdb1945b0408a3dbc9d20f9d24225aff02331fb58c5082dbf457a7ae", "after_sha256": "9ca9ea20db31bbaddb541051145dfbf022c99904466dd14c1e5dd14ee27039bd", "existing_exposure_prefix_unchanged": true}, {"shard": "c", "old_parents": 20, "new_parents": 17, "old_moments": 1556, "new_moments": 1470, "before_sha256": "bf258aeb23b6d0802e1e71fba57a4f82d421b8f77c852dd1392a66c429677e30", "after_sha256": "1151ecccc426ec607277d7beccee96ca99b1e9f21c5c4acadb0dd93ee747e1aa", "existing_exposure_prefix_unchanged": true}]。

### 因果锁定检查点 2026-11-15T13:41:59.202905-03:00
有效时刻 619/4410，完整parent 6/60，谓词 12999，审计错误 0，新增目录字节 123438151；GPUh=0。尚未进行回顾对照或独立复标。

## 永久排除别名核验
只读E82永久排除元数据，按canonical与parent_component双重核对，60训练parent与排除集合交集均为0。未打开被排除视频或缓存。证明见permanent_exclusion_audit.json；未只依赖canonical名称字面匹配。

## 发布溯源字段
最终发布器增加原lock路径/SHA、exposure路径、当前曝光帧区间与causal_horizon_basis=gate_recorded_display_cutoff_and_annotator_sequential_first_pass_attestation；不改变原答案、锁时、source_frame或标注者。明确这不是外部访问控制/注意力证明。修改以原子替换完成，原审计器已保留版本。再次验证主标不全时--publish拒绝且根标签未创建。

### 因果锁定检查点 2026-11-15T13:49:45.342754-03:00
有效时刻 734/4410，完整parent 7/60，谓词 15414，审计错误 0，新增目录字节 148566455；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T13:54:01.178123-03:00
有效时刻 798/4410，完整parent 8/60，谓词 16758，审计错误 0，新增目录字节 161338869；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T13:57:13.597834-03:00
有效时刻 840/4410，完整parent 10/60，谓词 17640，审计错误 0，新增目录字节 168554240；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T14:09:27.819534-03:00
有效时刻 995/4410，完整parent 11/60，谓词 20895，审计错误 0，新增目录字节 198154089；GPUh=0。尚未进行回顾对照或独立复标。

## 主标四分之一检查点 2026-11-15T14:21:34.455961-03:00
有效锁定1131/4410时刻，23751谓词，完整parent 11/60，审计错误0，当前新增目录224739847字节。仍IN_PROGRESS；未读回顾式答案，未开始441时刻独立复标，无训练，GPUh=0。

### 因果锁定检查点 2026-11-15T14:42:34.481405-03:00
有效时刻 1400/4410，完整parent 12/60，谓词 29400，审计错误 0，新增目录字节 277517274；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T14:43:41.156223-03:00
有效时刻 1413/4410，完整parent 13/60，谓词 29673，审计错误 0，新增目录字节 279760481；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T14:50:27.804198-03:00
有效时刻 1498/4410，完整parent 14/60，谓词 31458，审计错误 0，新增目录字节 297853956；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T14:53:50.779821-03:00
有效时刻 1534/4410，完整parent 16/60，谓词 32214，审计错误 0，新增目录字节 304923309；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T14:54:59.158076-03:00
有效时刻 1544/4410，完整parent 17/60，谓词 32424，审计错误 0，新增目录字节 306356990；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T15:01:48.336791-03:00
有效时刻 1612/4410，完整parent 18/60，谓词 33852，审计错误 0，新增目录字节 320244985；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T15:07:32.998935-03:00
有效时刻 1680/4410，完整parent 20/60，谓词 35280，审计错误 0，新增目录字节 331902528；GPUh=0。尚未进行回顾对照或独立复标。

### 因果锁定检查点 2026-11-15T15:17:57.823784-03:00
有效时刻 1804/4410，完整parent 21/60，谓词 37884，审计错误 0，新增目录字节 353342593；GPUh=0。尚未进行回顾对照或独立复标。

## 模型切换请求处理
用户要求三代理改为Luna high。运行中的模型不能热切换；为避免一个parent首遍混用标注者，A/B在当前完整parent结束后交接，C已在index540完整parent边界交接。既有sol锁不覆盖。C由以model=gpt-6-luna, reasoning_effort=high创建的新代理继续；可用工具不能验证运行时推理档位，配置依据为spawn参数。gate_v4.py允许显式填写label_source和annotator_id，哈希及审计更新已记录。A/B在其clean boundary后同样切新Luna high代理。

## 主标过程偏差与隔离 2026-11-15T15:52:20.794824-03:00
A Luna标注者主动报告index727–730复用了index726的JSON答案模板（全部21项unknown及observability），违反逐时刻独立判断要求。原锁不覆盖，已在shards/a/later_notes.jsonl记录相关键与各lock SHA，文件SHA256=b96105f56e042916bfccc99d7a330443960ce4e35e93e07d823ea7aedfdc1714。这5个时刻标为不可靠训练目标，必须安排新上下文盲复标；不能据当前自报后的说明在复标前查看原答案。index731起明确要求逐图逐点独立判断。该偏差不得隐藏在“全量合规”声明中，最终报告列为限制/排除。

## 接续检查 2026-11-15 BRT
接续时间：2026-11-15（BRT）；上次中断约16:10 BRT，终端关闭后原三个标注代理结束。仅核对了本目录接续文件、自动查询清单、锁定/曝光事件元数据及审计输出；未打开回顾式答案、旧标签说明或dev答案。

按磁盘事件重新审计（不采用过期progress检查点）：2599/4410有效锁，54579/92610谓词，29/60 parent完整，错误0；新增目录499986082字节，GPUh=0。A:1090锁/1470计划，17个完整parent，下一parent仅index1090首帧未锁exposure；B:820/1470，7个完整parent，下一parent仅index820首帧未锁exposure；C:689/1470，5个完整parent，当前parent有133/190锁。总计无其他pending exposure。audit_progress.py已将progress.json刷新至上述检查点；compare与发布仍未允许。A/B待曝光属于父级边界首帧，C为已开parent内继续；后续严格沿门控顺序，旧锁不覆盖。model_transition_log记录三分片在Luna high新标注者接续；逐parent边界执行。审计不验证模型真实注意力，仅验证结构、序列和哈希。

### 接续标注偏差与隔离 2026-11-15T23:54 BRT
新A标注者报告 shard a index1091（piper_45abb12c5e65d82dd0011469, frame 1）复用了前一时刻答案模板，违反逐时刻独立判断。immutable lock SHA256=77e2502cda95aa8d8e175e0a0d1b7d7e4e4a67e336f23b088af72737dc1f479e；未改锁，已在shards/a/later_notes.jsonl追加隔离记录并从可靠训练目标排除。index1090由其称为单独判断。该分片随后仅从门控下一时刻恢复，明确禁止答案模板。此为接续过程新增的自报偏差，最终统计需与此前已隔离的A分片index727–730合并披露；不能以结构审计无错等同语义首遍完全合规。

### 接续盲标进度 2026-11-15T23:58 BRT
主标有效锁2621/4410（55041/92610谓词），完整parent29/60，审计错误0；新增目录504358208字节，GPUh=0，compare/publish门禁仍拒绝。分片：A1098/1470、B830/1470、C693/1470。C在原Luna标注身份下继续当前parent，A/B从父级边界的新Luna身份逐帧继续。此检查点尚未进行回顾对照、复标或发布。

### 后续自检疑点 2026-11-16T00:03 BRT
C Luna标注者在后续帧观察后报告其对P082在同一parent较早source frames 2689、2699、2700、2701的标签可能不一致。按协议未改锁；shards/c/later_notes.jsonl已逐项追加lock SHA、标记为“待独立裁定”并暂排除可靠训练目标。该判断是标注者自检疑点，不代表视频事件真值；因果首遍原答案保持不变。

### 接续检查点 2026-11-16T00:03 BRT
有效锁2645/4410时刻，55545/92610谓词，完整parent29/60，结构审计错误0；新增目录509017850字节，GPUh=0。A1106、B839、C700。retrospective比较和发布仍未允许。later_notes中含A新增模板复用隔离及C的P082疑点隔离；旧锁均未改。

### 接续检查点 2026-11-16T00:14 BRT
有效锁2693/4410时刻，56553/92610谓词，完整parent31/60，结构审计错误0；新增目录518757308字节，GPUh=0。A1123/1470（18 parents完整）、B857/1470（8完整）、C713/1470（5完整）。尚未开放回顾式答案或启动比较、复标、发布。

### B分片可见性元数据修正记录 2026-11-16 BRT
B标注者发现index866之前的lock 00865（frame 986，P061）将可见性元数据写成了谓词答案“yes”。未改immutable lock；已在shards/b/later_notes.jsonl追加记录并绑定lock SHA，注明可见性应为partial（环形/入口处部分遮挡），谓词答案不变。后续可见性分组统计按later_note纠正该元数据；原始锁保持可审计。

### 接续检查点 2026-11-16T00:23 BRT
有效锁2733/4410时刻，57393/92610谓词，完整parent32/60，结构审计错误0；新增目录527291227字节，GPUh=0。A1137/1470（19完整）、B870/1470（8完整）、C726/1470（5完整）。retrospective比较/发布仍关闭。

B后续自检再发现index870（lock 00870，frame 1286）P051/P052可见性类别错写为no_visible而应为partial；原锁未改，later_note附锁SHA并保留答案。分析时依据later_note纠正可见性值。

### Dev隔离访问偏差与B分片接续 2026-11-16T00:30 BRT
B Luna标注者在核查允许脚本中的可见性枚举时，递归grep输出误包含 `data/e98_predicate_labels_dev/review_independent/reviewer_b/review_notes.md` 中一行；模型确认该行内容曾显示给它，但称未用于标注。此为违反“dev标签不读”的访问偏差，不能声称零dev访问。它在gate index875（frame1347）停止；对应exposure已存在但未锁定，之前锁未改。为避免污染后续判断，已用gpt-6-luna high全新上下文接续同一逻辑annotator_id e98c_b_luna，仅从未锁曝光继续；模型只知有访问事件，不获知被显示内容。正式可见性值已从允许的dev README / E97定义核实为visible/partial/hidden；后续分组仅接受该枚举。既有B因果锁中的非规范值将逐项用后续later_note更正元数据，绝不覆盖锁或改变答案。

### 可见性枚举统一
允许读取的E98 dev README规定observability=visible/partial/hidden；E97 schema的对应表达为清晰可见/部分可见/画面不可见。只对新因果目录B分片锁中的observability字段做了机器归一化注记：clear→visible（244条）、not_visible→hidden（75条）、no_visible→hidden（2条）、已自报的P061“yes”→partial（1条），共322条/55个lock。所有原锁不改、predicate答案不改；shards/b/later_notes.jsonl有逐predicate和lock SHA绑定的后续记录，最终按notes归一化分组。归一化脚本SHA256=742d4fabe345933ccc37cbc32144df8159c8e82d8dd4f944a91ebdb950c323f6。脚本仅读取本次因果锁元数据及哈希，不碰回顾答案/其他目录。

### B分片干净上下文恢复 2026-11-16 BRT
新gpt-6-luna high上下文经SSH读取允许的E97 schema/dev README后，重新查看了未锁exposure index875（piper_79f01400c4a5a6a25a342a3f, frame1347），逐谓词判断并以既有逻辑ID e98c_b_luna锁定；门控推进到876。前一会话明确未锁该点，故没有旧答案覆盖。

### 接续检查点 2026-11-16T00:35 BRT
有效锁2767/4410（58107/92610谓词），完整parent32/60，结构审计0错误；新增目录533837011字节，GPUh=0。A1153/1470、B876/1470、C738/1470。全部对照/发布仍未开放。

### 接续检查点 2026-11-16T00:51 BRT
有效锁2833/4410（59493/92610谓词），完整parent34/60，结构审计0错误；新增目录547514206字节，GPUh=0。A1176/1470（20 parents）、B899/1470（8）、C758/1470（6）。比较与发布仍关闭。

### 接续检查点 2026-11-16T01:11 BRT
有效锁2917/4410（61257/92610谓词），完整parent35/60，结构审计错误0；新增目录565423913字节，GPUh=0。A1206/1470（20完整）、B927/1470（8完整）、C784/1470（7完整）。comparison与publish仍关闭。

### C分片后续自检疑点 2026-11-16 BRT
C Luna标注者对source frames 839–841附近的S05/S06转移与P061/P062判定报告疑点；仅将三条锁的SHA和疑点写入shards/c/later_notes.jsonl，暂排除可靠训练目标，锁不覆盖。标注者继续按门控顺序；该内容是模型自检，不作为阶段真值或测试参考。

### 接续检查点 2026-11-16T01:30 BRT
有效锁3006/4410（63126/92610谓词），完整parent35/60，结构审计0错误；新增目录584055368字节，GPUh=0。A1237/1470（20完整）、B960/1470（8完整）、C809/1470（7完整）。回顾对照与发布仍关闭。

### C分片后续自检疑点（二）
C Luna标注者对frames 1439–1440附近S02/S04阶段边界报告另一个疑点；对应锁SHA与范围已写入shards/c/later_notes.jsonl，锁不修改，暂排除可靠训练目标并待独立裁定。

### 可恢复的B锁定输入校验拒绝
B在index987首次构造锁输入时使用了`parent_id`，门控拒绝；改成`canonical_id`后第二次又因缺少顶层`answers`拒绝。两次均发生在锁创建之前，未写lock且未推进cursor。核对gate_v4.py契约后按平坦字段、21个predicate键修正已观看index987的输入，随后index987和988成功锁定；无答案覆盖、无帧序列跳跃、审计无结构错误。

### 接续检查点 2026-11-16T01:48 BRT
有效锁3097/4410（65037/92610谓词），完整parent35/60，结构审计0错误；新增目录602673213字节，GPUh=0。A1263/1470（20完整）、B996/1470（8）、C838/1470（7）。比较和发布仍未允许。

A shard at index1264/frame2880 encountered an SSH connection closure during lock submission. Remote read-only check confirmed the exposure existed but lock did not; reconnect verified the same cursor, reused the already-viewed input and locked once. No duplicate lock or sequence advance occurred.

### 接续检查点 2026-11-16T02:05 BRT
有效锁3176/4410（66696/92610谓词），完整parent36/60，结构审计0错误；新增目录616770032字节，GPUh=0。A1286/1470（20完整）、B1025/1470（9完整）、C865/1470（7完整）。comparison/publish仍关闭。

B分片在准备expose index1050时误用系统python，因缺cv2未生成exposure/lock；核对事件目录确认01050.exposure.json和lock均不存在。正确e29解释器在`/mnt/workspace/dengjianfeng/venvs/e29/bin/python`（cv2 4.11.0），与T目录分开；同一逻辑annotator_id从该cursor恢复，未重复或跳过任何帧。

C分片在index894的输入生成中先因JSON布尔字面量写法错误而未生成文件；远端确认该exposure存在、无lock后，以同一已看帧修正JSON并成功锁定一次。随后index895曝光准备误用系统python导致缺cv2；远端确认无895 exposure/lock，e29解释器含cv2，已要求同一标注者用正确解释器从该cursor继续，不重标894、不跳帧。

### 接续检查点 2026-11-16T02:29 BRT
有效锁3270/4410（68670/92610谓词），完整parent37/60，结构审计0错误；新增目录635170647字节，GPUh=0。A1314/1470（20完整）、B1060/1470（10完整）、C896/1470（7完整）。comparison/publish仍关闭。

A分片index1333曾因把source_frame写成3899而被gate_v4在锁定前拒绝；远端exposure元数据为3989，lock不存在。标注者重看同一已曝光帧，依据exposure设3989后成功锁定，游标推进到1334；未查看后续帧、未覆盖锁。

### 接续检查点 2026-11-16T02:49 BRT
有效锁3371/4410（70791/92610谓词），完整parent39/60，结构审计0错误；新增目录656091670字节，GPUh=0。A1346/1470（21完整）、B1095/1470（11完整）、C930/1470（7完整）。比较/发布仍关闭。

### 接续检查点 2026-11-16T04:02 BRT
有效锁3471/4410（72891/92610谓词），完整parent40/60，结构审计0错误；新增目录677830212字节，GPUh=0。A1376/1470（21完整）、B1138/1470（11完整）、C957/1470（8完整）。比较/发布未开放。

### C分片后续自检疑点（三）
C标注者对index961/source_frame1319的S05/S06边界判定报告疑点；已在later_notes以lock SHA追加隔离，原锁不改，暂排除可靠训练目标并待独立裁定。

### 接续检查点 2026-11-16T04:28 BRT
有效锁3593/4410（75453/92610谓词），完整parent43/60，结构审计0错误；新增目录706076418字节，GPUh=0。A1415/1470（23完整）、B1189/1470（11完整）、C989/1470（9完整）。比较/发布未开放。

### 接续检查点 2026-11-16T04:52 BRT
有效锁3695/4410（77595/92610谓词），完整parent45/60，结构审计0错误；新增目录728672139字节，GPUh=0。A1445/1470（24完整）、B1227/1470（11完整）、C1023/1470（10完整）。比较/发布仍未开放。

A index1450 exposure的两次取图失败生成零字节本地文件，未观看/未锁定。改用较长SSH保活后取回非空同一门控图，确认观看后成功锁定；未提前曝光或锁定后续帧。

### B分片index1231锁后派生输出恢复
index1231锁文件已在gate抛出AttributeError前以独占方式写入，stage行也已追加；predicate与journal lock行缺失。核对实际键后，未重试或修改锁；按原锁SHA=0068910b131904618992f6b98d52d8c17f002938e959b9540a3742ced246d4ed，将21个predicate输出及journal lock事件从该不可变锁补写，并在B later_notes记录。原annotation的observability为标量partial，恢复时扩展为21谓词均partial，谓词答案不变。恢复脚本SHA256=74d36dd6b2df06a0960dfbaaa4c762a13ab79d4084072965deb5c3a53adcee1d。标注者已要求后续统一传observability字典，cursor自1232继续。

index1231部分提交修复后的原audit_progress.py复核：3726有效锁、其余结构错误0，但仍对该锁的scalar observability报类型错误；此锁属于已记录的标量partial恢复例外，derived JSONL已完整补齐且锁未改。最终将以新版本审计器读取later_note并严格验证标量恢复例外，旧audit脚本和SHA保留不改；在新审计通过前不发布。

### A分片完成 2026-11-16 BRT
Shard A gate返回ALL_LOCKED：1470/1470有效锁、25 parents、GPUh=0，无A分片审计错误。A未读取或触碰B/C。全局仍IN_PROGRESS，因B/C未完成且原审计器报告B index1231已记录的标量observability恢复例外；A分片完成不触发比较或发布。

### 全分片可见性归一化与strict audit v3
对因果锁的可见性元数据追加SHA绑定later_note：A observable→visible1194、not_observable→hidden60；B clear→visible244、not_visible→hidden75、no_visible→hidden2、P061 yes→partial1；C unobservable→hidden47、中文三种枚举按dev映射63、not_applicable→partial836。共2522个原始枚举纠正，另B index1231标量partial扩展到21个谓词可见性字段。对C中answer=not_applicable同时污染visibility的836项，partial仅作中性导出兜底，并排除visibility分组统计；所有谓词答案均未改。统一修正记录在`data/e98_predicate_labels_train_causal_v1/observability_later_notes_v1.jsonl`，脚本SHA256=bce2d0869d9bd412c6f6d14dac8a3d990959664e00577229fc7a3179a498a102。

strict audit v3读取分片later_notes和上述归一化记录，并以lock SHA校验例外，不修改旧audit脚本。C later_notes曾有一行包含两个JSON对象、由字面反斜杠n连接；原文件精确备份`shards/c/later_notes_preformat_backup.jsonl`，原SHA256=b592cec5cdc143e5a867d309e0d7c800dd110ec01c3159d71158116fbb4b5da1，修复后的JSONL SHA256=aefe8aa18531bc7bfb0839c528b2a662de157cf513ab46769471c4da5081f11f，语义记录拆分而非丢弃。

2026-11-16T05:21 BRT strict audit：3776/4410有效锁，0错误，46/60 parent完整，新增目录745148725字节，GPUh=0；A1470/1470、B1255/1470、C1051/1470。首遍未完成，比较/发布仍关闭。strict审计脚本SHA256=4a713eac1de816167d45964a2ce67c1876de78580b58268ff6aebb39e5fe3288。

C分片index1072/source1619因SSH参数键入BatchBatch导致输入复制失败；远端确认exposure存在、无lock。重连后沿同一exposure锁定一次并顺序推进至1080，无帧跳跃/覆盖或新later_note。

### 接续检查点 2026-11-16T05:53 BRT
strict audit v3：3878/4410有效锁（81438谓词），0错误，47/60 parent完整；新增目录766452880字节，GPUh=0。A1470/1470、B1300/1470、C1108/1470。因果首遍仍未完成，比较与发布未启动。

### C分片后续自检疑点（四）
C标注者对indices1126/source3000与1127/source3001的P022报告疑点，later_notes已记录exact keys及lock SHA，待独立裁定并暂排除可靠训练目标；原锁不改。接续后的index1123输入脚本语法首写失败未触发门控，已从同一已看帧修正并锁一次；index1125遮挡歧义通过unknown策略继续，没有查看未来帧。

### 接续检查点 2026-11-16T06:25 BRT
strict audit v3：3984/4410有效锁（83664谓词），0错误，49/60 parent完整；新增目录787211104字节，GPUh=0。A1470/1470、B1355/1470（13 parents完整）、C1159/1470（11完整）。回顾对照和发布仍关闭。

### 接续检查点 2026-11-16T06:56 BRT
strict audit v3：4087/4410有效锁（85827谓词），0错误，52/60 parent完整；新增目录809576651字节，GPUh=0。A1470/1470、B1408/1470（15 parents完整）、C1209/1470（12完整）。首遍/对照/复标/发布尚未完成或启动。

### 接续检查点 2026-11-16T07:29 BRT
strict audit v3：4187/4410有效锁（87927谓词），0错误，53/60 parent完整；新增目录832326781字节，GPUh=0。A1470/1470、B1463/1470（16完整）、C1254/1470（12完整）。

### B分片完成 2026-11-16 BRT
Shard B gate返回ALL_LOCKED，1470/1470有效锁、18 parents，GPUh=0。交接后仍使用e98c_b_luna逻辑身份，未读取回顾式训练答案；全局仍等C分片完成后再审计。

C index1267/source2041曾在SSH无响应期间提交锁命令。首轮远程状态检查当时未见lock；命令随后异步完成。第二次提交被gate安全拒绝（must expose before lock），远端复核显示仅一个01267.lock及1268个总锁；没有重复锁或后续帧曝光，cursor继续从1268。

### 用户指示继续后的接续检查与标注上下文交接（2026-11-16 BRT）

用户在报告上一次中断后要求“继续完成”。检查实际不可变锁计数：A1470/1470、B1470/1470、C1378/1470；progress.json显示的C=1286已过期。C索引1378已有曝光元数据（canonical_id=piper_d9c59e4eb4a1e5b3eb7296a3，source_frame=1500，viewed_frame_max=1500，frames/01378.jpg SHA256=5815eafa4605616ed8897723ec38b64eb1d4d70800693b3491b176fca5bce4e3），没有对应锁。原C子代理在该曝光后随用户主动中断结束。为不跳过或重曝当前时刻，已创建新的gpt-6-luna high盲上下文，从既有曝光核验并继续C分片；保持逻辑annotator_id=e98c_c_luna、label_source=model(gpt-6-luna)，报告将明确此为上下文交接，非无缝同一运行实例。新上下文不得读取主锁、later_notes或回顾式/开发集答案。交接开始时C实际锁数1378，GPUh=0；该parent尚余source_frame1500、1800、1967三个时刻，之后继续C余下assigned parents。此段记录交接而不回写任何锁。

### 接续上下文的允许文件统计信息披露（2026-11-16 BRT）

恢复标注的gpt-6-luna high上下文按原任务允许读取`notes/e98_dev_labels_20261114.md`作为因果协议参考时，注意到该笔记包含dev标签总体分布统计。笔记是明确许可读取的文件，不是逐时刻dev答案文件；未读取任何dev答案/逐parent结果。收到提醒后，不再查看该统计，也不得将其用于训练标签判断、阈值或其他选择。因总体先验暴露仍可能构成偏倚风险，作为局限披露。该上下文已在现有C曝光顺序完成source_frame1500与1800，继续处理source_frame1967；锁定答案不回改。

### 第二次用户中断后的parent边界接续（2026-11-16 BRT）

用户未撤销“继续完成”目标，但在长时间执行中再次中断对话；之后确认无子代理仍运行。按实际锁文件核验C=1432/1470，A=B=1470/1470。上一parent的最后锁为index1431；index1432是新parent `piper_f99e0a3b82bf990d24247429` 的source_frame=1首个时刻，已在parent边界曝光，viewed_frame_max=1，图页SHA256=7fc44f799d601c1fcb47116bf849531ab9cef844e9ab75f5d74ce734eb7a82e6，无对应锁。由新的gpt-6-luna high上下文从此完整parent边界接续，保留逻辑annotator_id=e98c_c_luna，单parent单标注者约束未跨parent拆分。该上下文不读取任何primary lock/later_note或旧训练标签；将独立处理当前曝光并完成余下C分片。GPUh=0；首遍未完成，审计/回顾对照/独立复标/发布未启动。

### 主因果首遍全量完成与发布（2026-11-16 BRT）

C在第二次接续后于完整parent边界恢复，使用新gpt-6-luna high上下文完成剩余C任务；C门控返回ALL_LOCKED。三分片锁计数A/B/C各1470，共4410时刻、92610谓词、60 parents，GPUh=0。首轮全局严格审计发现C index1428与1429的可见性元数据含`clear`与`absent`两种自由文本，不是允许枚举；仅读取这两锁的observability元数据。追加SHA绑定归一化记录：clear→visible，absent→hidden（将“当前视图不见”解释为hidden），4个元数据字段，因果答案和锁均不变。随后strict audit v3为4410/4410时刻、92610/92610谓词、60/60 parents、errors=[]、causal_blind_pass_completed=true，新增数据目录877165594 bytes，GPUh=0。`audit_progress_v3.py --publish`通过，发布`predicates_causal.jsonl`和`stage_causal.jsonl`；发布后复核无错误，因果回顾对照门槛已打开。

完整接续链包含：A/B早期gpt-6.1-sol与后期gpt-6-luna模型转换；C在index1378前的会话中断，之后在完整parent边界上由新的luna high上下文继续；再有一次对话中断，实际核验发现其前一parent已完整、下一parent首时刻(index1432/source_frame1)已曝光，因此新luna high上下文在该新parent首时刻接续。所有分界均在parent边界，parent内不混标，逻辑ID仍为e98c_c_luna；报告明确这是操作上下文交接。C index1372–1377六条锁由标注者later_note记录疑似答案模板复用，已隔离出可靠训练目标；锁未改。最初A分片index727–730模板复用、A index1091异常等历史隔离记录仍适用。


### 回顾式对照与操作性时间偏移（2026-11-16 BRT）

首遍全部锁定后运行`compare_after_lock_v3.py`，其门槛调用严格审计v3。与旧回顾表按(parent,source_frame,predicate)对齐9956条；这是旧表中存在且可对齐的子集，并非全量92610条。四值一致率50.80%，Cohen κ=0.3246；双方均为yes/no的4780条一致率75.40%。回顾式yes共3313条，其中因果为no/unknown的990条，占29.88%。按因果可见性：visible n=4617，一致率62.25%、κ=.4217、yes/no子集80.71%；partial n=2773，一致率35.20%、κ=.1371、yes/no子集65.06%；hidden n=704，一致率70.74%、κ=.2993（无双方二值子集）；未测量/门控默认partial n=1862，一致率38.13%、κ=.1664、yes/no子集59.29%。回顾标签自身未带可用visibility (`missing`)，所以不把它当作真实可见性分组。各谓词完整统计在`comparison_v1/report.json`。

回顾yes/因果no或unknown共990条。按旧cycle/attempt上下文计算首个观察到的因果yes偏移（不是独立验证的物理事件时刻），657条有同上下文参考；333条没有该上下文中的观察到yes。657条中482条回顾yes早于首个因果yes（73.36%）；领先帧差中位数119，P25=41.5、P90=359、最大900帧。其余可匹配样本可能落在参考帧之后。该统计为稀疏采样、模型导出的cycle/attempt内操作性观测偏移，不是物理事件提前量或训练真值，不用于阈值/模型选型。完整记录见`event_offsets_v1/report.json`。

10%独立复标已按固定seed98、441时刻、三份review shard启动。代理只收 gate 提供的原始因果历史图页，不接触主答案或旧回顾答案。RB在按指示读取允许的dev README时误用`cat`扫过后续总体review统计，但没有打开逐时刻dev标签、回顾表或主标签；已要求不把总体分布用于先验/答案决策，并在报告披露。这是统计先验暴露风险，不影响主首遍锁；review代理继续从其当前未锁曝光点盲标。


### 独立复标完成与Sol/Luna分层自一致性（2026-11-16 BRT）

固定seed98抽样441/441时刻、9261谓词，review audit mixed v1全部通过；50个parent覆盖，错误0。已锁Sol结果未重跑：Sol复标324时刻；Luna high在完整parent边界接续117时刻。RB因阅读许可README时扫过dev总体review统计已单独披露，但未读逐时刻dev/主答案；该总体先验风险不等于人工准确度。

合并展示的混合复标对主标注：四值一致率50.33%、κ=.3318；双方yes/no的2794条一致率91.59%；stage一致率52.38%、κ=.4727。必须按精确模型配对解释：primary Sol→review Sol共202时刻/4242谓词，一致率73.95%、κ=.6353、yes/no子集2235条一致率96.64%；primary Luna→review Luna共117时刻/2457谓词，一致率30.97%、κ=.0653、yes/no子集96条一致率83.33%；另primary Luna→review Sol为122时刻/2562谓词，一致率29.78%、κ=.1077、yes/no子集463条一致率68.90%。Sol→Luna没有配对样本。Luna同模型一致性偏低；RA Luna末14条阶段/谓词均用unknown（因目标身份关系证据不足），稀疏历史也使RC若干样本unknown，这是结果解释的重要限制，未过滤以抬高指标。结果来自同模型家族复标，不是对人类真值的准确度。详情`review_self_consistency_mixed_v1/report.json`。

本review从Sol到Luna的转换均发生在各shard完整parent边界；既有Sol锁原样保留，新parent由Luna high标注，review audit接纳两个显式model label并分模型输出。复标者只看每个样本当前图和gate选出的过去图，历史条件不同于主标注的长历史；图像共享偏差仍可能存在。


### 源数据最终身份校验（2026-11-16 BRT）

在主标签与独立复标都完成后，运行新增只读脚本`source_audit_verify_final.py`（CPU、setsid/nohup、独立状态文件）按source_identity.json逐文件重新读取并核对SHA256、mtime、size、视频帧范围与query范围；结果COMPLETE：60/60 parents、180/180 camera videos、55,292,895,938 bytes读取、GPUh=0、source_files_unchanged=true、永久排除demo3 episode142确认。source_identity.json SHA256=4034526ca9e2e60df09cb5c766f099dcf0701352c292b887caeb842dd6fbe795；结果见`source_audit_final_verification.json`。全程没有改动源数据、旧标签、dev标签或其他shard。


### 后锁疑点与训练池隔离汇总

按锁SHA绑定later_note所标记的模板复用或疑似谓词/阶段不一致键去重，共23个(parent,source_frame)时刻从可靠训练目标池排除，等待独立人工/后续审查；这不会删除发布JSONL中的原锁，也不把模型标签当作真值。包括A index727–730的4条模板复用、A index1091的1条模板复制、C index1372–1377的6条模板模式复用，以及其余10条遮挡/阶段边界/P022/P082疑点。raw lock及键SHA清单在分片`later_notes.jsonl`和根`later_notes.jsonl`中。可见性元数据归一化记录未计入此训练池排除数。


### 最终交付状态

状态COMPLETE（2026-11-16 BRT）。主标签文件：`data/e98_predicate_labels_train_causal_v1/predicates_causal.jsonl`（4410×21行，SHA256=b99bef4e0e0c1fb9373286acbb697d252f0bd990dead0469474237a89026d4c4）与`stage_causal.jsonl`（4410行，SHA256=e775d5829a0d10962564e1c062b44aee6b86717d1c7a27e9c59c5622813c0141）。查询导出器`export_queries.py` SHA256=79ef6465a3fa36eff7811f84429dc6c4ec0831bde4f6f47fe1ba9c0daa8f71d1；全部关键脚本见`release_scripts.sha256`。final manifest记录源身份、查询、输出与审计哈希。新数据目录最终约1.05GB，远低于10GB上限；GPUh=0。EXPERIMENT_LOG与worklog已追加E98-C完成条目。E102-S2留待后续，不在本轮训练或选型。
