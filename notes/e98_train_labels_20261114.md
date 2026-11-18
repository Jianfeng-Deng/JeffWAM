# E98-L-train 阶段与完成谓词训练标注

- 开始 BRT: 2026-11-14T14:03:46.126058-03:00
- label_source=model(gpt-6-astra)，非人工；仅训练用途。
- 纯 CPU，GPUh=0；rev4 train only，永久排除 demo3 ep142。

## 选择与划分审计
待完成。

## 抽帧与标注协议
待完成。

## 独立复标
待完成。

## 验证、资源与结果
状态：进行中。

### 选择冻结

seed=9801；P3 V3窗口数 {'demo3': 11598, 'demo4': 13750, 'demo6early': 4723, 'demo6native': 4706}；Hamilton配额 {'demo3': 20, 'demo4': 24, 'demo6early': 8, 'demo6native': 8}。候选为E83-A代表的rev4训练parent（仅元数据筛选），不要求P3接纳以保留失败样本。排除ep142及其parent。60个parent互异，demo3占1/3。独立复标seed=9802，6个parent，名单单独冻结。

### 抽帧进度

60个parent三视角粗抽帧完成，间隔240原帧（名义8秒），另保存首末帧；源映射 /mnt/workspace/colleague-a/Datasets/piper/ → /mnt/workspace/dataset/piper/，只读原位视频。首次旧路径失败已保留日志，未使用GPU。当前frames约107MiB，远低于8GiB。粗抽帧完成不代表标签完成。


### Visual review progress 2026-11-14T14:28:56.754020-03:00
All 60 coarse contact sheets reviewed; targeted refinement reviewed for indices 1-47, with observations saved in primary_visual_notes.jsonl. Parent 0 has a separate refined draft. Independent reviewer completed 6 parents, reported 93 stage rows and 855 predicate rows; semantic outputs remain unread by primary annotator. Agreement pending primary freeze. Frames approximately 230 MiB; GPUh=0. Full primary JSONL and keyframe coverage are unfinished. Both passes are retrospective model annotation, not a causal blind first pass. label_source=model(gpt-6-astra), non-human.

## 2026-11-14T14:46:02.616293-03:00 primary keyframe progress

All 60 parents have completed coarse and first targeted visual inspection. Primary structured annotations are completed for indices [0, 1, 2, 3, 4] with 37 stage rows and 394 predicate rows. These include stage start/middle/end, boundary-neighbor samples and uniform 10-second samples. Additional boundary2 inspection corrected near-entry phase boundaries for indices2,3,4. The 6 independent outputs remain isolated pending the primary freeze. All labels are nonhuman model(gpt-6-astra), retrospective, with causal_blind_pass_completed=false. Frames occupy 269.23 MiB. GPU use remains zero. Remaining work includes indices5-59, dataset-wide validation, independent agreement and final manifests/log entries.

## 2026-11-14T14:54:20.006689-03:00 primary structured progress

Completed indices0-12, 83 stage rows and 898 predicate rows. All stage start/middle/end and planned uniform/boundary samples are explicitly answered. Parent8 native-resolution left-wrist inspection identified the thin line behind the head at2160 as a background table tie. Its second confirmed threading boundary was corrected to the2161-2280 interval. Parent10 placement/release was refined using1830-1921 samples. Parent11 is an abandoned straight target with uncertain intentional release versus involuntary grasp loss. Independent semantic results remain unread. Remaining main parents13-59, agreement, full validation and final manifest.

### 主标注进展 2026-11-14T15:00:15.786915-03:00

完成索引0–19，共{'completed_parents': 20, 'stage_rows': 114, 'predicate_rows': 1207}。独立复标语义内容尚未读取。纯CPU，GPU用量0。继续核验索引20及之后的关键帧。

### 主标注进展 2026-11-14T15:09:15.744800-03:00

完成索引0–31，共{'completed_parents': 32, 'stage_rows': 169, 'predicate_rows': 1869}。新增边界加密纠正了20、22–26、30、31的粗采样边界。32–43关键帧CPU提取完成。所有主标注继续显式标记回顾性视觉判断，causal_blind_pass_completed=false。独立复标语义内容仍未读取。

### 主标注进展 2026-11-14T15:20:33.565086-03:00

完成索引0–43，共{'completed_parents': 44, 'stage_rows': 252, 'predicate_rows': 2752}。35的短暂穿过后退出及同一目标重试已按多视角记录，34、39、40、42的放下边界增加细采样。44–47四轮循环关键帧提取已分离启动。所有标签为非人工 model(gpt-6-astra)，回顾性视觉判断，causal_blind_pass_completed=false。独立复标语义结果仍未读取，一致率等待主标注冻结。

### 主标注进展 2026-11-14T15:25:30.036745-03:00

完成索引0–44，共{'completed_parents': 45, 'stage_rows': 288, 'predicate_rows': 3096}。44包含四根扎带循环，第一轮释放与第四轮新料抓取边界已逐步加密修订。48–59关键帧抽取已启动并写独立状态文件，纯CPU。源起点27、28的uniform_10s角色按原始源起点计算，帧号保持原值。独立一致率与最终汇总仍待全部主标注冻结。label_source=model(gpt-6-astra)，非人工，回顾式标注。

### 主标注进展 2026-11-14T15:37:11.968971-03:00

已完成 48/60 parent，396 阶段，4197 谓词答案。frames 684.71 MiB，GPUh 0。全体关键帧抽取已结束。标注为 model(gpt-6-astra)，非人工，回顾式判读。

已完成主标注50/60 parent，共468段阶段、4889条谓词答案。第51个parent的26张关键帧表已逐张判读，正细化多次重试与放置边界。CPU执行，GPUh为0。独立复核语义结果尚未读取。

## 进度更新 2026-11-14T16:03:24.439427-03:00

主标注已完成 54/60 parent，共 612 个阶段和 6772 条谓词答案。原生数据确认左手直接取料与误提既有完成环的非标准动作，保留 unknown 和恢复阶段及证据。所有标注 label_source=model(gpt-6-astra)，非人工。接收区域谓词历史修正、候选下一阶段覆盖审计和独立一致性统计仍待完成。回顾式看帧，causal_blind_pass_completed=false。GPUh 为 0。

## 主标注进度 2026-11-14T16:16:02.553540-03:00

已完成57/60 parent，708阶段、8353谓词。第57号已看完关键帧，正在CPU抽取108个边界加密帧。主标注尚未冻结，P081协议纠正、候选下一阶段覆盖和独立一致率仍待完成。label_source=model(gpt-6-astra)，非人工。

### 2026-11-14T16:27:07.961138-03:00 主标注60/60

已完成789阶段、9956谓词。逐条帧归属检查无错误。P081接收区域定义修正正在逐帧核查，候选下一阶段覆盖及独立一致率待汇总。全部为回顾性模型标注，因果盲看未执行。GPUh为0。

## 回顾性版本交付与协议缺口 2026-11-14T16:35:15.672531-03:00

label_source=model(gpt-6-astra)，非人工。状态为 retrospective_delivered_protocol_incomplete。

### 选择与范围

seed9801 分层无放回选择60个rev4训练parent，demo3/demo4/demo6early/demo6native分别20/24/8/8，demo3占33.33%。P3 V3窗口数分别11598/13750/4723/4706，按最大余数法分配。候选来自可定位的E83-A训练parent，选其最长连续代表片段。选择仅用元数据，E83叙述仅供参考，答案依据三相机原帧。永久排除canonical已验证，60个parent全部重新核对为train。源视频原位读取。

### 已交付内容

- selection.json 与 select_train.py 保存选择清单和可复现脚本。
- stages.jsonl 含789个阶段，predicates.jsonl 含9956条答案，覆盖4410个源帧。
- 3441个必需抽帧点覆盖完整，包括阶段起中终、边界前后邻帧与原始片段起点起每300帧。逐条阶段归属、相机文件存在、历史证据不晚于当前帧、P073未知和输入SHA检查通过。
- 20段阶段保留unknown，包含左手直接取料、误提既有完成环、静态或无法归入既定阶段的片段。
- 252帧P081重新看帧，250条答案更正，2条因裁切仍unknown。修正层及原始答案均保存。另更正一条P082的桌面措辞为筐内。
- 每条答案记录可观测性、证据相机、源帧号、历史证据和回顾性答案。因果字段为空。边界为采样证据区间估计。

### 独立重标一致率

seed9802抽6个parent，独立子代理不继承主上下文且未读主标注。独立产物含93阶段、855谓词，查看451个源帧的三相机证据。主版本冻结后读取独立语义结果。

- 精确同parent、同源帧、同谓词共有156对，四分类一致116/156，即74.36%。
- 双方均为yes/no的87对，一致76/87，即87.36%。共同配对覆盖独立答案18.25%，其余因抽帧位置或提问不同未进入该分母。
- 共同配对中的unknown率，主标注32.69%，独立标注33.33%。逐谓词混淆矩阵与逐parent结果保存在review_independent/agreement.json。
- 时间加权阶段一致率78.91%。77个同cycle/attempt/stage且重复次数相同的边界，绝对差中位20帧、P90 76帧、最大220帧。区间重叠67/77，即87.01%。
- 不同次数或attempt定义的组保留未匹配记录。左手取料与第四轮误提完成环导致部分阶段及attempt分歧。双份原始结果保留，共识裁决尚未执行。

### 协议缺口与使用范围

当前为回顾性模型标注训练目标。B2因果首遍盲看未执行，causal_observable_answer全部为空，causal_blind_pass_completed=false。当前上下文已看过未来帧，真正的因果首遍需另用未接触这些视频未来帧的标注上下文。

候选下一阶段问题覆盖不完整。按实际后继阶段、必需抽帧点作下界审计，需6538个谓词对，已有0个，缺6538个。该审计还未穷尽状态图所有分支。当前阶段起中终答案已覆盖，完整B2协议验收仍有待办。

需用户决定是否先将本回顾性版本用作训练目标，或要求在训练前完成候选下一阶段补标及全新上下文因果盲看。独立模型一致率仅描述标注一致性，不作为测试真值、阈值或checkpoint选择依据。

### 资源、命令与目录

CPU执行，GPUh=0。frames为855680338字节，约0.856GB，含独立复核与P081审计联系表。产物目录约0.883GB。未下载数据集或大权重，未复制源视频。

远端目录为 /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e98_predicate_labels_train。长抽帧通过setsid nohup与状态文件运行。主要命令为 select_train.py、extract_frames.py、各e98_refineXX.py与author_completeXX.py、e98_apply_p081_audit.py、e98_freeze_primary.py、e98_agreement.py、e98_finalize_retrospective.py。SSH使用 ssh -o BatchMode=yes <gpu-server>。manifest.json记录输入输出SHA、计数和未完成项。
