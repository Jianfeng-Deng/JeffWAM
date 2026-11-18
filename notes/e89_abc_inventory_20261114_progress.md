# E89-A ABC-130k 盘点（2026-11-14 BRT）

状态：盘点交付完成（候选清单，非训练放行）；只读数据源、CPU、无训练/特征提取；产物≤3GB，review≤500MB，后续特征缓存≤60GB。

## 1 环境与来源
见后文对应编号的分批完成记录与最终汇总。

## 2 任务分类
见后文对应编号的分批完成记录与最终汇总。

## 3 抽样质量与标注
见后文对应编号的分批完成记录与最终汇总。

## 4 子集与成本
见后文对应编号的分批完成记录与最终汇总。

## 5 世界头时间合同
见后文对应编号的分批完成记录与最终汇总。

## 6 风险、许可与决定项
见后文对应编号的分批完成记录与最终汇总。

## 执行记录
- 创建新目录 data/abc_inventory_v1；未访问 Piper dev/test/sealed_test。

### 1.1 环境完成（00:18 BRT）
独立 Python3.12 venv：data/abc_inventory_v1/env/venv；未改 e29/系统环境。仅安装 mcap 1.6.0、mcap-protobuf-support 0.6.0，必需依赖 protobuf 7.36.2、lz4 4.4.5、zstandard 0.25.0。无额外 foxglove 包：MCAP 内嵌 protobuf schema 可由 DecoderFactory 动态解码。安装日志、pip report、各 wheel SHA256 见 env/install.log、install_report.json、packages.json。已可读取 episode 并 CPU 解码三路中段1秒，探得640×480 H.264；env/smoke_decode.json记录。GPU0。

### 2.1 元数据完成（00:21 BRT）
train报告197任务/129032条/42980有标注/3541.1h；val189任务/1671条/0标注/49.6h。逐任务小时已四舍五入，求和train3541.0h、val49.8h，不替换报告总数。197项分A16/8998条/1335有标注/233.5h，B27/20927/5586/621.8h，C140/87665/30690/2263.5h，排除14/11442/5369/422.2h。A标注14.84%，B26.69%。分类为工程相关性判断，不是模型评分或人工效果真值。
实际train有201个任务目录，额外4目录2219条未在报告：assemble_paper_box(468)、insert_wireless_bluetooth_earbuds_into_charging_case(486)、pack_books_pencil_case_and_school_supplies_into_a_backpack(385)、unlock_a_padlock_with_a_key(880)。可能别名但尚无来源合并证明，全部隔离，不把它们增算为报告净量。tasks.json、stats.json、本地LICENSE缺失；README的示意结构不等于磁盘实物。val/dagger仅登记/报告阅读，不读episode。
候选仅train：16A+10B+2C=28任务；已完成24151条MCAP summary与首尾小块哈希读取。全库不跑du，不解析或使用动作/状态消息作为P1数据，不加载模型。

### 3.1 抽样检查完成（00:32 BRT）
固定seed891114，逐任务独立SHA256派生随机流；16个A每任务15条，10B+2C每任务8条，共336条无偏随机样本；另6条只为补看标注的富集样本，不混进站型/标注覆盖率估计。342条/1026路三相机中段约1秒CPU解码全部通过，共32993帧、相邻完全相同解码帧0；这只说明抽样中段无比特级冻结，不能排除片外静止/冻结或完整视频损坏。ZED只解码顶左+双腕，顶右仅做时间索引统计，未把立体右目计入三视角配方。无偏样本站型RS277/336=82.44%，ZED59/336=17.56%；候选28任务全体summary站型另见后表，不外推全数据集。样本时长P5/P50/P95为43.64/106.64/233.01秒。
CPU方法：MCAP索引定位中段附近，动态protobuf解析CompressedVideo，按format提取Annex B的data；从前一IDR/CRA及参数集起在内存拼接，经ffprobe探尺寸/codec、ffmpeg threads=1解码原时间戳覆盖的中段1秒，不落盘视频。发现部分流间隔中位数16.7ms但均值约33ms，早版按中位fps换算seek产生18路假零帧；已改为原packet时间戳→解码帧号trim，复核43条后零失败，rawclock_v1审计留存。此类抖动不能用“容器标30Hz”掩盖。ffmpeg6.1.1；pip check通过。
标定检查扩至候选所有21371条RealSense：实际三路640×480仅16444条，848×480有4715、1280×720有195、800×600有12、1280×1024有5。不能用top-camera话题直接断言640。子集仅选三路标定均640×480；其他RealSense分辨率本轮排除。MCAP内嵌schema可解码CameraCalibration；339/342样本具有完整相应标定，缺失者排除。intrinsics存在不等于世界外参已核，不用于动作/FK。

### 3.2 子任务文本检查完成
93条带标注样本合2070段；每episode段数5–55，P50=21/P95=41；段长P50=3.97s/P95=28.18s，最短0秒、最长203.23秒。126段mistake、84段reset both arms（合10.14%），还有大量adjust/重复pick-up；并非全部是完成进展。时间戳单调、相对相机重叠区超出1秒的边界0，但存在同一时刻零长标签，使用时须显式舍弃零长段、把mistake/reset隔为事件，保留原始文本。语言一般为可读英语命令句，细粒度拿起/放下与长达数十秒绑带并存，不能当作统一阶段粒度。
8条episode×3段中点×3视角追加联系表：扎带、解缆、花束、领带、插头、钥匙环、笔帽、拉链袋；图文粗语义一致，遮挡较多，未做逐帧起止精确真值核验。样例“Pick up the cable...”与“Place the cable...”可辨抓取/放置，“Tie the half-tied ribbon...”可见花束丝带；扎带被叫ribbon，物体词存在混用。报告者Codex目视属于训练池审阅，非人工测试真值。
本地README/格式说明、公开论文§2/附录H未给逐条标注作者/模型版本或人工复核链，因此human/model来源=unknown，不凭语法猜测。可作待清洗的长程阶段训练文本/辅助目标；当前P1统一简单文本，不直接喂阶段oracle，不作阈值/测试真值/checkpoint选型。论文[原文](https://arxiv.org/html/2606.27375v1)说明有连续子轨迹子目标及条件化，但不是本地导出标注来源证明。

### 6.1 许可核对与版本风险
本地根目录不存在LICENSE/NOTICE；已核README元数据license=apache-2.0及License段，不能写成“核过本地LICENSE全文”。上游[HF当前目录](https://huggingface.co/datasets/XDOF/ABC-130k/tree/main)显示Apache-2.0但文件访问受gated条款限制，LICENSE链接无法读取；本轮不登录、不下载或绕过限制。[发布方说明](https://www.xdof.ai/blog/abc-130k)同样声明Apache-2.0。依据[Apache官方2.0全文](https://www.apache.org/licenses/LICENSE-2.0)：许可复制、修改、分发及衍生作品；再分发须附许可证、显著标记改动、保留相关版权/专利/署名，有NOTICE时保留相关声明；有相应专利授予及提起相关专利诉讼的终止条款，不授予商标权；按现状提供且不保证、不承担约定范围外责任。授权声明不替代隐私权，也不证明来源链和第三方内容无风险。缺失许可证文件是交付溯源缺口，不能宣称法律文件闭合。用户已批准本地数据盘点，不新增审批阻塞。未来若发布派生物需补齐实际许可证/NOTICE与上游revision。
上游文档/论文数量与本地meta不同（论文195任务等），且annotation独立更新；只以本次本地source_snapshot、源stat+首尾哈希及派生清单哈希固定本盘点版本，不把未来上游main当本地revision。

### 5.1 时间合同与估算方法（非窗口缓存）
按本轮用户的v3合同使用30Hz规范网格；未来+10/+25/+50/+100/+200/+300分别为0.333/0.833/1.667/3.333/6.667/10秒，历史0.2秒=6规范步（E81旧版仅三端点，本轮以用户六端点要求为准）。网格原点=三路首次曝光时刻最大值，t_i=start+round(i×1e9/30)。O每路取timestamp≤t的last-available，不用对称最近邻；历史同理t−0.2。实际接收时间存在时还必须要求receipt≤决策时刻，否则该帧不可得。W可对目标时刻做最近邻，容差=该路中位原生帧间隔一半、上限50ms；30Hz约16.7ms，60Hz约8.35ms。不以容器CFR/序号假装时钟统一，不线性插值图像，不把未来最近图喂给O。
每路O age≤100ms；同组三路曝光跨度≤50ms；W最近帧组三路跨度也≤50ms。整个[t−0.2,t+10]任一路不得跨>100ms时间洞；整episode有效窗口率<50%、实测均fps<20或>65、任一路gap>1秒、>100ms gap比例>0.5%者排除。阈值为本轮训练候选工程规则，不源于模型生成标签或dev/test；详细排除原因在selected_timing.jsonl。强制完整六端点，有任何目标缺失便丢该窗口，不尾帧填充，不跨episode/拼缝。只保留计数和摘要，未建立窗口缓存。bitset优化与独立逐点参考在真实130.7秒扎带episode上均给出2896/3616个有效窗口。
实际MCAP元数据记录名为episode-metadata（并非格式文档示意的session-metadata），含session_id、camera_type、camera_polling_fps、auto_exposure等；抽样与最终清单另核session_id=目录UUID。D405示例polling_fps=60而存储曝光流约29.9Hz，polling不等于有效曝光/保存率。protobuf timestamp与log_time在已解码段一致，但这是导出日志时基，不是独立曝光/接收延迟测量；没有独立receipt字段、曝光起止/持续时间或传感器时钟标定。故本轮仅证明记录时基上的因果选帧与同步可用性，无法证明物理在线接收时延。auto_exposure=true也不提供实际曝光时间。缺口不以人为时间平移修补，训练前应把假设写入合同。
10Hz网格的观测可按同样last-available和age规则从原曝光读取，历史为2网格步；若保持六个物理时距，则目标仍查原生曝光，不把+10规范帧改成1秒。sample_audit里的10Hz估计正是“较少anchor、保留原30Hz物理端点”，不是新的+10…+300/10Hz合同。若真正把模型合同改成10Hz而仍用整数+10…+300，则未来变成1/2.5/5/10/20/30秒，训练问题、窗口有效率与短片淘汰率全部改变，须新版本并给R/P两臂一致；本轮未改模型。

### 3.3 联系表结论与覆盖边界
28候选任务各有随机episode起/中/终×三视角联系表，已逐任务目视一张；另有8条episode的三段标注中点联系表，24段粗语义可对上物体/动作。没有在这些显示帧中观察到人脸/人体，但这不是整episode/全子集隐私认证；未逐帧审查者字段保留unknown。没有使用人脸识别模型或生成测试标签。具体episode、观察和版本在visual_review.json及annotation_visual_review.json。常见域差异：背景花纹/纸箱/白墙差异；大黑/黄夹爪遮挡锁孔；腕自运动模糊、自动曝光；细线在顶视分辨率有限。roll_the_ties实为宽领带，软管粗且刚度不同，袋结/衣物拉链也非塑料锁头；A类内相关程度有强弱，不能由名称全当扎带演示。

### 2.2 候选任务全表与分配（最终版）
候选元数据总计24,151条、738.03h；RS21,371条/641.02h，ZED-X2,780条/97.00h。三路标定均640×480的RS16,444条、原summary跨度约470.00h，仍需逐episode时间闸门。以下RS640小时未做时钟剔除；最终S列按三路共有视频跨度。A/B分类和逐任务目视依据在classification.json，非候选任务只做任务名分类，不冒称看过全部201任务录像。

|任务|类|RS640 条/有标注|RS640 h|随机样本数|S1/S2/S3 h|
|---|---|---:|---:|---:|---:|
|`connect_and_route_the_hose`|A|75/9|2.00|15|1.97/1.97/1.97|
|`place_the_bra_in_the_laundry_bag_and_zip_it_closed`|A|241/34|5.19|15|1.89/4.74/4.74|
|`place_the_fake_fruits_into_the_plastic_bag_and_tie_a_complex_knot`|A|155/6|4.37|15|1.90/3.81/3.81|
|`place_the_food_into_the_grocery_bag_and_tie_it`|A|6/2|0.20|15|0.17/0.17/0.17|
|`place_the_mixed_food_into_the_grocery_bags_and_tie_them`|A|48/0|1.56|15|1.33/1.33/1.33|
|`roll_the_ties`|A|350/349|11.78|15|1.91/5.77/9.29|
|`tie_the_bouquet_of_fake_flowers`|A|279/175|14.78|15|3.83/8.83/8.83|
|`tie_the_shoes`|A|292/23|8.31|15|3.80/5.64/5.64|
|`untangle_the_cables`|A|2225/51|62.13|15|3.80/11.52/37.48|
|`untie_the_knot_of_the_grocery_bags_and_take_out_the_mixed_food`|A|32/1|0.98|15|0.93/0.93/0.93|
|`untie_the_knot_of_the_plastic_bag_and_take_out_the_fruit`|A|174/27|3.31|15|1.90/3.27/3.27|
|`unzip_the_laundry_bag_and_remove_the_bra`|A|378/29|4.76|15|1.91/4.55/4.55|
|`wrap_the_cables`|A|136/7|3.24|15|2.98/2.98/2.98|
|`wrap_the_headphones`|A|909/11|8.81|15|1.90/5.74/5.74|
|`zip_tie_the_wires`|A|57/57|3.02|15|3.02/3.02/3.02|
|`zip_up_the_jacket`|A|519/4|8.08|15|1.90/5.76/6.14|
|`insert_the_pens_into_the_pen_caps`|B|372/59|11.71|8|1.36/2.72/4.57|
|`insert_the_plug`|B|508/53|13.53|8|1.40/2.70/4.56|
|`insert_the_plug_into_the_switch_port`|B|262/16|7.27|8|1.36/2.72/4.57|
|`insert_the_wireless_bluetooth_earbuds_into_the_charging_case`|B|1683/658|59.75|8|1.35/2.72/4.57|
|`lock_with_the_key`|B|205/172|6.97|8|1.37/2.71/4.60|
|`open_the_zip_top_bag_and_remove_the_fake_food`|B|671/110|14.59|8|1.39/2.73/4.58|
|`place_the_food_in_the_zip_top_bag_and_seal_the_zipper`|B|1747/1651|41.68|8|1.38/2.72/4.57|
|`put_the_keys_on_the_keyring`|B|1955/21|65.88|8|1.38/2.71/4.56|
|`remove_the_keys_from_the_keyring`|B|317/22|7.68|8|1.39/2.71/4.56|
|`unlock_with_the_key`|B|261/15|8.34|8|1.36/2.71/4.56|
|`fold_and_stack_the_towels`|C|290/265|12.54|8|0.78/1.51/2.29|
|`fold_the_paper_box`|C|1936/6|77.53|8|0.79/1.53/2.29|

### 4.1 三个嵌套候选子集（最终版）
推荐先审阅S1；S2/S3为扩展候选，不自动扩量或启动训练。S1⊆S2⊆S3，均为ABC train真实视频、三路640×480 RealSense，整episode、目录UUID唯一；被检查源间无UUID/小块身份哈希重复。只允许P1视觉，不使用动作/状态，不进入P2/P3。每任务先有标注、再固定seed+UUID SHA256伪随机顺序；时长不足用整episode轻微超额，不截断/平移标签。组内建议任务等权、任务内episode等权、episode内有效窗口均匀，不让长视频决定抽样概率。

|子集|episode|三路共有录像h|A/B/C h|有标注条数/比例|最大单任务时长占比|六端点窗口有效率|有效锚点数|
|---|---:|---:|---:|---:|---:|---:|---:|
|S1|1,620|50.456|35.14/13.75/1.57|675/41.7%|7.6%|71.86%|3,560,314|
|S2|3,902|100.246|70.04/27.16/3.04|1037/26.6%|11.5%|73.32%|7,064,237|
|S3|5,631|150.182|99.90/45.70/4.58|1401/24.9%|25.0%|74.38%|10,784,484|

小时数是三路共同覆盖的物理录像跨度，**不是全部锚点可训练，也不是独立有效窗口小时**。有效率分母为先去掉0.2秒历史和10秒未来边界后所有30Hz候选锚点；窗口高度重叠，不能把窗口数当独立样本量。以有效锚点数/(30×3600)表示的锚点等价小时为S1=32.97, S2=65.41, S3=99.86，仅用于吞吐/采样估算。
本轮累计严格时间索引检查9,833条（含前期RS不同分辨率检查和后续原生60Hz补查），通过8,240、剔除1,593；失败原因可重叠：{'strict_valid_windows_below_50pct': 1593, 'gap_fraction_gt0.5pct': 629}。排除表见preexcluded.jsonl、逐episode计数/age/gap见selected_timing.jsonl，未检查源不列成质量通过。原生60Hz不是坏数据：最终允许原生均fps20–65并按原时间重采样；初期20–40Hz筛选版保留proposed_subsets/initial_fps20_40/，不可与最终版混用。342抽样内有27条三路640原生高于40Hz样本，全部短段解码通过。

各源视频的sha256字段为`sha256(tag + 文件尺寸 + 首64KiB + 尾64KiB)`，含固定NUL分隔，详见每条hash_scope；**不是视频全文件SHA256，也不证明中部未更新**。清单JSON本身则做完整SHA256，S1S2S3.json索引文件含S1/S2/S3路径与哈希；annotation.mcap体积小才整文件hash。元数据/源stat/mtime和source_snapshot同时固定口径。最终所选5631条另核MCAP episode-metadata.session_id与目录UUID相符，错配0，见selected_source_metadata.jsonl。

### 4.2 成本与60GB硬上限
十进制GB，冻结SigLIP2池化特征每相机帧2×4×768×2=12,288B；三路30Hz每物理小时3.981312GB。下表为30Hz规范网格逐曝光复用前的覆盖量与一次覆盖工作量估算，不是本轮执行的提取任务。吞吐沿用用户给定E86含解码有效218相机帧/s，并额外乘1.25；未在ABC上用GPU实测，重算/抖动/IO/原生60Hz解码开销可能超出。

|子集|30Hz相机帧|30Hz payload GB|GPUh含25%余量|10Hz payload GB|10Hz GPUh含25%|有界实际缓存硬上限GB|
|---|---:|---:|---:|---:|---:|---:|
|S1|16,347,589|200.88|26.04|66.96|8.68|60|
|S2|32,479,801|399.11|51.73|133.04|17.24|60|
|S3|48,658,826|597.92|77.50|199.31|25.83|60|

50h即使10Hz也超过60GB；不能只说S3超预算。60GB纯payload最多容纳15.070h@30Hz或45.211h@10Hz，尚未给索引留空间。建议每个子集统一采用**单桶有界LRU**：特征及其索引合计≤54GB、盘点/review/日志≤3GB、预留3GB，总≤60GB；解码视频仅内存，不另落盘像素/视频或窗口缓存，90%阈值停止扩张。缓存键含源camera曝光ID、E*/pool/预处理hash，按需读取；未命中重算，不能靠移动世界时距来提高命中。54GB只是约13.56h的30Hz纯payload等价容量，扣索引后更少；因此三份子集**都可在≤60GB容量内运行，但不保证只需一次覆盖的GPU时间**。任务轮转+episode局部性可减少反复解码，同时保留任务/episode均衡。
仅按均匀稀疏缓存、忽略时间合同，S1@8Hz约53.57GB、S2@4Hz约53.21GB、S3@2Hz约39.86GB。S3@3Hz约59.79GB，几乎无索引余量；按54GB预算，3Hz最大135.63h。8/4/2Hz的最大观测age约125/250/500ms，均不能直接满足100ms和原30Hz精度；10Hz目标最近误差最多50ms也大于30Hz的约16.7ms，且历史0.2秒需原生帧。所以不把稀疏持久缓存率偷换为世界头时间单位；保留原曝光查询与LRU是本轮推荐。

### 4.3 解码、ZED-X可选扩展与混合
抽样单路中段1秒解码+探流的CPU墙钟RS中位0.305s/P950.567s，ZED中位0.904s/P951.396s，均含短段预卷及ffprobe，不含完整训练管线；均值约2.89倍。建议未来RS以2–4个episode解码worker、每ffmpeg1–2线程起步，测IO与GPU等待后再调，避免多进程×多线程挤占共享CPU。原生60Hz须解码更多源帧才能取30Hz，固定218吞吐不能保证，不能把25%余量当正式GPU配额。
ZED顶视H.265、腕H.264，采样探得1920×1200；三视角像素量为RS640的7.5倍；若再取顶右立体成为4路，像素总量约10倍、相机帧/feature容量增4/3，但不是可直接获得的Piper收益。CPU HEVC随机访问需IDR/CRA和预卷；本轮三视角短段探流总成本约RS的2.9倍，不代表GPU端相同倍数。未来每加入10h ZED三视角，30Hz payload另需39.81GB（或同一LRU内挤出其他源）；沿218+25%底线约5.16GPUh，若解码使有效吞吐降为1/2–1/3，则约10.32–15.48GPUh，属于情景估算非实测。默认ZED不纳入S1/S2/S3，97.00h只作可选候选列示。
建议P1窗口分配：Piper≥60%，本地yam≤20%，ABC≤20%（ABC绝对上限25%，外域合计≤40%）；缺本地yam时回拨Piper，不因ABC小时多按时长淹没本域。这里的本地yam指E87合规allowlist，不重新打开其dev/test；ABC内部先类/任务、再episode均衡，单任务采样及物理时长均≤25%。来源显式记录SRC_PIPER/SRC_YAM_LOCAL/SRC_ABC，不喂session/operator/date作为提示；P1文字所有源统一“Manipulate objects on a tabletop.”，不靠标注文风识别来源。原始子任务文本单独保存，只作未来经清洗的训练目标/阶段辅助候选，不作当前阶段oracle或测试选型信号。

### 6.2 风险与仍需决定的事项

|风险|本轮结果与边界|
|---|---|
|域差异|YAM夹爪/背景/视角与Piper不同；同为640不等于同内外参，领带/袋结不等同穿塑料锁头；ABC小比例起步|
|站间差异|RS也有848/1280/800分辨率，已排除；ZED立体/HEVC另列，未用右目当额外默认输入|
|人脸/隐私|已显示联系表无人脸/人体；未逐帧认证的帧仍unknown，不公开视频/联系表，不用人脸识别模型|
|数据更新/同源|本地报告与任务目录有2219条额外未核别名；独立annotation会更新，必须锁本地身份哈希/mtime，首尾摘要无法查出所有中部变化|
|文本质量|来源作者未知、粒度不一、零长段及mistake/reset；不能作为结果/成功真值或checkpoint依据|
|时钟|log_time=payload不提供独立接收时钟；本轮窗口率是记录时基条件估计，不冒称部署在线验收|
|许可|本地缺LICENSE原件，官方声明Apache2.0与其官方标准条款已核；实际revision/NOTICE/gated条件未闭合，未宣称可公开再分发|
|解码/磁盘|不转码/复制视频，不用GPU；60GB必须有界，Lru miss和60Hz/HEVC可能增算力；/mnt99%约1.4TB为共享剩余非项目额度|

默认建议：采用S1作为下一轮审阅起点、继续排除ZED、保留30Hz世界合同+有界LRU、ABC初始20%采样。需要用户决定的是后续是否采用这套缓存/混合策略，以及是否另开ZED或改10Hz时间合同的版本；本轮不启动训练或提特征。缺失独立接收时钟、标注作者链、本地LICENSE/NOTICE原件无法由当前文件补出，均保留unknown；完整选中视频解码/逐帧隐私审查不在本次抽样验收覆盖内，清单不是P1_ready。

### 7. 交付、命令与验证
产物根T/data/abc_inventory_v1：task_table.csv、classification.json、task_audit_summary.csv、episode_index.jsonl、calibration_index.jsonl、sample_audit.jsonl、selected_timing.jsonl、proposed_subsets/S1S2S3.json及S1/S2/S3.json、review/、env/、scripts/、logs/、state/。初期结果与抽段假零帧版本已保留，使用顶层最终清单。

主要命令（在T执行，完整脚本/日志与阶段状态可断点续跑）：
```sh
python3 -m venv data/abc_inventory_v1/env/venv
data/abc_inventory_v1/env/venv/bin/pip install --disable-pip-version-check --report data/abc_inventory_v1/env/install_report.json mcap mcap-protobuf-support
# 下列长任务实际以 setsid nohup ... > logs/... 2>&1 < /dev/null 分离运行
python scripts/scan.py scan
python scripts/audit.py
python scripts/calibration.py
python scripts/refine.py
python scripts/subset_logic.py timing
python scripts/finalize.py
```
上面scripts为data/abc_inventory_v1/scripts的缩写；实际命令使用新venv绝对/项目相对路径，未调用e29、未改系统环境。已检查包依赖、真实MCAP读取/CPU片段解码、源UUID、子集嵌套/唯一性/时长占比、派生SHA256、采样数量、磁盘字节数。只对自己的worker做停止/断点重跑；无他人进程操作。GPU0GPUh，无权重/数据集下载，无正式训练、无特征或窗口缓存，Piper dev/test/sealed_test未打开、未改任何E82会话文件。日志与worklog追加E89-A条目。
