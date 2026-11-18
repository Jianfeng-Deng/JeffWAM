## C 历史“优于/劣于复制”全量检索审计

检索指定 `docs/experiment_log/EXPERIMENT_LOG.md` 与 notes 顶层全部 Markdown：133 文件、73 文件命中、314 行命中（含方案、文件复制等非结果文本）。完整路径、行号、命中文本保存在 `results/e93_dev_diag_20261114_01/history_search_hits.json`。本节对实际结果去重；未运行方案不当作实验收益。旧 SAM/DINO 自动掩码、模型潜变量指标仅按历史代理指标记录，本轮未用它们作测试真值、阈值或 checkpoint 选型。未运行旧模型、仿真或天机训练。

表中“同域”只指同一任务/采集域的 episode 留出，不代表独立场景或未反复查看；“未记”表示已读来源未记录，绝不补猜。除最后的 Piper 行，下面真实数据实验均为天机瓶子/箱子任务，与 Piper 穿扎带不同。

|实验/域|编码器与目标|训练损失|指标、归一化及复制比较|时域|训练长度|动作条件；train/dev|
|---|---|---|---|---|---|---|
|早期 V-JEPA 历史/天机|V-JEPA 潜变量，旧报告摘要|JEPA，详细配置未核实|历史总结约优于 persistence 0.28%、优于无动作0.04%；弱收益，非 E90 指标|未记|未记|有；同域旧留出，来源摘要 `notes/2026-11-03-root-cause-astra.md`|
|E01 R0/R1/R2/R3/天机|upstream keypoint latent；R1 frozen drawer encoder|LeWAM、SIGReg；R2 IDM .1|日志只记 R2 world gain约15%，明细/归一化未记；动作 .147rad仍劣hold|未记|未记|有；同域验证2289窗|
|E02/历史模拟域（仅文本审计）|LeWAM 192D object latent|latent预测|gain persistence scratch83.1%、ftreal52.4%、ftfrozen64.1%；不可作真实 Piper 证据|未记|未记|有；5fold；本轮没有使用模拟数据|
|E03 PhaseA/天机|ResNet18-GN 12×20空间，未来RGB|change-weighted image + policy MSE，latent/image-only|日志记未来变化区比copy好49%，错动作丢约85%增益；定义细节未记|原run短时，具体h未记|配置30epoch；image_only末日志epoch23/17273步、latent_image epoch19/14269步；policy峰值5–6|有；同域留出|
|E18 RunE+RegionJiT/天机|DINOv2-B runE latent→JiT RGB→DINO聚类区域|旧世界图像/潜损失+独立JiT FM|moving瓶质心1.237 vs copy1.615patch、手1.209 vs2.266；胜copy70.3%/83.9%。全窗瓶IoU .372<.411，手 .661>.504|+10|世界终点未记；JiT20000步|有；256同域验证窗；聚类标签代理|
|E21 对E18重评分/天机|同上；SAM3分别标真实/生成图|只重评|瓶误差约259 vs copy20px（moving），被判标注域差失效；手moving IoU .604>.389，质心26.73<27.36px|+10|沿用旧权重|有；同域验证；瓶结果无效，手仍是自动标签代理|
|E24/天机|DINO+object latent/mask/RGB|latent rollout、图像、对象、policy/IDM|瓶31px>copy11–23；手82–86>45–79；world劣copy；动作门失败|+10/20/30|25epoch/17475更新（config与日志）|有；93集训练/留出同域|
|E24r/天机|同E24，clean手FK标签重评|不更新权重|当前定位已有约140px误差；moving仅9–25%窗胜copy，copy10–41px|+10/20/30|沿用E24|有；同域2109验证窗|
|E25/天机|DINO，clean手+对象+global latent|重配world/rollout/image等损失|手moving55–70%窗胜copy；瓶约30>11–23px；E26口径global误差/copy1.287→2.378|+10/20/30；H1–4|12epoch/8388更新（training_report.json）|有；同域验证，标签清理与结构改动混杂|
|E26a E24审计/天机|同固定E24，clean对象token|不更新E24；另无动作MLP|raw token误差逐窗除copy再episode平均：global1.28→2.45，瓶2.82→5.42，均劣；MLP global .978→.982略好|H1–4（每跳10）|E24沿用；MLP512训练窗8epoch|E24有、MLP无；2019窗留出同域|
|E27a A0/A1/A2/A3/A4/天机|DINO对象/global latent；A4残差|A1重配、A2对象加权RGB、A3无未来图像、A4残差|global误差/copy H1→H4：A0 1.279→2.401，A1 1.231→2.241，A2 1.225→2.279，A3 1.224→2.258，A4 1.152→1.912，全部劣；手部分像素定位胜、瓶/箱劣|+10/20/30与H1–4|各从E25微调4epoch|有；同域2019H4有效验证窗|
|E28 full/nogate/nofk/天机|DINO+FK/对象，latent/mask|JEPA/对象/图像，消融FK与gate|瓶质心约20.3px>copy13.2–14.0；旧总结称cosine优于copy，不代表raw-L2或瓶定位胜；原生value门未测|+10/20/30|从A2初始化，各4epoch|有；同域验证；旧bridge value结果不能冒充原生结果|
|E29-S4/天机→DROID教师域差|冻结V-JEPA2-AC 16×16×1408，独立中心读出|仅真实特征读出热图头|squash AR moving瓶31.60>copy22.40px；左手11.39>10.71、右手14.82<16.26；TF手改善较稳，AR手对copy CI跨0|1–4token步（原始每4帧）|世界教师冻结；读出180epoch|有；84训练/9留出同域输入，教师预训练异域|
|E32 student 4epoch/天机|DINOv2最后4block，空间latent+RGB|JEPA+rec+flow|日志称动作所有步劣hold；没有据此证明世界劣copy|1–32动作；视觉+16/32|4epoch|视觉真实示教动作；同域84/9集|
|E32 student无教师/天机|同上，顶视192×320，240token|cos+ .1 LN-L2 JEPA、对象加权rec、flow|RGB PSNR19.22/18.52>copy18.91/17.90；对象MSE .0271/.0327<.0319/.0416；确有像素收益|+16/+32|40epoch|有；2812训练/266留出窗、9episode；同域|
|E32 student有教师/天机|上行+OpenWAM Alpha distill|上行+teacher|日志称每项略劣无教师，未记精确copy增益；教师瓶读出33px>copy10px，不是世界预测比较|+16/+32|40epoch|有；同上；单seed|
|E33 B/天机|DINOv2→共享256D×240token，EMA target，RGB/mask|前景JEPA(cos+.1LN-L2)1、flow1、rec .25、IDM .1、mask .5、var .01|最终JSON PSNR20.352/19.938>18.913/17.896；legacy对象MSE .020505/.023580<.031853/.041639；瓶IoU .437/.408<copy .673/.576，箱 .759/.742<.890/.850；右手胜，左手+16劣/+32胜|+16/+32|80epoch|视觉输入真实未来示教动作；同域80fit/4dev/9eval；像素胜不等于对象身份/位置胜|
|E33 C/天机|同B，在线stopgrad target，无EMA|同B，SIGReg替换EMA/var|PSNR20.078/19.575>copy；对象MSE .022566/.025919<copy；瓶/箱IoU仍劣copy|+16/+32|80epoch|有；同B|
|E33 D/天机|同B，动作头不读未来token|同B，世界仍训练|PSNR20.467/20.110>copy；对象MSE .020153/.022479<copy；瓶/箱IoU仍劣copy|+16/+32|80epoch|世界有；同B|
|E33sub R1/R2/天机|全240×256残差子目标，无rec|子目标FM/JEPA相关损失，详细权重见旧run|绝对未来特征cos R1 .960<copy .966，R2 .972>copy；动作generated1.169/1.164劣B1.160；用户已终止|+16/+32|60epoch|有；266窗同域留出；绝对特征cos非残差方向cos|
|E34 R0-AC/S/NOZ/天机|E33-B冻结特征+mask辅助初始瓶状态→显式2D状态|状态/模式监督；详细权重见config|moving rollout AC比copy好约38–45%，比无动作S好19–25%；h10仅14窗、h20约30窗，未达≥50门槛|+5/10/20|80epoch，AC seed34=1680更新|AC有，S无；3seed、同域留出；mask-assisted不等于RGB-only|
|E36b/天机|在E33上做动作/世界错配条件修复|条件FM相关，旧报告未完整定位|指定日志/notes仅有实现与采样修复，未检到独立世界胜copy数值；不列为已证实世界收益|+16/+32|预算746步上限，不能替代实际完成值|有；同域；动作hold结果不作世界copy证据|
|E47/B-refit/天机|官方DINO细网格960token，RGB/mask与direct/rollout|JEPA/LN-L2+cos、RGB/对象、rollout、IDM、flow|seed47 common网格h16/32瓶 .521/.489<copy .667/.559；手h32 .527/.570> .337/.225；native h32瓶 .618> .586、箱 .871>.860（不同时网格结果不可混）；RGB/object多数门通过，mask总门未过；rollout24/24门过|+4/8/16/32|两seed各80epoch/13440更新|有；80fit/4dev/9test同域；标签为旧自动mask代理|
|E58 h10/天机|冻结E33-B，复制预测器改时域与前三动作token|JEPA|h10 JEPA .0364< persistence .0513（skill29.0%），E33h16 .0399只是跨时域参照；hold动作输入OOD loss .0907|+10|已读日志未记步数|有；78训练/6内部dev，冻结eval不训练|
|X1/天机|冻结DINO 3×5格随机投影8D、线性ridge，质心目标|ridge平方误差|notes主要报告历史相对单帧，不提供可核对copy总收益；不把历史有益写成世界胜copy|+4/8/16|CPU闭式拟合|有/无对照；两组64/16同域内部切分|
|架构五组1–5/天机|DINO dense，reader共享/本体史/手ROI/视觉史逐项增量|JEPA+mask+flow/IDM/rollout|h32 moving瓶IoU−copy +.149/.107/.129/.133/.123，配对CI均>0；static −.040/−.038/−.019/−.020/−.038，多数劣。copy为当前latent经各自decoder，不是当前真实mask|+4/8/16/32|best epoch23/23/21/23/23，结束步数不一致|有；64fit/16内部holdout，已用于选ckpt，探索性|
|E72 W0/天机|冻结SigLIP2，原生4×4+global共17token768D，绝对目标由delta恢复|视觉delta MSE+训练归一化q MSE|dev视觉MSE .08444/.11331/.16109 vs copy .08251/.11048/.16983：短时劣、+32好5.15%；holdout .08544/.11518/.14116 vs .08574/.12232/.16417：好.35/5.83/14.02%|+8/16/32|5400更新 batch16|有真实未来动作+当前q/dq/历史8命令；64fit/4dev/16holdout/9旧冻结，同域|
|E72 W1/天机|同W0|W0+.1转移/语言事件回归|dev .08299/.11241/.16120：短时仍劣copy、+32好5.09%；holdout .08450/.11399/.14105 vs同copy：好1.45/6.81/14.08%|+8/16/32|5400更新|同W0；生成训练标注不是测试真值|
|E73/天机|冻结SigLIP2原生196patch，解码RGB，q后果|主world特征/状态损失+RGB梯度范数5%辅助，非单纯E90 FM|+32 dev MSE1.72608<copy1.77540（2.78%）；holdout1.63412<1.71795（4.88%）；historical1.76429<1.86073（5.18%）。旧诊断：+8真实未来latent解码仍比复制RGB差约43%，表示/解码有瓶颈|+8/16/32（引用数值+32）|decoder1500、flow5400、world5400|有真实动作、q/dq/命令史；与E72同域划分；196patch MSE尺度不可与17token/E90互比|
|E74/天机|D0沿用P0策略候选，联合世界主实验未运行|无新world训练|4样本动作均值 .470357优于oracle完整单候选 .474732；只反映动作池，不是世界赢/输复制；不得列联合世界收益|动作1–32|D0无训练|无已完成联合world；dev4episode|
|E86/ Piper|冻结SigLIP2、2×4视觉与动作flow|多模态FM烟雾|记录的是动作未胜hold（夹爪约15–16mm vs1–4mm）；不是完成的世界胜copy证据|世界+10/25/50，动作50|烟雾与100步profile，未正式|P3有动作；6parent/48窗同域dev|
|E88 smoke02 / Piper|SigLIP2残差W，原768→256噪声瓶颈|FM速度损失|固定训练参考6时域skill约−.67至−1.35；满维噪声旁路修正后另run保留旧结果|+10/25/50/100/200/300|烟雾，非收敛训练|W-only无动作；训练子集，不是dev泛化|
|E88 smoke03 / Piper|修正干净残差头，v=(rhat−w)/(1−t)|FM+.1短cos+.1adapter+链|P3_long终点dev skill−10.68/−11.16/−9.88/−2.17%；长域稀疏；未胜copy|+10/25/50/100|P1 160、各P3分支192更新|W-only无动作；同原6parent dev|
|E90 ckpt-1 / Piper|SigLIP2三相机2×4，100每相机global；训练RMS归一化残差|FM+.1短cos+.1adapter+计划链|v1 CPU4Euler单样本：−5.81/−5.54/−5.24/−20.51%，前3CI<0；100仅1窗|+10/25/50/100|5000更新固定终点|无动作/状态；242train parent，6dev缓存，属于E93直接复现对象|

历史主证据路径：`runs/e27a_chain_20261103/summary.md`、`runs/e29_vjepa_ac_probe_20261103/s4/REPORT.md`、`runs/e33_20261104/{B,C,D}/eval_final/ep80/student_summary.json`、`notes/2026-11-04-e33-contract.md`、`runs/e34_20261104/eval_all/G1_G4.json`、`runs/e47_p0p1_20261104/GATES_{SEED47,TWO_SEED}_REPORT.md`、`notes/arch_five_group_analysis_20261110.md`、`runs/e72_transition_jepa_20261111/{E72_REPORT,PROTOCOL}.md`、`runs/e73_causal_wam_20261111/{E73_REPORT,PROTOCOL_FORMAL}.md`、`runs/e74_joint_v2_20261112/E74_REPORT.md`。所有路径相对T。E33早期 `report/compare.md` 仍写“B/C/D不存在”，为过时中间产物；本审计优先使用真实 final ep80 JSON，不能依据该旧表说E33没做。

### C 可比性判断

用户记得的旧结构优势有真实记录支持，尤其 E32/E33 的未来RGB和E72的中长时潜变量；同时“旧结构始终胜复制”不成立。E33瓶/箱mask经常仍输当前真实mask复制；E27a的raw latent大幅输copy；five-group moving胜而static劣。不同指标可以给相反判断。

E88修正后与E90使用相同Piper数据、冻结SigLIP2/归一化残差，最接近直接比较，但训练阶段、步数、GPU/CPU RNG及查询布局仍须匹配，不能将表中差异全归因于结构。E90原CPU v1与E93原48窗复现才是严格直接比较。E72/E73只是同编码器家族：机器人/任务、视角数、动作与本体条件、token池化、归一化、时域均不同；E33则连编码器、目标、像素解码损失和动作条件都变了。它们证明“某些预测目标在某些条件可胜copy”，不能证明本次无动作Piper目标应达到相同skill，也不能替E90负结果辩护。
