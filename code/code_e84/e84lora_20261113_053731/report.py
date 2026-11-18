import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import sys
sys.dont_write_bytecode=True
from common import *
from contract import sha
import numpy as np,collections,shutil

def load(p):return json.loads(pathlib.Path(p).read_text())
def fmt(x,pct=False,d=2):
 if x['estimate'] is None:return 'N/A'
 scale=100 if pct else 1
 return f"{x['estimate']*scale:.{d}f} [{x['ci95'][0]*scale:.{d}f}, {x['ci95'][1]*scale:.{d}f}]"
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('|'+ '|'.join(map(str,r))+'|\n' for r in rows)
def make():
 j=load(P/'paired_comparisons.json');dr=load(P/'drift_results.json');ready=load(L/'READY_training_data.json');progress=load(P/'progress.json');roots={'冻结':OLDP/'frozen','LoRA':P/'lora','打乱LoRA':P/'shuffled'};s=[]
 s.append(f'# E84 LoRA对比微调验证\n\nRun `{RUN}`；BRT {brt()}。继承 `{OLD}` 的selection_full_v4（SHA256 `{sha(P/"selection_full_v4.json")}`）与锁定parent划分。用户已授权本轮；未训练WAM、不合并/替换原权重。以下均为parent等权估计及2000次parent bootstrap 95%CI，探索性多指标未校正多重比较；固定seed20261112。')
 conclusion=(C/'conclusion.txt').read_text() if (C/'conclusion.txt').exists() else '**结论待最终审阅；下表是已完成的数值，不是验收决定。**'
 s.append(conclusion)
 s.append(f'严格配对修正：先导run `{PILOT}` 的331条E83正配对有320条近邻锚点，为严格遵守不平移标签，先导结果仅保留诊断，不作为本报告正式证据；停止自己的未完成任务后，新run仅保留11条同锚点caption，FK1483对、原selection/split与所有真值均不变。E83每8步一次，避免11条文本过度重复，固定768步，不按任何val指标选checkpoint。此前调试/训练/复提{PRIOR_LORA_GPU_HOURS:.6f} GPUh全部计入同一4GPUh预算。11条E83覆盖不足限制对完整阶段一方案的外推。')
 s.append('沿用已完成的本地型号审计：SigLIP2 base-patch16-224、375187970参数、视觉/文本双塔、196×768 patch、文本64token，不支持NaFlex。没有下载。预处理保完整FOV、保比例bilinear缩到224再edge-pad，保存仿射；这与官方直接方形缩放有差异，所有冻结/LoRA比较均沿用同一锁定合同。')
 s.append(f'## 预注册与训练\n\n视觉与文本双塔各最后4层Q/V，rank8、196,608可训练参数，lr1e-5，AdamW weight_decay0.01、gradient clip1；batch4（每8步一次3个FK+1个E83，其余为4个FK对），每对4张224图。固定768步，各variant同初始化seed、同样本顺序、同优化次数，只保存最终adapter。FK：top+该臂腕视，起点与H50终点归一化差分，48类含左右臂，中英文候选分别CE；另加方向反转、换臂难负样本0.25。E83：既有短caption与三视角平均，权重0.5，左右/颜色文本替换负例；64token截断，未生成新语言。HSV橙色patch弱训练目标与蓝/绿负例，权重0.25；不是实例像素真值。冻结图像与文本余弦偏离约束0.1。\n\n训练库FK {ready["fk_pairs"]}对、E83 {ready["e83_pairs"]}对、共{ready["parents"]}个probe-train parent；每parent每臂最多4对FK、E83仅保留原标注source_frame_index与锁定anchor_source_row完全相同的11条；固定hash，不看val选样。训练样本是锁定selection的子集，复提与评估仍用全部22,230时刻（train11,376/val10,854，193/49parent），保留三视角、四历史与H10/H50终点。实验2终点图只表示已发生运动；实验3绝不输入终点。\n\n打乱LoRA：FK标签在FK训练行间固定置换，E83 caption在E83训练行间固定置换，训练步数与初始化完全相同。橙色HSV正例全为橙色，样本置换不能改变其标签，所以该项在两组一致；颜色对照不能排除共享弱监督效应，缺少蓝/绿正例。')
 dist=load(P/'training_distribution.json');s.append('训练子集分层：'+str(dist['distribution'])+'；48个FK类别均有训练例，最少7对；E83训练caption完整唯一句11。各分层parent可能交叉，不可直接加总；去重共189parent，详见training_parent_deduplication.json。按parent均衡并不等于按方向类别均衡，native/early训练parent仅9/8，外推受限。')
 rr=[]
 for v in ['lora','shuffled']:
  a=load(L/v/'training_result.json');last=a['logs'][-1];rr.append([v,a['steps'],a['config']['trainable_parameters'],f'{last["loss"]:.4f}',f'{last["visual_cosine_deviation"]:.4f}',f'{last["text_cosine_deviation"]:.4f}',a['adapter_sha256'][:16]])
 s.append(table(['组','优化步','参数','末步loss','末步视觉偏离','末步文本偏离','adapter SHA前缀'],rr))
 s.append('## 三类对齐 × 层级\n\n冻结零样本、冻结小头与LoRA均在同一probe-val评估。LoRA列分别给原生时序差分检索和重新训练的同结构小头；编码器微调不能替代阶段二动作分支。画面像素运动仍为**无真值，未评估**。')
 def image_auc(root):
  return next(r['auroc'] for r in load(root/'pixel_results.json')['summaries'] if r['split']=='probe-val' and r['quality']=='main' and r['radius_multiplier']==1 and r['condition']=='correct' and r['word']=='orange zip tie')
 def pair_motion(root,head=None):
  ms=[]
  for arm in ['left','right']:
   a=load(root/f'motion_robot_{arm}_h50.json');m=a['zero_shot']['zh']['difference'] if head is None else a['learned'][head]['metrics'];ms.append(fmt(m['direction_top1'],True))
  return ' / '.join(ms)
 def pair_behavior(root):return ' / '.join(fmt(load(root/f'behavior_{arm}.json')['counterfactual']['language']['follow_rate'],True) for arm in ['left','right'])
 s.append(table(['对齐','冻结零样本','冻结特征小头','正式LoRA'],[
 ['语言↔图像','橙带AUROC '+fmt(image_auc(OLDP/'frozen'),False,3),'未训练像素头','橙带AUROC '+fmt(image_auc(P/'lora'),False,3)+'；左右仍不可判定'],
 ['语言↔机器人运动（H50方向%,左/右）',pair_motion(OLDP/'frozen'),pair_motion(OLDP/'frozen','contrastive_zh'),'原生 '+pair_motion(P/'lora')+'；同结构头 '+pair_motion(P/'lora','contrastive_zh')],
 ['语言↔动作（反事实跟随%,左/右）','编码器无动作输出',pair_behavior(OLDP/'frozen'),pair_behavior(P/'lora')],
 ['语言↔画面像素运动','无真值，未评估','无真值，未评估','无真值，未评估']]))
 rows=[]
 for arm in ['left','right']:
  for h in [50,10]:
   for label,root in roots.items():
    a=load(root/f'motion_robot_{arm}_h{h}.json');m=a['zero_shot']['zh']['difference'];rows.append([arm,h,label,fmt(m['top1'],True),fmt(m['top3'],True),fmt(m['direction_top1'],True),fmt(m['amplitude_top1'],True),fmt(m['direction_angle_error_deg'])])
 s.append('## 实验2：机器人系FK方向/幅度零样本\n\n两臂自身基座XY，8方向×3幅度；随机24类top1/top3=4.17/12.5%，方向12.5%，幅度33.3%。')
 s.append(table(['臂','H','编码器','24类top1 %','top3 %','方向top1 %','幅度top1 %','扇区角误差°'],rows))
 rows=[]
 for arm in ['left','right']:
  for label,root in roots.items():
   a=load(root/f'motion_robot_{arm}_h50.json')
   for kind in ['linear','contrastive_zh']:
    m=a['learned'][kind];z=m['metrics'];rows.append([arm,label,kind,m['trainable_parameters'],fmt(z['top1'],True),fmt(z['direction_top1'],True),fmt(z['top3'],True),fmt(z['direction_angle_error_deg'])])
  a=load(P/'matched_frozen'/f'motion_robot_{arm}_h50.json');z=a['metrics']['true'];rows.append([arm,'冻结','matched contrastive',a['trainable_parameters'],fmt(z['top1'],True),fmt(z['direction_top1'],True),fmt(z['top3'],True),fmt(z['direction_angle_error_deg'])])
 s.append('### 冻结小头与适配后同结构小头（H50）\n\n原线性头76,043参数，原对比头738,144参数，并不与LoRA严格同参数量。新增192,793参数小头（hidden25），比LoRA少1.94%，同原小头40epoch/lr1e-3。该基线只训练头，LoRA原生零样本只有196,608个可训练参数；LoRA+头合计另加头参数。各头完整probe-train，LoRA微调仅上述子集；参数量近似匹配，样本曝光和训练计算不匹配。')
 s.append(table(['臂','编码器','头','头参数','24类top1 %','方向top1 %','top3 %','角误差°'],rows))
 rows=[]
 for arm,results in j['motion'].items():
  for pair,z in results.items():rows.append([arm,pair,fmt(z['top1'],True),fmt(z['direction_top1'],True),fmt(z['amplitude_top1'],True),fmt(z['top3'],True),fmt(z['angle_error_reduction_deg'])])
 s.append('### 成对增益（百分点）\n\nCI下界>0才记为有正向证据；与零相交为无明确证据，CI上界<0为退化。')
 s.append(table(['臂/H','比较','24类top1差','方向top1差','幅度top1差','top3差','角误差改善°'],rows))
 s.append('## 实验3：语言↔动作位移\n\n同观测更换8方向、固定medium；未来FK句仅作为给定条件，输出该臂H50的3D位移。只用当前/历史图和当前14D状态，60固定epoch，标准化仅train。属于受控映射，不是未来预测、视觉grounding或实机控制证明。')
 rows=[]
 for arm in ['left','right']:
  for label,root in roots.items():
   a=load(root/f'behavior_{arm}.json')
   for condition in ['language','no_language','shuffled_language_training']:
    m=a['counterfactual'][condition];rows.append([arm,label,condition,fmt(m['follow_rate'],True),fmt(m['prediction_instruction_cosine'],False,3),fmt(m['prediction_true_cosine'],False,3),fmt(a['conditions'][condition]['displacement_L2_error_m'],False,4)])
 s.append(table(['臂','编码器','条件','跟随率 %','预测↔请求XY余弦','预测↔真实XY余弦','事实L2 m'],rows))
 rows=[]
 for arm,a in j['behavior'].items():
  for pair,z in a.items():rows.append([arm,pair,fmt(z['follow_rate'],True),fmt(z['prediction_instruction_cosine'],False,3)])
 s.append(table(['臂','比较','跟随率差 pp','请求余弦差'],rows))
 s.append('## 实验1：150帧近似像素诊断\n\n仅40帧/32parent为probe-val；110训练帧另存诊断，未用于像素训练或checkpoint选择。主分析排除low与ambiguous/不可见；low单独分层。整臂只能用夹爪点代理，TIE为紧框代理，其他为点圆邻域；不是mask真值。主半径沿用r=max(3×分层重复差P90,一个patch原图边长)，demo4/early45.71px、native91.43px，同时报2r。')
 s.append('Patch文本图沿用原方法：原MAP以学习query对整组patch做attention，全局attention权重本身与文本无关；singleton-MAP对单patch时attention权重恒1，等价out_proj(V(patch))加残差MLP，再归一化与文本余弦。LoRA只改双塔Q/V，MAP头保持原样。patch含全图上下文，逐patch套全局池化头偏离训练分布，余弦不是像素分割概率。14×14图固定双线性插值到224，按保存仿射映回原图，不从标签拟合。')
 rows=[]
 for label,root in roots.items():
  p=load(root/'pixel_results.json')
  for a in p['summaries']:
   if a['split']=='probe-val' and a['quality']=='main' and a['condition']=='correct':rows.append([a['word'],label,a['radius_multiplier'],f'{a["frames"]}/{a["parents"]}',fmt(a['pointing'],True),fmt(a['auroc'],False,3)])
 s.append(table(['查询','编码器','半径倍数','帧/parent','pointing %','AUROC'],rows))
 rows=[]
 for label,root in roots.items():
  for a in load(root/'pixel_results.json')['summaries']:
   if a['split']=='probe-val' and a['quality']=='main' and a['radius_multiplier']==1 and a['condition']!='correct':rows.append([a['word'],label,a['condition'],fmt(a['correct_minus_control_pointing'],True),fmt(a['correct_minus_control_auroc'],False,3)])
 s.append('### 左右/颜色/随机文本对照（r）\n\n正值代表正确文本更好，均为成对parent CI。')
 s.append(table(['查询','编码器','对照','pointing差 pp','AUROC差'],rows))
 rows=[]
 for label,root in roots.items():
  for a in load(root/'pixel_results.json')['left_right_switch']:
   if a['split']=='probe-val' and a['scope']=='gripper':rows.append([label,a['radius_multiplier'],a['disjoint_targets'],f'{a["frames"]}/{a["parents"]}',fmt(a['both_queries_follow'],True),fmt(a['argmax_changed'],True),fmt(a['correct_minus_opposite'],True)])
 s.append('### 左右随文本切换\n\nargmax改变不等于正确身份跟随；分离目标r子集仅7parent，2r仅2parent。经验bootstrap[0,0]不代表总体确定为零。左右可泛化结论仍为**不可判定**。')
 s.append(table(['编码器','半径倍数','两圆分离','帧/parent','双句均跟随 %','argmax改变 %','正确−反向命中差 pp'],rows))
 s.append('正式LoRA在r且两圆分离子集为0/7双句成功；在独立二项假设下，Clopper–Pearson双侧95%区间上界约40.96%，比经验bootstrap退化区间更能显示小样本不确定性，仍不包含共享标注偏差。')
 rows=[]
 for word,a in j['pixel'].items():
  for radius,b in a.items():
   for pair,z in b.items():rows.append([word,radius,pair,fmt(z['pointing'],True),fmt(z['auroc'],False,3)])
 s.append('### LoRA像素指标增益（成对CI）\n'+table(['查询','半径倍数','比较','pointing差 pp','AUROC差'],rows))
 rows=[]
 for word,a in j['semantic_contrast_gain'].items():
  for control,z in a.items():rows.append([word,control,fmt(z['pointing'],True),fmt(z['auroc'],False,3)])
 s.append('### 语义区分增益（LoRA正确−对照，再减冻结正确−对照）\n'+table(['查询','对照','pointing语义差增益 pp','AUROC语义差增益'],rows))
 rows=[]
 for word,a in j['semantic_contrast_true_minus_shuffled'].items():
  for control,z in a.items():rows.append([word,control,fmt(z['pointing'],True),fmt(z['auroc'],False,4)])
 s.append('### 语义区分差：真实文本LoRA−打乱文本LoRA\n\n两组共享恒定橙色HSV正例，以下只反映FK/E83标签正确性引起的额外差异。\n'+table(['查询','对照','pointing差 pp','AUROC差'],rows))
 s.append('## 原编码偏离与遗忘\n\n固定hash每capture一个锚点，比较所有有效当前/历史global；CKA为线性CKA点估计，余弦给parent CI。文本为预定中英文FK候选与对象/负例109句。未提供通用图文检索小样本，因此通用检索退化**未评估**；无下载，Piper域内漂移不能证明通用能力保留。')
 rows=[]
 for v,a in dr['results'].items():
  for split,z in a['visual'].items():rows.append([v,split,z['instances'],fmt(z['cosine'],False,4),f'{z["linear_CKA"]:.4f}',f'{a["text_cosine"]:.4f}',f'{a["text_cosine_min"]:.4f}',f'{a["text_linear_CKA"]:.4f}'])
 s.append(table(['组','划分','视觉实例','视觉余弦','视觉CKA','文本均值余弦','文本最小余弦','文本CKA'],rows))
 s.append('## 像素真值局限与结论降级\n\n同模型近似点位150帧/119parent，三遍重复差中位2.83px/P90 5.39px，native两遍P90 15.90px；重复性不是独立绝对误差。demo4/early/native标定验证中位26.90/9.07/18.97px，均NO-GO，未将FK投影用作像素真值。无相邻源帧对、无≤1s标注对，画面运动无真值未评估。可信可见目标总数（train+val）L72/R72/TIE110/HEAD98/TIP35；可见low分别7/17/4/2/1，TIE/HEAD/TIP ambiguous36/35/36，逐帧分层继承旧审计。左右、锁头/尖端与细穿带结论统计力有限。机器人FK不依赖像素投影，但URDF/反馈、短窗口图像年龄、单任务相关性、少模板与训练顺序均限制外推。一次固定seed不包含优化随机性CI；bootstrap只反映留出parent抽样，不修正模型标注共有误差。')
 s.append('## E83语言信息量（继承已完成审计）\n\nE83-A任务指令226条/156完整唯一句；B94条/8句，B最短variant仅1句；总task320条/164句/243parent，实际64token输入161种。全部长task超过64token。FK435,404条/18,393完整句，去数字471句；受控方向模板每臂仅24类。A/B有77个parent重叠，不可视为独立。FK短局部运动条件比E83-B任务文本包含更多可控变化；E83提供任务/可见语义补充，长文需因果裁短，部署不能自动输入未来FK。scene_description_en另有320条/320唯一/243parent；FK覆盖243parent。原逐句parent映射和SHA仍在旧run e83_language_information_afc37bf06a7070ab.json与e84p_20261112_153338/fk_information.json，不重做审计；本run11条caption的逐句parent/锚点对应另存training_distribution.json和e83_training_pair_provenance.json。')
 size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())+sum(p.stat().st_size for p in (T/'code_e84').rglob('*') if p.is_file())+sum(p.stat().st_size for p in (T/'notes').glob('e84*') if p.is_file());assert size<30_000_000_000,size
 ledger=[load(x) for x in (P/'gpu_ledger').glob('*.json')];hours=PRIOR_LORA_GPU_HOURS+sum(x['elapsed_s'] for x in ledger)/3600;assert hours<=4 and all(x['status']=='finished' for x in ledger)
 s.append(f'## 资源、交付与续跑\n\n本LoRA阶段含先导作废/调试/训练/两组全局复提{hours:.4f} GPUh/4，E84累计{1.276657608018981+hours:.4f}/10 GPUh；单GPU4、小batch、allocator12GiB、空闲≥18GiB。峰值与10s监测在gpu_peak_*.json/gpu_monitor.jsonl；resource_stop.json存在时须停止验收。派生总量约{size/1e9:.3f}GB/<30GB。未改/杀他人进程、无MPS/抢占、无下载/仿真；所有源视频stat读取前后核对相同。新adapter仅在`{L}`，未合并。\n\n长任务通过Python subprocess(start_new_session=True)分离会话（与setsid同效），stdin=/dev/null、stdout/stderr落盘并写状态文件；初始CPU准备使用setsid nohup；训练每16步保存resume，特征每capture receipt可续跑。主命令：`{C}/launch.py driver_gpu.py`；CPU评估`cpu_eval.py lora/shuffled`；分析`analyze.py`；近参数量冻结头`matched_head.py`。训练清单/原文/固定置换、两组训练逐步损失、终版adapter与SHA、全局特征与150帧patch、实验1–3JSON/逐例NPZ、混淆矩阵、成对CI和漂移均在本run目录。\n\n结果目录：`{P}`；特征：`{R/"features"/RUN}`；代码：`{C}`；断点：`{P/"progress.json"}`。完整H10/中英文、低置信层、2r对照、反事实排除原方向、输出范数/零率/事实误差见各JSON，未在正文重复所有分项。')
 if (P/'visual_review.json').exists():s.append('## 固定图目视核查\n\nG002/G027/G103/G126沿用冻结run固定ID，已实际查看原图与三种编码器热图对照。仿射映回未见明显错位；夹爪响应仍弥散，橙带在真实/打乱两组均增强；G103无可见夹爪仍有纹理热点。未修改GT、半径、抽样或checkpoint。图与SHA/观察见visual_review.json；热图不是分割概率。')
 if (P/'verification.json').exists():s.append('## 交付校验\n\n'+str(sum(load(P/'verification.json')['checks'].values()))+'项检查通过，包含双塔LoRA更新、同步数/配置、train parent归属、11条E83原索引与锚点相同、特征SHA/有限值与542capture、150帧patch及GPU预算。CPU校验台账首次numpy布尔序列化已修正；近参数量冻结头的parent元数据改存unicode，标签/分数/顺序逐项相等，未重训模型或使用GPU。')
 text='\n\n'.join(s)+'\n';out=T/'notes/e84_lora_summary_20261113.md';out.write_text(text);atomic(P/'report_draft_receipt.json',dict(timestamp_brt=brt(),path=str(out),sha256=sha(out),bytes=len(text.encode()),derived_total_bytes=size,gpu_hours=hours));print('REPORT_DRAFT',out,'GPUh',round(hours,4),'GB',round(size/1e9,3))
if __name__=='__main__':make()
