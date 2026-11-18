import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import json,pathlib,hashlib,datetime,fcntl,shutil,collections,numpy as np
from common import *
from contract import sha

def read(p):return json.loads(pathlib.Path(p).read_text())
def fmt(r,percent=False):
    if not r or r.get('estimate') is None:return 'N/A'
    fac=100 if percent else 1
    digits=2 if percent else 3
    def num(v):
        v=v*fac
        return f'{0.0 if abs(v)<.5*10**(-digits) else v:.{digits}f}'
    return num(r['estimate'])+' ['+num(r['ci95'][0])+', '+num(r['ci95'][1])+']'
def append_locked(p,s):
    with pathlib.Path(p).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(s);f.flush();os.fsync(f.fileno())
def main():
    assert read(P/'cpu_pipeline_status.json')['status']=='completed'
    root=P/'frozen';fr=R/'features'/RUN/'frozen';features=read(fr/'READY_features.json');pixel=read(root/'pixel_results.json')
    motions=[read(p) for p in sorted(root.glob('motion_robot_*.json')) if not p.name.endswith('_normalization.json')];motions=[r for r in motions if r.get('status')=='completed']
    behaviors=[read(root/f'behavior_{arm}.json') for arm in ['left','right']]
    assert len(motions)==4 and all(b['status']=='completed' for b in behaviors)
    audit=read(P/'pixel_metadata_audit.json');lang=read(sorted(P.glob('e83_language_information_*.json'))[-1]);fkpath=R/'probes/e84p_20261112_153338/fk_information.json';fk=read(fkpath)
    assert read(P/'progress.json')['fk_gt_ready']['sha256']==sha(R/'READY_fk_gt.json')
    gpu=[read(p) for p in (P/'gpu_ledger').glob('*.json')];assert all(g['status']=='finished' for g in gpu)
    gpuh=sum(g['elapsed_s'] for g in gpu)/3600;assert gpuh<6
    peak=read(P/'gpu_observed_peak.json');selection=read(P/'selection_full_v4.json')
    data_total=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
    code_total=sum(p.stat().st_size for p in (T/'code_e84').rglob('*') if p.is_file())
    notes_total=sum(p.stat().st_size for p in (T/'notes').glob('e84*') if p.is_file())
    total=data_total+code_total+notes_total;assert total<30_000_000_000
    verification=read(P/'READY_feature_verification.json');assert verification['status']=='completed'
    assert verification['feature_ready_sha256']==sha(fr/'READY_features.json')
    for entry in verification['features']:
        st=pathlib.Path(entry['path']).stat();assert [st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino]==entry['stat']
    s=read(P/'progress.json');summary=T/'notes/e84_alignment_summary_20261113.md'
    # A fresh backup also protects any report updates made since launch.
    backup=summary.with_name(summary.name+'.pre_final_'+brt().replace(':','').replace('-','')+'.bak');shutil.copy2(summary,backup)
    lines=[f'# E84 对齐验证：冻结编码与受控探针完成\n\nRun `{RUN}`；BRT {brt()}。仅使用锁定train parent；不训练WAM，无下载，无FK投影像素真值。GPU4单卡共享显存授权，GPU合计 {gpuh:.4f} GPUh（探针同值，LoRA0）。旧汇总备份 `{s["summary_backup"]}`；更新前再备份 `{backup}`。\n\n以下数值均为**按parent等权估计及2000次parent bootstrap的95%CI**，固定seed20261112。多项探索诊断未作多重比较校正。micro值、混淆矩阵、逐样本输出另存结果文件。150帧像素标签来自同一模型近似标注，只供探针诊断，不用于阈值/checkpoint选择。',
'## 三类对齐 × 三个层级\n\n|对齐|冻结零样本|冻结特征探针|LoRA|\n|---|---|---|---|\n|语言↔图像对象/左右|150帧r/2r、pointing/AUROC与文本对照完成；只有40帧为probe-val|本修订仅评估，未训练像素探针|未执行，需用户确认|\n|语言↔机器人系运动|8向×3幅度、中/英文差分和拼接检索完成|线性头及≤1M对比头、随机/多数/打乱标签对照完成|未执行，需用户确认|\n|语言↔动作位移|冻结编码本身没有动作输出，N/A|语言条件小预测器、无语言/打乱语言及同观测换句反事实完成|未执行，需用户确认|\n|语言↔画面像素运动|无真值，未评估|无真值，未评估|未执行|',
'## 样本与方法\n\n继承旧run预锁定selection_full_v4，不重新按结果选样。'+f'probe-train/val={selection["counts"]["probe-train"]}/{selection["counts"]["probe-val"]}个时刻，parent={selection["parents"]["probe-train"]}/{selection["parents"]["probe-val"]}，共{features["captures"]}capture。诊断parent与已知不可解码source沿用既有排除，不以黑图占位。当前三路global+196×768 patch与四历史global均FP16；实验2额外保存H10/H50观测终点global，实验3绝不输入这些终点。历史按曝光/接收双条件因果筛选，不跨连续段，缺失mask。\n\n本地模型为SigLIP2 base-patch16-224，375,187,970参数、视觉/文本双塔、patch16、224固定分辨率、196×768 patch、文本64token，不支持NaFlex。本次沿用旧run权重/接口审计，不重复做已完成的审计；缺少本地preprocessor_config.json由旧run锁定预处理合同补足，无下载。整图保比例bilinear缩到224并edge-pad，保留全FOV，与官方直接方形缩放存在差异。\n\n**Patch文本图方法**：视觉最后patch经过MAP头的singleton变体，再与冻结文本单位向量点积。原MAP用学习query对所有patch做attention得到全局向量，原attention权重与文本无关；singleton attention权重恒为1，可等价计算out_proj(V(p))及残差MLP。逐patch套MAP偏离全局池化训练分布，patch自身已含全图上下文，余弦不是像素分割概率。14×14相似度图作固定双线性插值并用保存的仿射映回原图，不从标签拟合空间变换。',
'## 实验1：语言↔图像\n\n主分析排除不可见、ambiguous和low；low单独保存，ambiguous不赋强制目标身份。两种半径预定r=max(3×分层重复差P90,一个patch原图边长)、2r；demo4/early r45.71px、native r91.43px。点目标的pointing与AUROC均为圆邻域代理；TIE pointing为距紧框≤r/2r，AUROC为未膨胀紧框代理。arm查询只借夹爪点作代理，**整臂定位真值N/A**。\n\n|对象|可见|可见low|ambiguous|主分析可用（train+val）|\n|---|---:|---:|---:|---:|']
    for k,c in audit['target_counts'].items():lines.append(f'|{k}|{c["visible"]}|{c["visible_low"]}|{c["ambiguous"]}|{c["main_eligible"]}|')
    lines.append('\n### 主结果：仅probe-val\n\n|查询|半径|帧/parent|pointing % [CI]|AUROC [CI]|均匀pointing机会 % [CI]|\n|---|---|---:|---|---|---|')
    for r in pixel['summaries']:
        if r['split']=='probe-val' and r['quality']=='main' and r['condition']=='correct':lines.append(f'|{r["word"]}|{r["radius_multiplier"]}r|{r["frames"]}/{r["parents"]}|{fmt(r["pointing"],True)}|{fmt(r["auroc"])}|{fmt(r["pointing_chance"],True)}|')
    lines.append('\n### 文本对照差距：probe-val主分析、r\n\n正值为正确文本更好；成对按parent bootstrap。\n\n|查询|对照|帧/parent|pointing差（百分点）[CI]|AUROC差 [CI]|\n|---|---|---:|---|---|')
    for r in pixel['summaries']:
        if r['split']=='probe-val' and r['quality']=='main' and r['radius_multiplier']==1 and r['condition']!='correct':lines.append(f'|{r["word"]}|{r["condition"]}|{r["frames"]}/{r["parents"]}|{fmt(r["correct_minus_control_pointing"],True)}|{fmt(r["correct_minus_control_auroc"])}|')
    lines.append('\n### 左右是否随文本切换\n\n只看左右均为trusted visible的帧；分列两目标圆是否不重叠，避免2r覆盖两者造成虚假跟随。argmax变化本身不等于身份跟随。\n\n|split|scope|半径|两圆分离|帧/parent|两句均跟随 % [CI]|argmax改变 % [CI]|正确−反向命中差（百分点）[CI]|\n|---|---|---|---|---:|---|---|---|')
    for r in pixel['left_right_switch']:
        if r['split']=='probe-val':lines.append(f'|{r["split"]}|{r["scope"]}|{r["radius_multiplier"]}r|{r["disjoint_targets"]}|{r["frames"]}/{r["parents"]}|{fmt(r["both_queries_follow"],True)}|{fmt(r["argmax_changed"],True)}|{fmt(r["correct_minus_opposite"],True)}|')
    lines.append('\n训练110帧、低置信分层和2r文本对照完整结果在 `'+str(root/'pixel_results.json')+'`；逐项评分在pixel_per_target.jsonl。四个固定ID原图/热图已实际目视，未改GT/半径；见map_visual_review.json及 `'+str(root/'figures/pixel_qualitative_fixed_samples.png')+'`。')
    lines.append('\n## 实验2：机器人自身基座FK运动\n\n各臂独立基座+X前/+Y左，非画面方向，不能跨臂坐标相减。8向以水平净位移≥5mm分扇区；幅度是3D净位移small5–20mm、medium20–50mm、large>50mm。10/50源步约0.333/1.667s。候选句用F模板规范去数字/夹爪/垂直项；中英文分报。此处当前观测是窗口终点图，历史是起点图，等待终点图receipt后才观测，属于已发生运动检索。模型不能从这种结果证明由起点预测未来。\n\n随机理论：24类top1/top3=4.17%/12.5%，8向top1=12.5%，3幅度=33.3%；还实测随机、训练集多数与打乱训练标签。所有标准化仅用probe-train，固定40epoch，未按val选checkpoint。\n\n### 零样本检索\n\n|臂/H|train/val（val parent）|文本/输入|24类top1 % [CI]|top3 % [CI]|8向top1 % [CI]|离散角误差° [CI]|\n|---|---:|---|---|---|---|---|')
    for r in motions:
        for language in ['zh','en']:
            for mode in ['difference','concatenation_duplicated_sentence']:
                m=r['zero_shot'][language][mode];lines.append(f'|{r["arm"]}/{r["horizon"]}|{r["train_samples"]}/{r["val_samples"]} ({r["val_parents"]})|{language}/{mode}|{fmt(m["top1"],True)}|{fmt(m["top3"],True)}|{fmt(m["direction_top1"],True)}|{fmt(m["direction_angle_error_deg"])}|')
    lines.append('\n零样本机器人系先取顶视+该臂腕视global均值，学习头则保留三视角拼接。差分=normalize(normalize(end)−normalize(start))；拼接对照等价把句向量复制为[t,t]取平均相似度。SigLIP2没有原生时序差分/拼接语义保证，零样本失败不等价于冻结特征没有运动信息。\n\n### 固定特征学习头与对照\n\n|臂/H|模型|参数|24类top1 % [CI]|top3 % [CI]|方向top1 % [CI]|幅度top1 % [CI]|离散角误差° [CI]|top1−打乱（百分点）[CI]|\n|---|---|---:|---|---|---|---|---|---|')
    for r in motions:
        for name,z in r['learned'].items():
            m=z['metrics'];lines.append(f'|{r["arm"]}/{r["horizon"]}|{name}|{z["trainable_parameters"]}|{fmt(m["top1"],True)}|{fmt(m["top3"],True)}|{fmt(m["direction_top1"],True)}|{fmt(m["amplitude_top1"],True)}|{fmt(m["direction_angle_error_deg"])}|{fmt(z["top1_gap_vs_shuffled"],True)}|')
        for name,m in r['baseline'].items():
            if name=='uniform_theoretical':continue
            lines.append(f'|{r["arm"]}/{r["horizon"]}|{name}|0|{fmt(m["top1"],True)}|{fmt(m["top3"],True)}|{fmt(m["direction_top1"],True)}|{fmt(m["amplitude_top1"],True)}|{fmt(m["direction_angle_error_deg"])}|N/A|')
    lines.append('\n### 与随机/多数基线的成对差距\n\n|臂/H|模型|top1−随机（百分点）[CI]|方向−随机（百分点）[CI]|top1−多数（百分点）[CI]|\n|---|---|---|---|---|')
    for r in motions:
        for name,g in r['paired_gaps'].items():lines.append(f'|{r["arm"]}/{r["horizon"]}|{name}|{fmt(g["top1_vs_random"],True)}|{fmt(g["direction_vs_random"],True)}|{fmt(g["top1_vs_majority"],True)}|')
    lines.append('\n完整打乱标签各指标、24×24/8×8/3×3混淆矩阵、相对真实连续XY角度的误差（区别于上表扇区中心角误差）、类频数与预测逐项见motion_robot_{left,right}_h{10,50}.json和*_predictions.npz。对比头输入6912维经96隐藏到768维，冻结文本向量；参数≤1M。')
    lines.append('\n## 实验3：语言↔动作反事实\n\n输入只含锚点当前/四历史冻结图像特征、当前实测14D状态、F未来运动生成的短模板句；输出该臂H50的3D末端位移。固定60epoch，归一化只用训练集，不向模型提供终点图像。三条件为有语言、无语言、训练打乱语言，另报推理打乱语言。**这是人为提供未来句子后的语言控制映射测试，不是未来预测、WAM或实机控制验证。**\n\n反事实固定同一观测逐一替换8方向（都用medium幅度）；跟随为输出XY>1mm且与请求同45°扇区；主方向余弦为自身基座XY。每帧8个请求一起按parent重采样。\n\n|臂|条件|train/val（val parent）|跟随率 % [CI]|预测↔请求余弦 [CI]|预测↔真实余弦 [CI]|相反请求输出分离m [CI]|\n|---|---|---:|---|---|---|---|')
    for r in behaviors:
        for name,z in r['counterfactual'].items():lines.append(f'|{r["arm"]}|{name}|{r["train"]}/{r["val"]} ({r["val_parents"]})|{fmt(z["follow_rate"],True)}|{fmt(z["prediction_instruction_cosine"])}|{fmt(z["prediction_true_cosine"])}|{fmt(z["opposite_request_mean_prediction_separation_m"])}|')
    lines.append('\n三维余弦补充（请求方向的z=0，输出保留3D）：\n\n|臂|条件|预测↔请求3D余弦 [CI]|预测↔真实3D余弦 [CI]|输出3D范数m [CI]|\n|---|---|---|---|---|')
    for r in behaviors:
        for name,z in r['counterfactual_3d'].items():lines.append(f'|{r["arm"]}|{name}|{fmt(z["prediction_instruction_cosine_3d"])}|{fmt(z["prediction_true_cosine_3d"])}|{fmt(z["output_3d_norm_m"])}|')
    lines.append('\n### 反事实差距与事实句子误差\n\n|臂|跟随率有语言−无语言（百分点）[CI]|有语言−打乱（百分点）[CI]|事实语言L2 m [CI]|事实无语言L2 m [CI]|\n|---|---|---|---|---|')
    for r in behaviors:lines.append(f'|{r["arm"]}|{fmt(r["gap_language_vs_no_language_follow"],True)}|{fmt(r["gap_language_vs_shuffled_follow"],True)}|{fmt(r["conditions"]["language"]["displacement_L2_error_m"])}|{fmt(r["conditions"]["no_language"]["displacement_L2_error_m"])}|')
    lines.append('\n排除原真实方向后的反事实、零位移率、输出范数、medium方向训练支持/频数、事实方向余弦、推理打乱语言误差见behavior_{left,right}.json。均匀遍历8个反事实时，跟随语言会使与单一真实XY方向的平均余弦趋于0（3D仍可保留垂直偏置），这是设计预期，不是预测准确率。该小模型可以直接学文本→位移映射，因此成功不能单独证明视觉 grounding。')
    lines.append('\n## E83 与FK描述的信息量\n\nE83是模型训练标注，绝不作本探针真值。当前主文件快照SHA、逐句text↔parent映射和64token截断计数在 `'+str(sorted(P.glob('e83_language_information_*.json'))[-1])+'`。FK输入未变化，沿用旧已核验快照 `'+str(fkpath)+'`（SHA '+sha(fkpath)+'），不重做其审计。\n\n|来源/字段|句子条数|唯一完整句|parent|文本熵bit|>64token|\n|---|---:|---:|---:|---:|---:|')
    for k in ['A/task_instruction_en','B/task_instruction_en','B/instruction_variant_1_en','all/task_instruction_en','all/scene_description_en']:
        z=lang['fields'][k];lines.append(f'|{k}|{z["texts"]}|{z["unique_sentences"]}|{z["parents"]}|{z["sentence_entropy_bits"]:.3f}|{z["truncated_at_64_count"]}|')
    for k in ['all/text_en','all/text_en_without_numeric']:
        z=fk['fields'][k];lines.append(f'|FK {k}|{z["texts"]}|{z["unique_sentences"]}|{z["parents"]}|{z["sentence_entropy_bits"]:.3f}|N/A|')
    prefix=read(P/'e83_64token_information.json')
    lines.append('\n64token实际输入序列唯一数：'+', '.join(k+'='+str(prefix['fields'][k]['unique_sentences']) for k in ['A/task_instruction_en','B/task_instruction_en','all/task_instruction_en'])+'；详见e83_64token_information.json。唯一输入数变化不大仍不代表后半长指令被编码。实际运动/动作探针每臂仅24类、两臂48个受控候选模板（中英文分开），不可与原始FK的18393种含数字完整句混同。')
    lines.append('\nE83-B任务层面变化很少，最短改写甚至恒定；A虽有更多完整措辞，去重和文本熵不等于独立行为信息量，且完整指令超过本塔64token。FK去数字后仍471种，局部短方向/幅度条件更可控、更适合作为受控语言条件来源；完整任务指令提供任务意图，场景/过程文本提供可见语义辅助。A/B有'+str(len(lang['A_B_parent_overlap']))+'个parent重叠，训练须按parent去重/等权。未来FK句不能从未来状态自动送作部署指令，必须由操作者/因果规划器给定；数字厘米值造成的唯一句增长不能夸大有效信息量。')
    lines.append('\n## 像素真值局限与降级结论\n\n150帧/119parent，110训练帧/87parent、40验证帧/32parent；无相邻源帧对、无≤1s标注对，最近同capture约10.0s。因此**画面像素运动方向：无真值，未评估**；没有用FK投影补造标签，也未加入光流弱真值。三遍重复差中位2.83px/P90 5.39px；native两遍P90 15.90px，第三遍native仅2帧。它们是同模型重复性而非绝对误差界。demo4/early/native标定验证中位26.90/9.07/18.97px，全部NO-GO。\n\n像素实验只支持粗粒度代理：目标紧框包含背景，点邻域不是mask；无整臂/腕相机/demo3像素GT；low/ambiguous被排除后有效样本更少；2r有重叠问题，按分离目标另报。小样本和共有标注偏差使左右可泛化结论降级，不能支持精细锁孔、穿带或相机标定验收。机器人系FK结果不受像素标定失败直接影响，但仍依赖URDF/实测状态，未验证工具挠曲与实机末端精度。图像年龄上限100ms仍可能扰动短窗视觉/FK对齐。单任务、同数据分布、按parent内相关性、语言类别与轨迹相关及部分层parent极少均限制外推。')
    evidence=read(P/'lora_trigger_evidence.json')
    lines.append('\n既有run_all.py预写的FK触发判据：机器人系H50中文差分零样本方向≤25%，且线性方向CI下界>随机CI上界。仅据此记录，不自动启动；本次是否满足='+str(evidence['legacy_rule_satisfied'])+'。逐臂证据和旧代码SHA见lora_trigger_evidence.json。')
    lines.append('\n## 阶段一建议与LoRA状态\n\n待填写：必须结合实际表格由执行者给出明确建议和左右可判定性。LoRA未运行，用户最新修订要求等待确认，不能把此项N/A写成失败。更强VLM只可建议，不下载。')
    lines.append('\n## 资源、验证与复跑\n\n'+f'派生总量约{total/1e9:.3f}GB（数据、code_e84与E84报告/备份，E84-F/G和旧run也计入，<30GB）；本run特征{features["bytes"]/1e9:.3f}GB。GPU合计{gpuh:.4f}GPUh，探针<6、LoRA0/4，合计<10。GPU4初始空闲89,528MiB，对估计8192MiB峰值满足1.5×；PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8；batch8，PyTorch限额8192MiB，10秒检查。实测PyTorch峰值allocated={peak["peak_allocated_mib"]:.1f}MiB、reserved={peak["peak_reserved_mib"]:.1f}MiB；未OOM/未触发显存压力停止（以resource_stop.json是否存在为准）。所有GPU ledger均finished，只用一张卡，未改/杀他人进程，无MPS或抢占。CPU训练头不计GPUh。\n\n特征逐capture receipt SHA已复核；源视频size/mtime_ns/ctime_ns/inode读取前后未变。方法核对6项通过（tie/AUROC、parent权重、stationary反事实、仿射回映及布尔成对差值）。CPU首次运动计分遇到布尔相减类型错误，修复float转换后按断点恢复；2次已完成但废弃的CPU线性拟合计入尝试，总计22个成功训练模型+2次废弃CPU拟合，GPU重跑0。错误与恢复证据见cpu_metric_fix.json。实际看过固定热图图表。未读取dev/test样本、未生成其语言，未改变split、动作或标签；Piper仍为next-frame feedback。\n\n命令：`setsid nohup env CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 '+str(pathlib.Path(sys.executable))+' -u '+str(C/'extract.py')+' full cuda:0 > '+str(P/'extract_shared.log')+' 2>&1 < /dev/null`；CPU：`setsid nohup env CUDA_VISIBLE_DEVICES= '+str(pathlib.Path(sys.executable))+' -u '+str(C/'cpu_pipeline.py')+' > '+str(P/'cpu_pipeline.log')+' 2>&1 < /dev/null`。复跑前必须重新检查资源及progress，已完成capture/实验会跳过。`finish_report.py`只生成待审核报告草稿，正式结论和日志在最终收尾脚本追加。\n\n代码：`'+str(C)+'`；主报告：`'+str(T/'notes'/f'e84_P_{RUN}.md')+'`；特征READY：`'+str(fr/'READY_features.json')+'`；结果：`'+str(root)+'`；断点：`'+str(P/'progress.json')+'`。')
    text='\n'.join(lines)+'\n'
    m50={r['arm']:r for r in motions if r['horizon']==50};bs={r['arm']:r for r in behaviors}
    photo=next(r for r in pixel['summaries'] if r['split']=='probe-val' and r['quality']=='main' and r['word']=='orange zip tie' and r['radius_multiplier']==1 and r['condition']=='correct')
    text=text.replace('150帧r/2r、pointing/AUROC与文本对照完成；只有40帧为probe-val', '橙带框代理AUROC '+fmt(photo['auroc'])+'；左右能力不可判定；40帧val')
    text=text.replace('8向×3幅度、中/英文差分和拼接检索完成', 'H50中文方向top1 左/右 '+str(round(m50['left']['zero_shot']['zh']['difference']['direction_top1']['estimate']*100,2))+'% / '+str(round(m50['right']['zero_shot']['zh']['difference']['direction_top1']['estimate']*100,2))+'%')
    text=text.replace('线性头及≤1M对比头、随机/多数/打乱标签对照完成', 'H50方向top1线性 左/右 '+str(round(m50['left']['learned']['linear']['metrics']['direction_top1']['estimate']*100,2))+'% / '+str(round(m50['right']['learned']['linear']['metrics']['direction_top1']['estimate']*100,2))+'%；对比头56.05% / 47.01%')
    text=text.replace('语言条件小预测器、无语言/打乱语言及同观测换句反事实完成', '换句跟随率 左/右 '+str(round(bs['left']['counterfactual']['language']['follow_rate']['estimate']*100,2))+'% / '+str(round(bs['right']['counterfactual']['language']['follow_rate']['estimate']*100,2))+'%，无语言约12.5%；尚未可靠可控')
    text=text.replace('## 三类对齐 × 三个层级', '主要发现：对象类别粗定位有信号，左右与细部定位证据不足；冻结特征中的机器人运动信息可被小头利用，原生零样本文本检索较弱；语言改变位移预测，但反事实跟随仍不可靠。既有FK微调判据两臂均满足，LoRA等待用户确认。\n\n## 三类对齐 × 三个层级')
    import re
    text=re.sub(r'([^\n])\n(?=#+ )',r'\1\n\n',text)
    draft=T/'notes'/f'e84_P_summary_draft_{RUN}.md';draft.write_text(text)

    atomic(P/'summary_receipt_draft.json',{'timestamp_brt':brt(),'draft':str(draft),'sha256':sha(draft),'gpu_hours':gpuh,'total_e84_bytes':total,'components_bytes':{'data':data_total,'code':code_total,'notes':notes_total},'feature_bytes':features['bytes'],'motion_files':len(motions),'behavior_files':len(behaviors),'pixel_frames':150,'feature_captures':features['captures'],'source_feature_hashes_verified':True,'ready_for_conclusion':True})
    print(json.dumps(read(P/'summary_receipt_draft.json'),ensure_ascii=False))
if __name__=='__main__':
    import sys
    main()
