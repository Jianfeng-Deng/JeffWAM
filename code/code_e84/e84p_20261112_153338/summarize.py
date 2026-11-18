import os,json,pathlib,datetime,fcntl
from common import *
from contract import sha
def read(p):return json.loads(pathlib.Path(p).read_text()) if pathlib.Path(p).exists() else None
def metric(m,percent=True):
    if not m or m.get('estimate') is None:return 'N/A'
    s=100 if percent else 1;ci=m.get('ci95');value=f'{m["estimate"]*s:.1f}'
    return value+(f' [{ci[0]*s:.1f}, {ci[1]*s:.1f}]' if ci else '')+('%' if percent else '')
def motion(version,frame,arm):return read(P/version/f'motion_{frame}_{arm}_h50.json')
def color(version):
    r=read(P/version/'localization.json')
    if not r:return None
    return next((x for x in r['color_proxy'] if x['method']=='singleton_MAP' and x['camera']=='all' and x['query']=='orange zip tie'),None)
def size(p):return sum(x.stat().st_size for x in pathlib.Path(p).rglob('*') if x.is_file())
def main():
    state=read(P/'progress.json');f=read(R/'features'/RUN/'frozen/READY_features.json');assert f
    for name in ['READY_motion_results.json','READY_behavior_results.json','localization.json']:assert (P/'frozen'/name).exists(),name
    trigger=read(P/'lora_trigger.json');lora=read(R/'features'/RUN/'lora/READY_features.json')
    if trigger and trigger['triggered']:assert lora,'LoRA triggered but rerun incomplete'
    fk=read(P/'fk_information.json');gt=read(state['fk_gt_ready']['accepted_snapshot']);sel=read(f['selection_path'])
    audits=sorted(P.glob('e83_language_information_*.json'),key=lambda p:p.stat().st_mtime);lang=read(audits[-1]) if audits else read(P/'e83_language_information.json')
    now=brt();date=datetime.datetime.fromisoformat(now).strftime('%Y%m%d');report=T/'notes'/f'e84_alignment_summary_{date}.md'
    if report.exists() and RUN not in report.read_text():raise RuntimeError('existing summary belongs to another run; do not overwrite')
    versions=['frozen']+(['lora'] if lora else [])
    table='|对齐范围|冻结零样本|冻结特征探针|LoRA后|\n|---|---|---|---|\n'
    cc=color('frozen');lc=color('lora') if lora else None
    table+='|语言↔图像：HSV颜色proxy|point '+metric(cc['pointing'] if cc else None)+'；AUC '+metric(cc['auroc'] if cc else None,False)+'|本轮无对象线性探针；精确对象GT缺失|'+('point '+metric(lc['pointing'] if lc else None)+'；AUC '+metric(lc['auroc'] if lc else None,False) if lora else '未触发')+'|\n'
    for frame,label in [('robot','语言↔运动：各臂base XY FK'),('screen','语言↔像素运动：demo4粗投影')]:
        cols=[]
        for kind in ['zero','probe','lora']:
            cells=[]
            for arm,az in [('left','左'),('right','右')]:
                r=motion('lora' if kind=='lora' else 'frozen',frame,arm)
                m=None
                if r and r.get('status')=='completed':m=r['zero_shot']['zh']['difference']['direction_top1'] if kind=='zero' else r['learned']['contrastive_zh']['metrics']['direction_top1']
                cells.append(az+' '+metric(m))
            cols.append('；'.join(cells) if kind!='lora' or lora else '未触发')
        table+='|'+label+'（8向top1）|'+'|'.join(cols)+'|\n'
    bc=[]
    for version in versions:
        vals=[]
        for arm,az in [('left','左'),('right','右')]:
            r=read(P/version/f'behavior_{arm}.json');m=r.get('counterfactual',{}).get('language',{}).get('follow_rate') if r else None;vals.append(az+' '+metric(m))
        bc.append('；'.join(vals))
    table+='|语言↔动作：8向反事实跟随|N/A（位移回归需训练）|'+bc[0]+'|'+(bc[1] if lora else '未触发')+'|\n'
    text=f'''# E84 对齐实验汇总\n\nID `{RUN}`；完成快照 BRT {now}。本轮没有训练WAM，没有下载权重/数据，没有MuJoCo/仿真/retarget。**像素真值缺口使完整阶段一决策尚未闭环；下列FK和颜色proxy结果只在注明范围成立。**\n\n## 结果总表\n\n{table}\n数值为parent等权均值与2000次parent bootstrap 95%区间。图像列是颜色成员proxy，画面运动列是未认证粗投影；它们不是精确对象/像素验收。中文24模板为主要运动检索，英文敏感性结果在完整JSON。\n\n## 样本、合同与资源\n\nE82 revision3 train-only；F锁定195/49候选parent，seed=20261112084。诊断parent整父排除；一个已知左腕视频无法解码的capture整段隔离，未送黑图占位。最终选中{sel['parents']}，时刻{sel['counts']}；模式 `{f['mode']}`，设备 `{f['device']}`。若CPU：5分钟轮询满2小时均无显存<512MiB且利用率<5%的卡，按合同使用缩小子集。\n\n当前三路global+196patch、4个过去约0.2s间隔global、10/50端点global均fp16。实验2只看已观测窗口起点O与终点W，终点观测时钟等待真实image receipt；FK/投影沿用F时间映射，没有平移动作或标签。实验3仅起点/因果历史/当前实测状态+未来FK运动句，输出位移；不是预测未来，也不是实机命令验证。动作仍是next-frame反馈。\n\n共享E84派生实际 {size(R)/1e9:.3f}GB / 30GB；P features {size(R/'features'/RUN)/1e9:.3f}GB。累计GPU {state.get('gpu_hours',0):.5f}h；按阶段账本 `{P/'gpu_ledger'}`（CPU实验为0）。全部长任务setsid nohup，按capture/结果文件断点恢复。源视频stat读取前后核验；同事路径仅作为输入，无进程操作。\n\n## 视觉权重与patch图方法\n\n本地 google/siglip2-base-patch16-224，revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2；375,187,970参数，双塔strict加载；视觉224、patch16、14×14×768；文本词表256000、64token。固定分辨率，非NaFlex。fast tokenizer可用。缺失的processor元数据通过[官方配置文档](https://huggingface.co/google/siglip2-base-patch16-224/blob/main/preprocessor_config.json)核实为bilinear/224/均值与std .5，未下载模型/数据；本轮显式全视野保比例edge pad替代官方方形直接缩放，保存仿射并披露此分布差异。文本lowercase、64长度padding/truncation。\n\nMAP头对所有patch用同一个学习probe池化为全局语义向量；其注意力本身不随文本变化。本轮主图逐patch运行同一MAP头，再与文本向量余弦；singleton attention=1，可等价优化为V/out投影与residual MLP。另报原始patch余弦。逐patch头与全局训练分布不同，token已混入全图上下文；14×14对细尖端/锁孔可能丢细节，分数不是分割概率。\n\n## 真值适用范围\n\nFK有效 {fk['rows'] if fk else 'N/A'} 行，来自实测反馈和URDF，各臂独立base_link，+X前、+Y机器人左、+Z上；不能跨臂直接当共同世界坐标。幅度使用三维净位移small<2cm、medium≤5cm、large>5cm，水平<5mm的hold单列/不纳入24类方向。往返可净位移很小，不等于整窗静止。\n\nF目视30投影帧/30 HSV帧/20方向对：投影可见点误差中位37.44px、P90 71.72px、最大116.64px，人工估点不确定15–25px；精确像素GT未获批准。HSV误检帧顶/左腕/右腕=8/10、10/10、6/10；顶视另有5/10可见橙色漏检，不能将这些帧比例当人口precision/recall。demo3/demo6无投影GT，demo6沿用demo4标定NO-GO；右腕手眼不开放为夹爪中心GT。\n\n因此right/left arm/gripper仅保留未认证投影圆盘诊断；locking head/tip没有测试GT。不能根据这些数值声称左右理解、小对象定位成立或失败；也不以它们触发LoRA。HSV只评橙色像素成员，不证明目标扎带身份。\n\n'''
    text+='## 运动检索和探针\n\n|版本/坐标/臂|24类top1|top3|8向top1|幅度top1|方向角误差°|\n|---|---:|---:|---:|---:|---:|\n'
    for version in versions:
        for frame in ['robot','screen']:
            for arm in ['left','right']:
                r=motion(version,frame,arm)
                if not r or r.get('status')!='completed':continue
                for kind,label in [('zero','zero'),('linear','linear'),('contrastive_zh','contrast')]:
                    m=r['zero_shot']['zh']['difference'] if kind=='zero' else r['learned'][kind]['metrics']
                    text+=f'|{version}/{frame}/{arm}/{label}|'+ '|'.join(metric(m.get(k),k!='direction_angle_error_deg') for k in ['top1','top3','direction_top1','amplitude_top1','direction_angle_error_deg'])+'|\n'
    text+='\n均匀随机24类top1=4.17%、top3=12.5%，8向=12.5%、3幅度=33.3%。每项另有实际随机、训练多数类与打乱训练标签控制；完整混淆矩阵/配对差值位于各motion_*json与predictions.npz。线性头方向/幅度分开训练；对比头738144参数；归一化只拟合probe-train，固定40epoch，无probe-val选checkpoint。画面幅度边界只用probe-train像素幅度三分位，故与机器人系cm幅度不是同一任务。拼接零样本把句子复制成[t,t]，不是模型原生时间编码；当前−历史表示也未被SigLIP2专门训练，因此失败不可自动归因于视觉塔丢失信息。\n\n'
    text+='## 行为可控性与反事实\n\n|版本/臂/条件|8向跟随率|预测↔请求余弦|预测↔真实方向余弦|<1mm输出率|\n|---|---:|---:|---:|---:|\n'
    for version in versions:
        for arm in ['left','right']:
            b=read(P/version/f'behavior_{arm}.json')
            if not b or b.get('status')!='completed':continue
            for condition,m in b['counterfactual'].items():text+=f'|{version}/{arm}/{condition}|'+ '|'.join(metric(m.get(k),k in ['follow_rate','near_zero_rate']) for k in ['follow_rate','prediction_instruction_cosine','prediction_true_cosine','near_zero_rate'])+'|\n'
    text+='\n同一观测换8方向，幅度固定medium；跟随以自身base XY最近45°方向，近零<1mm不记成功。真实方向与请求不同是有意反事实，跟随请求不应继续贴近真实方向。输入无未来像素；语言来自未来FK类别模板，检验小预测器的条件控制，不是未来预测准确性、WAM闭环或实机成功率。训练语言/无语言/打乱语言同固定60epoch，另有测试时打乱语言控制，参数数目与输出范数在behavior_*json。\n\n'
    text+='## LoRA、偏离与阶段一建议\n\n'
    if lora:
        tr=read(R/'lora'/RUN/'training_result.json');drift=read(P/'lora_drift.json');text+=f'条件LoRA已执行：rank8，双塔最后2层Q/V，98304参数，lr1e-5，固定3epoch，训练{tr["train_pairs"]}对/{tr["train_parents"]}parent，{tr["device"]}。触发来自FK方向冻结零样本≤25%且线性CI高于随机CI，未用未认证像素GT。方向反转/换臂与蓝绿难负例，编码偏离约束0.1；HSV目标只是弱颜色监督。最后epoch固定提交，不用val挑checkpoint。漂移与LoRA−冻结的parent配对区间见lora_drift.json。\n\n'
    else:text+='按预注册条件未触发LoRA，不将未认证GT导致的低分作为微调理由。\n\n'
    # The recommendation follows actual paired results and stays within qualified truth.
    drift=read(P/'lora_drift.json');positive=[];negative=[]
    if drift:
        for arm,rr in drift.get('paired_probe_comparisons',{}).items():
            for kind,rrr in rr.items():
                m=rrr['direction_top1_LoRA_minus_frozen']
                if m['ci95'][0]>0:positive.append(arm+'/'+kind)
                if m['ci95'][1]<0:negative.append(arm+'/'+kind)
    if positive:
        text+='**明确建议（FK运动范围）：需要考虑LoRA对比微调。** LoRA后方向探针的parent配对收益区间高于0：'+', '.join(positive)+'；这只支持本轮已验证的运动语义范围，不证明精确左右/尖端/锁孔对齐已修复。全量阶段一投入前仍需独立像素真值，保留冻结编码器对照，文本塔是否需LoRA必须另做视觉-only同预算消融。\n\n'
    else:
        text+='**明确建议：目前保留冻结SigLIP2+小对齐头；不据本轮结果投入全量双塔LoRA。** 这是一条保守实施选择，不是“冻结已足够”的完整验收；本轮LoRA没有呈现可信的parent配对方向收益'+('，存在明确退化项：'+', '.join(negative) if negative else '')+'。补齐独立像素真值后再决定阶段一是否需要视觉LoRA。\n\n'
    text+='本轮双塔LoRA未隔离视觉/文本贡献，不能单独决定文本LoRA；沿用用户方案先冻结文本塔。现有真值不足以支持“需要更强VLM”，不建议下载或更换权重。\n\n'
    text+='## E83与FK语言信息量\n\n'
    if lang:
        text+=f'E83主文件固定快照：{lang["records"]}条/{lang["unique_parents"]}parent，A/B交集={lang.get("A_B_parent_overlap",{})}。\n\n|文本|记录|唯一完整句|parent|文本熵bit|>64token|\n|---|---:|---:|---:|---:|---:|\n'
        for key in ['A/task_instruction_en','B/task_instruction_en','B/instruction_variant_1_en','all/task_instruction_en','all/scene_description_en','A/process_narration_en']:
            if key in lang['fields']:
                d=lang['fields'][key];text+=f'|{key}|{d["texts"]}|{d["unique_sentences"]}|{d["parents"]}|{d["sentence_entropy_bits"]:.3f}|{d["truncated_at_64_count"]}|\n'
    if fk:
        text+='\n|FK文本|条数|唯一句|parent|文本熵bit|\n|---|---:|---:|---:|---:|\n'
        for key in ['all/text_en','all/text_en_without_numeric']:
            d=fk['fields'][key];text+=f'|{key}|{d["texts"]}|{d["unique_sentences"]}|{d["parents"]}|{d["sentence_entropy_bits"]:.3f}|\n'
    text+='\nB的任务指令仅8种，短改写1种，主要表达单任务固定流程；不能支撑丰富指令可控性结论。另有'+str(len(lang.get('B_unverified_full_sequence_ids',[])) if lang else '13')+'条B记录未实证完整双臂流程，规范性补全不能当实际动作事件；坏腕视角记录需按missing_views筛选。FK数值厘米会人为增加精确句子数，因此同时看去数字版和8×3语义覆盖。建议部署任务层仍用短任务指令，局部方向/幅度控制信号优先来自FK短句；E83可分句/按过程段作语言训练辅助，需64token分块并按parent去重，场景/回顾不是动作真值，未来过程字幕不作当前部署输入。\n\n'
    text+='## 局限、未完成与交付\n\n- 精确物体/左右像素GT、锁头/尖端GT缺失；像素空间理解的正式验收未完成。需要用户决定是否另开独立人工点击/实例mask与demo4重新标定任务；demo6重标定单独验收。\n- 单任务、相同机位背景、parent内强相关；val虽全来自E82 train，仍只是探针留出，未读取dev/test，不能称原冻结测试集结果。demo6每层val只有2parent，跨日期推断区间尤其不稳定。\n- CPU缩小样本若启用，LoRA128训练对只是小规模可修复性尝试；不据失败断言更强VLM必需。不含v3 WAM训练、真实语言命令执行或实机成功率。\n- 模型生成E83仅训练目标，未用作测试真值、阈值或checkpoint选型。FK动作语言只能检验对应运动语义；语言—动作最终仍需阶段二动作分支学习。\n\n'
    text+=f'报告：`{state["paths"]["report"]}`；进度：`{P/"progress.json"}`；代码：`{C}`；数据：`{R}`。主结果 `{P}/{{frozen,lora}}/`，适配器 `{R/"lora"/RUN}`；图表 `{P/"figures"}`。\n'
    report.write_text(text)
    append_report('## E84-P最终状态\n\n'+now+' scoped实验已完成；完整阶段一像素验收仍被独立真值缺口阻塞。汇总 `'+str(report)+'`。未将N/A伪装为失败或通过；无需新增训练/下载授权。')
    manifest=P/'RESULT_SHA256.jsonl'
    with manifest.open('w') as m:
        for path in [report,pathlib.Path(state['paths']['report']),*sorted((P/'frozen').glob('*.json')),*sorted((P/'lora').glob('*.json'))]:
            if path.is_file():m.write(json.dumps({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size},ensure_ascii=False)+'\n')
    entry=f'\n## E84-P {RUN} 最终 | {now}\n- 目标：三种对齐探针，阶段一冻结/LoRA证据；无WAM训练。\n- 改动/命令：code_e84/{RUN}/{{prepare,extract,run_all}}.py；setsid nohup；入口SHA与parent gate。\n- 目录：data/e84_piper_align_v1/{{features,probes,lora}}/{RUN}；汇总notes/{report.name}。\n- 结果：{sel["counts"]}时刻，{sel["parents"]}parent；{f["device"]}；GPU {state.get("gpu_hours",0):.5f}h；E84总派生{size(R)/1e9:.3f}GB。\n- 结论：FK/受限颜色proxy和反事实结果已交；精确对象/像素GT不足，不能闭环全阶段一结论；保留冻结+小头候选，补独立GT再决策LoRA，不下载更强VLM。\n- 状态：completed_scoped_with_pixel_gt_blocker；精确像素验收未完成。\n'
    for path,line in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',entry),(T/'notes/worklog.md',f'\n- {now} E84-P {RUN}：受限对齐探针完成，{f["mode"]} {sel["counts"]}时刻，LoRA={bool(lora)}，GPU{state.get("gpu_hours",0):.5f}h；精确像素GT缺口保留；汇总notes/{report.name}。\n')]:
        with path.open('a') as z:fcntl.flock(z,fcntl.LOCK_EX);z.write(line);z.flush();os.fsync(z.fileno())
    atomic(P/'completion.json',{'timestamp_brt':now,'status':'completed_scoped_with_pixel_gt_blocker','summary':str(report),'sha256':sha(report),'gpu_hours':state.get('gpu_hours',0),'total_e84_bytes':size(R),'unfinished':['precise object/left-right pixel GT','full stage-one frozen-vs-LoRA decision'],'user_decision':'whether to commission independent pixel labels and recalibration'})
    print(json.dumps({'summary':str(report),'status':'completed_scoped_with_pixel_gt_blocker','bytes':size(R),'gpu_hours':state.get('gpu_hours',0)},ensure_ascii=False))
if __name__=='__main__':main()
