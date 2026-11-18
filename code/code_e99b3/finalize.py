from common import *
import numpy as np,fcntl,collections
assert (R/'PIPELINE_COMPLETE.json').exists()
results=read('metrics.json');main=results['causal_main'];b4=read('B4_results.json')['summary'];protected=read('protected_snapshot.json');changed=[p for p,h in protected.items() if sha(p)!=h];assert not changed,changed
inputs=read('input_hashes.json');changed_inputs=[p for p,h in inputs.items() if sha(p)!=h];assert not changed_inputs,changed_inputs
changed_sources=[]
for p,old in read('source_stat_snapshot.json').items():
 st=Path(p).stat();new=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
 if new!=old:changed_sources.append(p)
assert not changed_sources
ledger=read('gpu_ledger.json');assert all(v['status']=='closed' for v in ledger);rawsecs=sum(v['seconds'] for v in ledger);secs=rawsecs+30*len(ledger);assert secs<10800
prod=read('code_snapshot_during_locked_dev.json')['files'];production_names={'common.py','features.py','heads.py','linear.py','vlm.py','metrics.py','controller.py'};assert all(sha(p)==h for p,h in prod.items() if Path(p).name in production_names)
sources=read('sources.json');states=[read(s['parent']+'_state_audit.json') for s in sources];assert all(s['max_available_ahead_seconds'] is None or s['max_available_ahead_seconds']<=1e-7 for s in states)
# Check complete key coverage, finiteness, and single artifact per method/parent without repeating inference.
checks=[]
for s in sources:
 if s['pool']!='dev':continue
 z=np.load(R/(s['parent']+'_scores.npz'));a=np.load(R/(s['parent']+'_vlm.npz'));assert len(z['frames'])==s['n'] and a['frames'].tolist()==s['annotation_frames']
 for method in ('b','c','d'):assert np.isfinite(z[method+'_prob']).all() and np.isfinite(z[method+'_unknown']).all()
 assert np.isfinite(a['prob']).all();checks.append(dict(parent=s['parent'],source_frames=s['n'],vlm_frames=len(a['frames'])))
put('final_prediction_audit.json',checks)
newbytes=sum(p.stat().st_size for folder in [R,T/'code_e99b3'] for p in folder.rglob('*') if p.is_file())+REPORT.stat().st_size
upper=newbytes+100_000_000;assert upper<20_000_000_000
final=dict(status='complete',brt=now(),label_basis='基于模型标注的 dev 真值',gpu_seconds=secs,gpuh=secs/3600,gpu_raw_ledger_seconds=rawsecs,gpu_cleanup_reserve_seconds=30*len(ledger),new_disk_bytes=newbytes,conservative_new_disk_upper_bytes=upper,protected_files=len(protected),protected_changes=changed,input_hash_changes=changed_inputs,source_metadata_changes=changed_sources,state_samples_missing=sum(v['missing_sample_slots'] for v in states),state_sample_slots=sum(v['slots'] for v in states),dev_main_records=main['meta']['records'],dev_reference_yes_no=main['meta']['answer_counts']['yes']+main['meta']['answer_counts']['no'],dev_scoring_once=True,strict_one_forward_per_frame=False,o_contract_dev_frames=64,o_contract_extra_full_forward_calls=2,dev_once_semantics='one frozen score evaluation and no selection; first64 dev frames additionally have 2 full-A contract forwards, original O unchanged',bootstrap=2000,source_frame_replay=sum(v['n'] for v in sources if v['pool']=='dev'),no_download=True,no_shared_environment_change=True,remaining=[])
put('FINAL_STATUS.json',final)
put('RECOVERY_RECEIPT.json',dict(resolved=True,original_failure='PIPELINE_FAILURE.json / contracts_run.log: sklearn absent',resolution='existing SciPy implementation, synthetic contracts passed before DEV_ONCE_LOCK',dev_repeated=False,final_status='FINAL_STATUS.json'))
put('VALID_RESULTS.json',dict(causal_main='metrics.json:causal_main',retrospective='metrics.json:retrospective_event',mixed_appendix='metrics.json:all_answer_mixed_appendix',b4='B4_results.json',final_status='FINAL_STATUS.json'))
def fmt(v):return 'NA' if v is None else f'{v:.3f}'
def cell(m):return fmt(m['value'])+' ['+', '.join(fmt(v) for v in m['ci95'])+']'
lines=['## 最终完整表与解释 '+now(),'所有结论均为 **基于模型标注的 dev 真值**；不代表人工真值或实机成功率。主表包含4074条因果记录，其中 yes/no=2533，unknown=445、NA=1096。参考unknown与NA不混作no。排除NA后，参考可答覆盖率=2533/2978=85.06%；含NA总记录的yes/no占比=62.17%。','### 四方法宏指标（2000次 parent bootstrap 95%CI；缺类重采样不填零）','|方法|AUROC|yes F1|no F1|两类宏F1|平衡准确率|','|---|---|---|---|---|---|']
for method,d in main['methods'].items():
 m=d['all']['macro'];lines.append('|'+ '|'.join([method,*[cell(m[k]) for k in ('AUROC','F1_yes','F1_no','F1_macro','balanced_accuracy')]])+'|')
lines+=['','### 逐谓词因果主表','每单元依次为AUROC / yes F1 / no F1 / BA，均附95%CI。P073无二分类真值；P103等稀少类别单独显示不可估计，不以0补齐。','|谓词|yes/no/unknown/NA|a Qwen|b SigLIP2|c O|d O+本体|d−c AUROC|','|---|---|---|---|---|---|---|']
for pid in [p['id'] for p in read('spec.json')['predicates']]:
 p=main['methods']['a']['all']['per_predicate'][pid];cells=[]
 for method in ('a','b','c','d'):
  m=main['methods'][method]['all']['per_predicate'][pid]['metrics'];cells.append(' / '.join(cell(m[k]) for k in ('AUROC','F1_yes','F1_no','balanced_accuracy')))
 diff=main['paired_d_minus_c']['all']['per_predicate'][pid]['AUROC'];lines.append('|'+ '|'.join([pid,f'{p["yes"]}/{p["no"]}/{p["unknown"]}/{p["NA"]}',*cells,cell(diff)])+'|')
lines+=['','### 可观测性与阶段分组宏指标','|分组|方法|AUROC|yes F1|no F1|BA|非NA数|','|---|---|---|---|---|---|---|']
for group in main['methods']['a']:
 if group in ('all','coverage_pooled'):continue
 for method in ('a','b','c','d'):
  d=main['methods'][method][group];m=d['macro']
  if not m:continue
  lines.append('|'+ '|'.join([group,method,*[cell(m[k]) for k in ('AUROC','F1_yes','F1_no','balanced_accuracy')],str(d['n'])])+'|')
lines+=['','### 弃权与覆盖率','a固定输出二值，P073按无载荷证据的事前规则弃权；b/c/d的unknown分类器仅用train标注（unknown vs yes/no），NA不参与训练。主二分类表不删除模型弃权的参考yes/no；选择性BA只作补充。覆盖率的逐谓词与宏CI见metrics.json。','|方法|非NA覆盖/总数|参考yes/no覆盖|参考unknown弃权|宏unknown召回|选择性BA|','|---|---|---|---|---|---|']
for method in ('a','b','c','d'):
 d=main['methods'][method];c=d['coverage_pooled'];m=d['all']['macro'];lines.append('|'+ '|'.join([method,f'{c["covered"]}/{c["non_NA"]} ({c["coverage"]:.2%})',f'{c["known_covered"]}/{c["known_reference"]}',f'{c["unknown_abstained"]}/{c["unknown_reference"]}',cell(m['unknown_recall']),cell(m['selective_balanced_accuracy'])])+'|')
lines+=['','### 回顾式补充口径','`retrospective_event`表仅27285条离线事件答案；`all_answer_mixed_appendix`保留全部27600条answer，其中315条是原始纯因果答案。三张表使用相同一次冻结的模型分数；没有重复前向或再选型。完整逐谓词/可观测性/阶段及CI均见metrics.json。','', '### 本体信号证据与限制']
pair=main['paired_d_minus_c']['all'];lines.append('宏 d−c AUROC='+cell(pair['macro']['AUROC'])+'；BA='+cell(pair['macro']['balanced_accuracy'])+'。区间为相同parent抽样的配对差。')
positive=[];negative=[]
for pid,v in pair['per_predicate'].items():
 ci=v['AUROC']['ci95']
 if ci[0] is not None and ci[0]>0:positive.append(pid)
 if ci[1] is not None and ci[1]<0:negative.append(pid)
lines.append('本实验中加入本体后AUROC的配对CI整体高于0：'+(', '.join(positive) or '无')+'；整体低于0：'+(', '.join(negative) or '无')+'。其余证据不确定。单seed、相关视觉/本体与模型标注不能证明“必须本体”；P073即使q/开度/FK齐全也缺载荷/力证据，不能回答机械锁紧。')
counts=collections.defaultdict(collections.Counter)
for rr in read('train_labels.json'):counts[rr['predicate']][rr['answer']]+=1
oneclass=[p for p,c in counts.items() if c['yes']==0 or c['no']==0]
lines.append('P103因果主表仅1个yes、1个no；四法AUROC=1只代表两个样本的排序，不构成能力证据。P011因果主表没有yes，AUROC/BA不可估计。hidden组198条中没有可估计的二值谓词，不能据此判断本体是否能回答隐藏状态。')
lines.append('视觉已有较强代理可预测性：b在P022提离、P031锁头接持、P041尾端抓持、P051回到入口、P061进入孔道、P062穿出、P071出口抓持、P081降低、P082释放的AUROC约0.85–0.97。这里的“可答”限于模型标注代理，不证明物理关系被正确识别。')
lines.append('本体在c之上对P031和P032的AUROC分别增加约20.36和8.97个百分点，配对区间高于0；但b的视觉分数已很高，不能称必须本体。宏AUROC提升区间跨0，宏BA提高2.64个百分点[1.16,6.36]。')
lines.append('训练标签只有单一二值类别或没有二值的谓词：'+', '.join(sorted(oneclass))+'。这些谓词的低分不能归因于视觉不可观测；所有训练标注为回顾式、且下一阶段候选谓词覆盖不完整。')
for method,name in [('b','冻结视觉'),('c','视觉世界O')]:
 rows=main['methods'][method]['all']['per_predicate'];rank=sorted([(p,v['metrics']['AUROC']['value']) for p,v in rows.items() if v['metrics']['AUROC']['value'] is not None],key=lambda v:v[1],reverse=True)
 lines.append(name+' AUROC描述性靠前谓词：'+', '.join(p+'='+fmt(v) for p,v in rank[:6])+'；较弱：'+', '.join(p+'='+fmt(v) for p,v in rank[-5:])+ '。这只是冻结dev结果的描述，不据此重选阈值/模型。')
a=main['methods']['a']['all']['macro'];lines.append('VLM零样本水平：宏AUROC '+cell(a['AUROC'])+'，yes/no F1 '+cell(a['F1_yes'])+' / '+cell(a['F1_no'])+'。仅当前三视图、每图≤448²像素、固定英文原问题与单提示；小目标分辨率和缺少历史可能限制穿出/抓持/同根关联，不将这一配置外推为VLM能力上限。')
lines+=['','### B4解释','阶段序列平均编辑距离='+fmt(b4['mean_edit_distance'])+'；归一化='+fmt(b4['mean_normalized_edit_distance'])+'。匹配边界 '+str(b4['matched'])+'/'+str(b4['true_boundaries'])+'，绝对时间误差中位/P90='+fmt(b4['boundary_abs_error_seconds_median'])+'/'+fmt(b4['boundary_abs_error_seconds_p90'])+'秒；提前比例（匹配内）='+fmt(b4['early_ratio_among_matched'])+'，漏切换比例='+fmt(b4['missed_ratio'])+'。未匹配边界不填零误差。全部CI、逐parent与逐帧状态见B4_results.json及*_b4.json。','**B4失败，不能把较好的离线谓词分数视为可靠阶段控制。** 1511次预测边界中1492次未匹配；S10→S03 484次、S03→S10 469次，恢复与失败反复切换。BATCH_WAIT占32106/55754帧，S10占15773帧；S06/S07/S08未产生任何占用帧。仅19个匹配边界上的中位误差不能掩盖94.52%的漏切换。细分见B4_failure_decomposition.json。控制使用d而非dev最佳方法，初态固定S01。严格采用v1转移守卫；同优先级冲突保持，unknown不等价no，P071仅记过去事件，失败/重试/换根清空。v1的S06→S07使用P062，但E98裁决阶段以可见换抓开始，二者语义差异会造成提前边界；不按dev修规则。训练候选下一阶段谓词覆盖不足、截断初态/多轮身份缺少独立传感，也限制回放。','', '### 因果与方法边界','a只看当前源帧三路；b按相机池化；c读取当前O组及四个因果历史时点的O表示并作固定均值池化；d增加t/t−0.5/t−1秒q、原始夹爪开度、FK位置与6D姿态（99D含有效位）。SigLIP2预处理完全复用E86 224 aspect padding/valid-mask 2x4/fp16合同；按原始source row同步三视角，时间编码用HDF时间差，当前latency编码0，不声称复刻旧P1的传感接收延迟。历史不跨时间断点。冻结人类短指令，未向dev世界头加入模型标注语言。','VLM 的 BF16 缓存分支与完整提示在训练帧上的 yes/no logit 差值误差为0.25（缓存[no,yes]=[40,36.5]，完整=[40,36.25]）；使用dev前固定的缓存路径，不声称数值逐位相同。O仅计算完整模型中R/W不可写入的O子图；训练侧和dev侧合同均核验完整A前向与该子图一致、随机未来R/W不改变O。未用未来机器人轨迹，未输入stage/cycle/attempt/答案作为特征。d的每个状态来源右端点≤该历史时刻的截止时间，未满足则填零加有效位，未用未来插值。', '训练答案可能依赖回顾式事件解释，而主表只用首遍因果可观测答案，存在监督语义偏移；回顾式补充表用于披露差异，不用于重新校准。训练9956条为回顾式模型答案，60parent中48fit/12内部验证。b的C与c/d的epoch、各谓词阈值以及B4持续帧数都在dev前冻结，再以全60parent重拟合；统计、归一化和unknown头均只用train。单seed parent bootstrap不覆盖训练seed不确定性，也不校正多谓词比较。','', '### 资源、命令与交付',json.dumps(final,ensure_ascii=False),'目录：`code_e99b3/`；`results/e99b3_20261115_01/`；`notes/e99b3_predicates_20261115.md`。','命令：`ssh -o BatchMode=yes <gpu-server>`；在T/code_e99b3使用只读e29/bin/python运行prepare.py、features.py train，随后`nohup setsid ... python -u pipeline.py`分离执行，输出状态和分段日志。CPU最终审计`finalize.py`。各脚本与输入哈希、GPU账本、锁定文件、所有冻结预测保留。实际解释器为/mnt/workspace/dengjianfeng/venvs/e29/bin/python；pipeline续跑标记E99B3_ATTEMPT=02，第一次CPU依赖失败的原日志/状态保留，无dev重前向。','GPU上界含原账本全部加载/I/O/失败段，另每段保守计30秒CUDA清理余量；准入≥估算峰值×1.5，定期显存/预算守护；未用MPS、未操作他人进程，未写同事或dataset目录。新结果、代码、报告外额外预留100MB，合计远低20GB。','本轮指定四法、分组指标、2000次parent区间及B4已完成；无需新增用户决定。若以后需要“必须本体”或机械锁紧结论，需要另轮人工真值与受控传感/载荷检验，本轮不执行。']
report('\n'.join(lines))
# Replace only the skeleton placeholders; retain chronological progress and exceptions for audit.
s=REPORT.read_text();s=s.replace('进行中。','已完成，见后续审计与最终章节。').replace('待完成。','已完成，见后续最终章节。');REPORT.write_text(s)
put('code_hashes.json',{str(p):sha(p) for p in (T/'code_e99b3').glob('*.py')})
status('complete',gpuh=secs/3600,new_disk_upper_gb=upper/1e9)
entry=f'\n## E99-B3 | {now()} BRT | COMPLETE\n目标：四方法阶段完成谓词+B4。全部结论基于模型标注的 dev 真值；60train按48/12内部验证，dev20一次；因果4074条/二值2533。四法宏AUROC：'+', '.join(m+'='+fmt(main['methods'][m]['all']['macro']['AUROC']['value']) for m in ('a','b','c','d'))+f'。B4编辑距离={b4["mean_edit_distance"]:.3f}；边界匹配{b4["matched"]}/{b4["true_boundaries"]}。GPU={secs/3600:.6f}h，新增上界={upper/1e9:.3f}GB；{len(protected)}保护文件零变化。改动：独立code_e99b3，结果results/e99b3_20261115_01，报告notes/e99b3_predicates_20261115.md。命令：e29/bin/python prepare.py、features.py train、setsid nohup pipeline.py、CPU finalize.py；完整锁定/哈希/分段命令见报告。结论：视觉与本体收益按谓词及配对CI披露，不能证明必要性；P073缺载荷证据。指定工作完成，无待用户决策。\n'
for path,text in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',entry),(T/'notes/worklog.md',f'\n- {now()} BRT E99-B3 COMPLETE：四法+B4/4074因果dev，基于模型标注的dev真值；GPU {secs/3600:.6f}h，新增≤{upper/1e9:.3f}GB；报告 notes/e99b3_predicates_20261115.md。\n')]:
 with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text);f.flush();fcntl.flock(f,fcntl.LOCK_UN)
print(json.dumps(final,ensure_ascii=False))
