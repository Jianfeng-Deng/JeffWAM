from evaluate import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
lock=json.loads((R/'CONTROLLER_LOCK.json').read_text())
assert (R/'DEV_ONCE_STARTED.json').exists() and (R/'metrics.json').exists() and not (R/'FINAL_STATUS.json').exists()
records=json.loads((R/'frozen_parent_statistics.json').read_text())
metrics=json.loads((R/'metrics.json').read_text());summaries=metrics['models'];total_frames=metrics['frames']
predictions={s['parent']:{k:np.load(R/(s['parent']+'_frozen_sequences.npz'))[k] for k in ['S0','S1','S2','S3']} for s in dev}
with (R/'final_aux.pkl').open('rb') as f:aux=pickle.load(f)
before=json.loads((R/'protected_snapshot_before.json').read_text());hashes=json.loads((R/'input_hashes_before.json').read_text())
report('## 评估后制图恢复 '+now()+'\n一次冻结dev的序列、统计、2000次bootstrap均已完成并落盘。随后绘图颜色分箱配置报错，保留evaluate_01.log与EVAL_FAILURE文件。仅修正绘图并从已冻结序列/metrics继续报告、分析和审计；没有再次运行dev控制器、模型或阈值选型。')
# Worst parents fixed by train-selected recommendation, no controller modifications.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
recommended=lock['recommended_before_dev'];worst=sorted(records[recommended],key=lambda r:aggregate([r])['accuracy'])[:3];analysis=[]
for rank,rec in enumerate(worst,1):
 parent=rec['parent'];s=next(s for s in dev if s['parent']==parent);seq=predictions[parent];tt=truths[parent];fig,axs=plt.subplots(4,1,figsize=(16,4.4),sharex=True,constrained_layout=True);cmap=ListedColormap(['#ffffff']+list(plt.get_cmap('tab10').colors)+['#333333']);norm=BoundaryNorm(np.arange(-1.5,11.5),cmap.N)
 for ax,(name,yy) in zip(axs,[('Reference (model)',tt)]+[(k,seq[k]) for k in ['S0','S1','S2']]):
  ax.imshow(yy[None,:],aspect='auto',cmap=cmap,norm=norm,extent=[0,len(yy),0,1],interpolation='nearest');ax.set_yticks([]);ax.set_ylabel(name,rotation=0,ha='right',va='center')
 axs[-1].set_xlabel('Original source frame (30 fps nominal)');fig.suptitle(parent+' | worst '+str(rank)+' by '+recommended+' accuracy; S01..S10 = tab10, black = BATCH_WAIT');path=R/(f'worst_{rank}_'+parent+'.png');fig.savefig(path,dpi=140);plt.close(fig)
 valid=tt>=0;cm=np.zeros((10,11),int);np.add.at(cm,(tt[valid],seq[recommended][valid]),1);pairs=sorted([(int(cm[i,j]),S[i],S[j] if j<10 else 'BATCH_WAIT') for i in range(10) for j in range(11) if i!=j],reverse=True)[:4]
 notes=[dict(stage=r['stage_id'],start=r['stage_start'],reason=r.get('transition_reason','')) for r in stages['dev'] if r['parent_id']==parent];a=dict(parent=parent,rank=rank,accuracy=aggregate([rec])['accuracy'],recall30=aggregate([rec])['recall30'],top_frame_confusions=pairs,annotated_reasons=notes,plot=str(path));analysis.append(a)
 report('### 最差parent '+str(rank)+'：'+parent+'\n训练预选'+recommended+'准确率='+fmt(a['accuracy'])+'，recall@30='+fmt(a['recall30'])+'。主要帧混淆（帧数/真值/预测）：'+json.dumps(pairs,ensure_ascii=False)+'。\n\n!['+parent+']('+str(path)+')\n\n原标签边界说明（非新增标注）：'+json.dumps(notes[:5],ensure_ascii=False))
write('worst_parent_analysis.json',analysis)
# Descriptive score discrimination; frozen scores and existing labels only, no fitting.
diag=[]
scorecache={s['parent']:{m:np.load(OLD/(s['parent']+'_scores.npz'))[m+'_prob'] for m in ['b','d']} for s in dev}
for method in ['b','d']:
 for j,pid in enumerate(PIDS):
  yy=[];pp=[]
  for lab in read('dev_labels.json'):
   if lab['predicate']!=pid or not lab.get('causal_valid') or lab.get('causal_answer') not in ['yes','no']:continue
   parent=lab['parent'];yy.append(lab['causal_answer']=='yes');pp.append(float(scorecache[parent][method][lab['frame'],j]))
  yy=np.array(yy,dtype=bool);pp=np.array(pp);pos=pp[yy];neg=pp[~yy]
  from scipy.stats import rankdata
  auc=float((rankdata(pp)[yy].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))) if len(pos)*len(neg) else None
  diag.append(dict(method=method,predicate=pid,positive=len(pos),negative=len(neg),auc=auc))
write('frozen_predicate_discrimination.json',diag)
# Final audit compares metadata to snapshot and label hashes; no protected writes authorized.
after=snapshot();changed=[p for p,v in before.items() if after.get(p)!=v];new_protected=[p for p in after if p not in before];hashchanged=[p for p,h in hashes.items() if sha(Path(p))!=h];assert not changed and not new_protected and not hashchanged
disk=sum(p.stat().st_size for d in [R,T/'code_e101s'] for p in d.rglob('*') if p.is_file())+REPORT.stat().st_size;assert disk<5_000_000_000
conclusion='训练预选 '+recommended+'（'+lock['selected'][recommended]['method']+'）为因果研究候选；不建议直接实机部署。' if summaries[recommended]['point']['recall30']<.7 else '训练预选 '+recommended+' 达到代理recall门槛，但仍需人工真值与实机验证。'
recall=summaries[recommended]['point']['recall30'];r3=summaries['S3']['point']['recall30']
report('## 失败解释与部署建议\n'+conclusion+' recall@±30='+fmt(recall)+'，距离0.7还差'+fmt(max(0,.7-recall))+'。S3非因果recall='+fmt(r3)+'，仅为离线比较，不能保证一定高于在线，更不是理论上界。部署推荐保持训练预选结果，不依据dev换分数源或阈值。\n\n错误来源应区分：① 谓词分数不可分/监督覆盖：P042/P091/P102训练缺二值类别、P101等因果dev区分弱，恢复与结束判断易混；发射只含谓词分数，平滑不能创造缺失证据。② 阶段定义：S06→S07守卫P062可见穿出，与E98换抓开始不一致；训练复标边界差中位20/P90 76帧、dev复标P90 128.5帧，±15/30严于部分模型标注不确定性。③ 可见性：P073无载荷证据必须弃权，遮挡、同根身份和失败恢复缺独立证据。以上结合冻结分数、混淆及原标签注释作诊断；本轮没有重新观看视频或新增标注，不能将具体parent错误断言为已证实的物理原因。\n\n下一步优先补训练侧候选下一阶段/失败恢复/同根身份的因果谓词监督与分数，再统一S06/S07可见事件定义，并建立独立人工边界参考。P073若需机械锁紧须另定载荷检验；时间平滑本身不足。另需全60训练轨迹袋外特征以降低12-parent调参覆盖偏差。普通HMM只用自环拟合几何驻留，本轮经验驻留分布已保存，显式HSMM是后续可选项。不能把本轮模型标签dev作为真实部署验收。')
for method,a in aux.items():write('transition_durations_'+method+'.json',dict(transition=a['transition'].tolist(),durations=a['durations'],label_transitions_excluded_by_graph=a['illegal']))
report('## 资源、命令与审计\nCPU线程限制2，CUDA_VISIBLE_DEVICES为空，GPUh=0（含失败段）；新增落盘当前 '+str(disk)+' 字节，另预留1MB报告/日志，远低5GB。无下载、无大文件复制；旧代码/标签/结果只读；快照核对 '+str(len(before))+' 文件元数据零变化、5个标签/定义SHA256零变化。\n\n命令：`ssh -o BatchMode=yes <gpu-server>`；`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u T/code_e101s/run.py fit`，完成冻结后同环境运行`evaluate.py`。结果 `T/results/e101s_20261115_01/`。首段CPU启动因阶段unknown未映射失败，修正为不评分的-1并保留fit_01.log，dev尚未开始；fit_02.log为正式训练。没有GPU失败段。\n\n未完成/偏离：未提取其余48训练parent全帧特征；未跑可选b/d融合；HMM驻留使用自环+最短驻留，未显式使用完整经验驻留hazard；没有重新人工标注、视频证据复审或实机部署。无需用户决定即可交付本轮结果；后续全轨迹提取和人工真值工作需另轮授权与预算。')
write('FINAL_STATUS.json',dict(status='COMPLETE_WITH_DECLARED_LIMITATIONS',brt=now(),gpu_hours=0,new_disk_bytes=disk,new_disk_upper_bytes=disk+1000000,dev_evaluations=1,parents=20,frames=total_frames,bootstrap=2000,protected_metadata_files=len(before),protected_changes=changed,new_protected_files=new_protected,input_hash_changes=hashchanged,train_recommended=recommended,recall30=recall,deploy_ready=recall>=.7,limitations=['12/60 train full trajectories for controller tuning; 60-parent 5-fold scores at annotations','ordinary HMM geometric dwell + minimum dwell, not HSMM','model-labeled non-human dev truth','S1/S2 no BATCH_WAIT terminal state','no video reinspection']))
status('COMPLETE_WITH_DECLARED_LIMITATIONS',recommended=recommended,recall30=recall,gpu_hours=0)
