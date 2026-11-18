from run import *
import collections

def snapshot():
 files=[p for d in [T/'code_e99b3',T/'code_e100c',T/'code_e82',T/'code_e86',T/'code_e88',T/'data/e98_predicate_labels_train',T/'data/e98_predicate_labels_dev',T/'data/e97b_stages_v1',OLD] for p in d.rglob('*') if p.is_file()]
 return {str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files}
def name_transition(k):
 a,b=eval(k,{'__builtins__':{}},{});return (S[a] if a<10 else 'BATCH_WAIT')+'→'+(S[b] if b<10 else 'BATCH_WAIT')
def add_counts(rr,tol,key):
 c=collections.Counter()
 for r in rr:c.update(r['transitions'][tol][key])
 return dict(sorted(c.items(),key=lambda x:-x[1]))
def fmt(v):return 'NA' if v is None else f'{v:.3f}'
def interval(summary,key):return fmt(summary['point'][key])+' ['+', '.join(fmt(v) for v in summary['ci95'][key])+']'

def evaluate():
 assert (R/'CONTROLLER_LOCK.json').exists() and not (R/'DEV_ONCE_STARTED.json').exists()
 lock=json.loads((R/'CONTROLLER_LOCK.json').read_text());assert sha(T/'code_e101s/run.py')==lock['hashes']['script'];assert sha(R/'final_aux.pkl')==lock['hashes']['final_aux']
 before=snapshot();write('protected_snapshot_before.json',before)
 inputs=[T/'data/e97b_stages_v1/stages_predicates_v1.json',*[T/f'data/e98_predicate_labels_{pool}/{fn}.jsonl' for pool in ['train','dev'] for fn in ['predicates','stages']]]
 hashes={str(p):sha(p) for p in inputs};write('input_hashes_before.json',hashes)
 # Causality and legality contracts use only train OOF trajectories before dev starts.
 contracts=[]
 with (R/'final_aux.pkl').open('rb') as f:aux=pickle.load(f)
 for kind in ['S1','S2']:
  c=lock['selected'][kind];s=full[0];method=c['method'];z=np.load(R/(s['parent']+'_'+method+'_oof.npz'));a=build_aux(method,fold[s['parent']]);p=z['prob'];u=z['unknown'];e=emission(p,a)
  pred=smooth_rule(p,u,**c['config']) if kind=='S1' else hmm(e,a,**c['config']);cut=min(700,len(pred)//2);pre=smooth_rule(p[:cut],u[:cut],**c['config']) if kind=='S1' else hmm(e[:cut],a,**c['config']);assert np.array_equal(pred[:cut],pre)
  b=np.where(pred[1:]!=pred[:-1])[0]+1;assert all(A[pred[i-1],pred[i]] for i in b);assert len(b)<2 or min(np.diff(b))>=c['config']['m'];contracts.append(dict(kind=kind,prefix_frames=cut,prefix_equal=True,legal_edges=True,min_dwell=True))
 write('causality_contracts.json',contracts)
 write('DEV_ONCE_STARTED.json',dict(brt=now(),lock_sha=sha(R/'CONTROLLER_LOCK.json'),evaluation_script_sha=sha(Path(__file__)),bootstrap=2000,seed=10102,selection_frozen=True,models=['S0','S1','S2','S3'],worst_parent_rule='three lowest per-parent accuracy of train-selected causal recommendation'))
 records={k:[] for k in ['S0','S1','S2','S3']};predictions={};s0lock=read('B4_LOCK.json');total_frames=0
 for s in dev:
  parent=s['parent'];z=np.load(OLD/(parent+'_scores.npz'));assert np.array_equal(z['frames'],np.arange(s['n']));ss0=controller.replay(z['d_prob'],z['d_unknown'],s0lock['thresholds'],s0lock['persistence']);assert ss0==read(parent+'_b4.json')['stage_sequence'];seq={'S0':np.array([SI.get(v,10) for v in ss0])}
  for kind in ['S1','S2']:
   c=lock['selected'][kind];method=c['method'];p=z[method+'_prob'];u=z[method+'_unknown'];a=aux[method]
   if kind=='S1':seq[kind]=smooth_rule(p,u,**c['config'])
   else:
    e=emission(p,a);seq[kind]=hmm(e,a,**c['config']);seq['S3']=hmm(e,a,c['config']['temp'],c['config']['m'],True)
  predictions[parent]=seq;total_frames+=s['n'];np.savez_compressed(R/(parent+'_frozen_sequences.npz'),truth=truths[parent],**seq)
  for kind,pred in seq.items():records[kind].append(dict(parent=parent,**record(pred,truths[parent])))
  status('dev_frozen',parent=parent)
 write('frozen_parent_statistics.json',records)
 summaries={};rng=np.random.default_rng(10102);samples=rng.integers(20,size=(2000,20))
 for kind,rr in records.items():
  point=aggregate(rr);bs=[aggregate([rr[i] for i in ii]) for ii in samples];ci={}
  for k in point:
   vals=[b[k] for b in bs if b[k] is not None];ci[k]=np.quantile(vals,[.025,.975]).tolist() if vals else [None,None]
  transitions={tol:{key:{name_transition(k):v for k,v in add_counts(rr,tol,key).items()} for key in ['miss','extra']} for tol in ['15','30']};cm=np.sum([r['cm'] for r in rr],axis=0)
  per_stage={S[i]:dict(true_frames=int(cm[i].sum()),correct=int(cm[i,i]),precision=float(cm[i,i]/max(cm[:,i].sum(),1)),recall=float(cm[i,i]/max(cm[i].sum(),1)),f1=float(2*cm[i,i]/max(cm[i].sum()+cm[:,i].sum(),1))) for i in range(10)}
  summaries[kind]=dict(point=point,ci95=ci,transitions=transitions,per_stage=per_stage,confusion=cm.tolist())
 write('metrics.json',dict(label_basis='model(gpt-6-astra), non-human; user allowed dev reference truth',parents=20,frames=total_frames,bootstrap=2000,frame_rate=30,transition_match='exact directed stage pair, one-to-one max cardinality then minimum error within each type, tolerance 15/30 source frames; unmatched retained; unknown excluded only from stage scoring and truth boundaries',denominator='frame-weighted pooled confusion for accuracy and 10-class macro F1; boundary counts pooled; false/min uses total nominal source duration',delay='signed median among typed matches; nonnegative-only median also given; conditional on matches, not all true boundaries',models=summaries))
 # Paired bootstrap of S2-S0 supplied without selecting on dev.
 paired={k:[] for k in ['accuracy','macro_f1','recall30','f1_30','false_per_min30']}
 for ii in samples:
  a=aggregate([records['S2'][i] for i in ii]);b=aggregate([records['S0'][i] for i in ii])
  for k in paired:paired[k].append(a[k]-b[k])
 write('paired_S2_minus_S0.json',{k:dict(point=summaries['S2']['point'][k]-summaries['S0']['point'][k],ci95=np.quantile(v,[.025,.975]).tolist()) for k,v in paired.items()})
 status('metrics_complete',frames=total_frames)
 table='|方法|分数|逐帧准确率 [95%CI]|宏F1 [95%CI]|P@15|R@15|F1@15|P@30|R@30|F1@30|\n|---|---|---|---|---|---|---|---|---|---|\n'
 for kind,sm in summaries.items():
  method='d' if kind=='S0' else lock['selected']['S2' if kind=='S3' else kind]['method'];table+='|'+kind+(' 离线附录' if kind=='S3' else '')+'|'+method+'|'+'|'.join(interval(sm,k) for k in ['accuracy','macro_f1','precision15','recall15','f1_15','precision30','recall30','f1_30'])+'|\n'
 report('## 冻结 dev 一次评估 '+now()+'\n20 parent，'+str(total_frames)+'原帧，名义30Hz；347个参考边界口径另由统计核对。模型标注非人工；不能推断实机成功率。2000次parent成簇bootstrap，所有方法共享抽样；不重训、不按dev重选。\n'+table)
 table='|方法|匹配数15/30|绝对误差中位/P90 @15（帧）|绝对误差中位/P90 @30（帧）|虚假切换/分钟 @15|虚假切换/分钟 @30|有符号延迟中位 @30（帧）|非负延迟中位 @30（帧）|\n|---|---|---|---|---|---|---|---|\n'
 for kind,sm in summaries.items():
  p=sm['point'];table+='|'+kind+'|'+str(p['matched15'])+'/'+str(p['matched30'])+'|'+interval(sm,'error_median15')+' / '+interval(sm,'error_p9015')+'|'+interval(sm,'error_median30')+' / '+interval(sm,'error_p9030')+'|'+interval(sm,'false_per_min15')+'|'+interval(sm,'false_per_min30')+'|'+('非因果，不适用' if kind=='S3' else interval(sm,'signed_delay_median30'))+'|'+('不适用' if kind=='S3' else interval(sm,'nonnegative_delay_median30'))+'|\n'
 report(table+'\n误差和延迟仅在容差内成功匹配边界中计算，漏检不填0。带符号延迟负值表示提前；非负子集另列。S0历史19/347为无容差上限匹配，与本轮±15/30不可直接等同；S0逐帧序列与旧缓存逐项完全相同。宏F1固定10阶段；BATCH_WAIT作为额外错误预测列计入，unknown参考帧不评阶段。完整CI、每阶段表及所有转移漏检/误检见metrics.json。')
 report('## 按转移失败统计（±30帧）\n'+'\n'.join('### '+kind+'\n漏检最多：'+json.dumps(dict(list(sm['transitions']['30']['miss'].items())[:6]),ensure_ascii=False)+'\n\n误检最多：'+json.dumps(dict(list(sm['transitions']['30']['extra'].items())[:6]),ensure_ascii=False) for kind,sm in summaries.items()))
 # Worst parents fixed by train-selected recommendation, no controller modifications.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap,BoundaryNorm
 recommended=lock['recommended_before_dev'];worst=sorted(records[recommended],key=lambda r:aggregate([r])['accuracy'])[:3];analysis=[]
 for rank,rec in enumerate(worst,1):
  parent=rec['parent'];s=next(s for s in dev if s['parent']==parent);seq=predictions[parent];tt=truths[parent];fig,axs=plt.subplots(4,1,figsize=(16,4.4),sharex=True,constrained_layout=True);cmap=ListedColormap(['#ffffff']+list(plt.get_cmap('tab10').colors)+['#333333']);norm=BoundaryNorm(np.arange(-1.5,12.5),cmap.N)
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

if __name__=='__main__':
 try:evaluate()
 except Exception:
  write('EVAL_FAILURE_'+str(int(time.time()))+'.json',dict(brt=now(),traceback=traceback.format_exc()));status('EVAL_FAILED');raise
