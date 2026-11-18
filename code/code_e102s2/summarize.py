from tune import *
import collections,fcntl
import predicate_metrics as pm

def summary(arr,names):return {k:dict(value=pm.finite(arr[0,i]),ci95=pm.interval(arr[1:,i])) for i,k in enumerate(names)}
def main():
 assert (R/'DEV_FROZEN_COMPLETE.json').exists();lock=read('CONTROLLER_LOCK.json');hl=read('HEADS_LOCK.json');ds=sorted(dev,key=lambda s:s['parent']);parentnames=[s['parent'] for s in ds];pari={p:i for i,p in enumerate(parentnames)};samples=np.random.default_rng(10201).integers(20,size=(2000,20));draws=np.stack([np.bincount(v,minlength=20) for v in samples]);np.save(R/'bootstrap_parent_draws.npy',samples)
 baseline_audit=[]
 for ss in ds:
  aa=np.load(R/(ss['parent']+'_dev_frozen_scores.npz'));bb=np.load(OLD/(ss['parent']+'_scores.npz'));pe=float(abs(aa['H0_prob']-bb['b_prob']).max());ue=float(abs(aa['H0_unknown']-bb['b_unknown']).max());assert pe<1e-6 and ue<1e-6,(ss['parent'],pe,ue);baseline_audit.append(dict(parent=ss['parent'],prob_max_abs=pe,unknown_max_abs=ue))
 put('H0_E99_DEV_FROZEN_REPRODUCTION.json',baseline_audit)
 labels=[r for r in oldread('dev_labels.json') if r['causal_valid']];groups={'all':np.ones(len(labels),bool)};groups.update({'visibility:'+v:np.array([r['causal_visibility']==v for r in labels]) for v in ['visible','partial','hidden']});groups.update({'stage:'+v:np.array([r['causal_stage']==v for r in labels]) for v in S});yy=np.array([{'no':0,'yes':1,'unknown':2,'not_applicable':3}[r['causal_answer']] for r in labels]);pid=np.array([PI[r['predicate']] for r in labels]);par=np.array([pari[r['parent']] for r in labels]);predresults={};arrays={};paired={}
 for method in ['H0','H1','H2','H3']:
  scores={s['parent']:np.load(R/(s['parent']+'_dev_frozen_scores.npz')) for s in ds};p=np.array([scores[r['parent']][method+'_prob'][r['frame'],PI[r['predicate']]] for r in labels]);ab=np.array([scores[r['parent']][method+'_unknown'][r['frame'],PI[r['predicate']]]>=.5 for r in labels]);out={};paired[method]={}
  for g,sel in groups.items():
   sel=sel&(yy!=3);vals=[];by={};pdel={}
   for j in range(21):
    mask=sel&(pid==j);a=pm.stats_rows(yy[mask],p[mask],ab[mask],hl['methods'][method]['thresholds'][j],par[mask],draws);vals.append(a);by[PIDS[j]]=dict(n=int(mask.sum()),yes=int((yy[mask]==1).sum()),no=int((yy[mask]==0).sum()),unknown=int((yy[mask]==2).sum()),metrics=summary(a,pm.METRICS))
    if method=='H0':arrays[g,j]=a
    else:pdel[PIDS[j]]=summary(a-arrays[g,j],pm.METRICS)
   a=pm.mean(np.stack(vals),0);out[g]=dict(n=int(sel.sum()),macro=summary(a,pm.METRICS),per_predicate=by)
   if method=='H0':arrays[g,'macro']=a
   else:paired[method][g]=dict(macro=summary(a-arrays[g,'macro'],pm.METRICS),per_predicate=pdel)
  predresults[method]=out;status('predicate_statistics',method=method)
 put('predicate_metrics.json',dict(label_basis='model-generated, non-human',causal_main_records=len(labels),yes_no=int((yy<2).sum()),bootstrap=2000,methods=predresults,paired_vs_H0=paired,unknown='independent head; primary binary metrics retain all reference yes/no, no threshold from dev'))
 rr=read('dev_frozen_parent_statistics.json');rr={name:sorted(v,key=lambda r:r['parent']) for name,v in rr.items()};stage={};boots={};stagepaired={}
 for name,v in rr.items():
  pt=aggregate(v);bb=[aggregate([v[i] for i in ix]) for ix in samples];names=list(pt);a=np.array([[np.nan if row[k] is None else row[k] for k in names] for row in [pt]+bb]);boots[name]=(a,names)
  counts={tol:{key:dict(sum((collections.Counter(r['transitions'][tol][key]) for r in v),collections.Counter())) for key in ['miss','extra']} for tol in ['15','30']};stage[name]=dict(metrics=summary(a,names),transitions=counts,confusion=np.sum([r['cm'] for r in v],axis=0).tolist());status('stage_statistics',method=name)
 for name,(a,names) in boots.items():
  if name=='E101_S2':continue
  base=boots['E101_S2'][0];stagepaired[name]=dict(vs_E101_S2=summary(a-base,names),vs_H0_same_initial=summary(a-boots['H0__'+name.split('__')[1]][0],names))
 put('stage_metrics.json',dict(methods=stage,paired=stagepaired,bootstrap=2000,denominator='10-stage macro F1, typed one-to-one boundaries at inclusive +/-15/30 original frames; no boundary across unknown groundtruth gaps; nominal30Hz; errors and delays conditional on matched boundaries'))
 verdict={}
 for m in ['H2','H3']:
  initial=lock['preferred_initial'][m];name=m+'__'+initial;da=paired[m]['all']['macro']['AUROC'];dr=stagepaired[name]['vs_E101_S2']['recall30'];lo1=da['ci95'][0];lo2=dr['ci95'][0];rec=stage[name]['metrics']['recall30']['value'];verdict[m]=dict(training_selected_initial=initial,macro_AUROC_delta=da,recall30_delta_vs_E101=dr,recall30=rec,gap_to_point7=max(0,.7-rec),historical_information_effective=bool(lo1 is not None and lo2 is not None and lo1>0 and lo2>0))
 put('PREREGISTERED_VERDICT.json',dict(brt=now(),models=verdict,success=any(v['historical_information_effective'] for v in verdict.values()),E101_reference=stage['E101_S2']['metrics']['recall30']))
 fmt=lambda x:'NA' if x is None else f'{x:.4f}'
 def fi(a):return fmt(a['value'])+' ['+', '.join(fmt(v) for v in a['ci95'])+']'
 text='## 谓词层一次冻结结果\n**dev 真值均为模型生成，非人工。** 因果主表'+str(len(labels))+'条，yes/no '+str(int((yy<2).sum()))+'条；2000次共享parent bootstrap，宏只计可估计谓词。\n\n|方法|宏AUROC [95%CI]|二类宏F1|平衡准确率|相对H0 AUROC配对差|\n|---|---|---|---|---|\n'
 for m in predresults:
  z=predresults[m]['all']['macro'];text+='|'+m+'|'+fi(z['AUROC'])+'|'+fi(z['F1_macro'])+'|'+fi(z['balanced_accuracy'])+'|'+('基线' if m=='H0' else fi(paired[m]['all']['macro']['AUROC']))+'|\n'
 report(text+'\n按visibility、stage及每谓词完整CI与配对差见predicate_metrics.json。P073固定弃权；H0训练缺类别谓词用训练常数；不可估计不填0。H0沿用原锁定阈值，新头固定0.5，因此F1/BA比较含阈值协议差异，AUROC判定不依赖阈值。H2/H3相对H0也改变了头架构，H1是相同线性头的历史对照；本轮不能完全分离历史与模型容量因素。')
 group_text='## 可见性与阶段分组（因果主表）\n|分组|方法|AUROC [95%CI]|二类宏F1|平衡准确率|记录数|\n|---|---|---|---|---|---|\n'
 for group in groups:
  if group=='all':continue
  for method in ['H0','H1','H2','H3']:
   g=predresults[method][group];zz=g['macro'];group_text+='|'+group+'|'+method+'|'+fi(zz['AUROC'])+'|'+fmt(zz['F1_macro']['value'])+'|'+fmt(zz['balanced_accuracy']['value'])+'|'+str(g['n'])+'|\n'
 report(group_text+'\n分组AUROC不可估计时记NA；不把缺类或unknown计作no。')
 text='## 阶段层一次冻结结果\n|方法/初态|准确率|宏F1|P@15|R@15|F1@15|P@30|R@30|F1@30|\n|---|---|---|---|---|---|---|---|---|\n'
 for name,v in stage.items():text+='|'+name+'|'+'|'.join(fi(v['metrics'][k]) for k in ['accuracy','macro_f1','precision15','recall15','f1_15','precision30','recall30','f1_30'])+'|\n'
 report(text)
 text='|方法/初态|误差中位@30帧|误差P90@30帧|虚假切换/分钟@30|有符号延迟中位@30帧|非负延迟中位@30帧|\n|---|---|---|---|---|---|\n'
 for name,v in stage.items():text+='|'+name+'|'+'|'.join(fi(v['metrics'][k]) for k in ['error_median30','error_p9030','false_per_min30','signed_delay_median30','nonnegative_delay_median30'])+'|\n'
 report(text+'\n±15对应误差、延迟、虚假切换及相对H0/E101的配对差均见stage_metrics.json。误差/延迟只对匹配成功边界计算，漏检不填零。均匀初态由当前帧后验选起点，全程无前视。')
 report('## 预注册判定\n'+json.dumps(verdict,ensure_ascii=False,indent=2)+'\n结论依据训练锁定的初态变体；固定S01与均匀先验均完整报告，不按dev选初态。与E101的0.366比较使用同20 parent逐轨迹冻结预测的配对bootstrap，不把常数0.366当无误差真值。')
 text='## 漏检转移前后比较（±30）\n|方法/初态|S04→S05 漏/总|S05→S06 漏/总|S06→S07 漏/总|\n|---|---|---|---|\n'
 types=[(3,4),(4,5),(5,6)];tot={str(t):sum(sum((a,b)==t for f,a,b in boundaries(truths[s['parent']])) for s in ds) for t in types}
 for name,v in stage.items():text+='|'+name+'|'+'|'.join(str(v['transitions']['30']['miss'].get(str(t),0))+'/'+str(tot[str(t)]) for t in types)+'|\n'
 report(text+'\nS06/S07原定义：v1守卫按P062可见穿出，E98边界按换抓开始；本轮只记录，未改定义、未平移标签。新H0全60完整OOF调参，而E101只有12完整轨迹，因此H2/H3对E101改善混合了分数与训练覆盖/初态变化；应同时查看相对本轮H0、同初态的配对差。')
 deltas={}
 for method in ['H0','H1','H2','H3']:
  initial=lock['preferred_initial'][method];name=method+'__'+initial;deltas[name]={S[a]+'→'+S[b]:stage[name]['transitions']['30']['miss'].get(str((a,b)),0)-stage['E101_S2']['transitions']['30']['miss'].get(str((a,b)),0) for a,b in types}
 report('## 训练预选初态下漏检变化解释\n相对E101-S的漏检数变化（负数=改善，正数=恶化）：'+json.dumps(deltas,ensure_ascii=False)+'。本表只描述冻结结果，不据此更换模型或初态。S06/S07语义差异仍然存在，不能将所有边界误差归因于视觉时序能力。')
 # Read-only metadata and content audit; concurrent sessions' own dirs intentionally excluded.
 before=read('protected_before.json');changes=[]
 for path,v in before.items():
  p=Path(path)
  if not p.exists() or [p.stat().st_size,p.stat().st_mtime_ns]!=v:changes.append(path)
 hashchanges=[p for p,h in read('input_hashes.json').items() if sha(Path(p))!=h]
 sourcechanges=[]
 for p,v in oldread('source_stat_snapshot.json').items():
  st=Path(p).stat()
  if dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)!=v:sourcechanges.append(p)
 assert not changes and not hashchanges and not sourcechanges,(changes[:5],hashchanges,sourcechanges[:5])
 ledger=read('gpu_ledger.json');gpu=sum(v['seconds'] for v in ledger)+30*len(ledger);assert gpu<=7200;disk=sum(p.stat().st_size for d in [R,T/'code_e102s'] for p in d.rglob('*') if p.is_file())+REPORT.stat().st_size;assert disk+100_000_000<=20_000_000_000
 final=dict(status='COMPLETE',brt=now(),gpu_seconds_including_cleanup=gpu,gpuh=gpu/3600,new_disk_bytes=disk,upper_disk_bytes=disk+100_000_000,train_parents=60,dev_parents=20,full_frames=187641,dev_evaluations=1,protected_files=len(before),protected_changes=changes,input_hash_changes=hashchanges,source_metadata_changes=sourcechanges,verdict=verdict,limitations=['non-human model-generated dev labels','single seed; bootstrap does not cover training-seed uncertainty','H2/H3 vs H0 also changes head architecture; H1 is linear history control','H0 inherits original thresholds; new heads fixed .5','stacked crossfit auxiliary training is not fully nested','S06/S07 definitions unchanged','no deployment or physical success evaluation'])
 report_text=REPORT.read_text().replace('状态：执行中。dev 参考标签为模型标注，非人工。','状态：完成。dev 参考标签为模型标注，非人工。').replace('待完成。','已完成，见后续追加章节。').replace('尚未开始。','已一次冻结完成，见后续追加章节。')
 REPORT.write_text(report_text)
 put('FINAL_STATUS.json',final);report('## 资源、审计与交付 '+now()+'\n'+json.dumps(final,ensure_ascii=False)+'\n命令：ssh -o BatchMode=yes <gpu-server>；PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=T/code_e102s:T/code_e86 setsid nohup e29/bin/python -u code_e102s/{extract,driver,evaluate,summarize}.py。所有执行均在新目录，已有缓存原位只读；无下载/安装/大文件复制。前置本体长度错误两次CPU失败保留extract_01/02.log，均未进入GPU；后续失败按文件保留。')
 for path,text in [(T/'docs/experiment_log/EXPERIMENT_LOG.md','\n## E102-S 完成 | '+now()+'\n目标：因果历史谓词+S2，改动：全80逐帧SigLIP2、两态本体、H0-H3五折、双初态。命令：setsid nohup e29 python extract/driver/evaluate/summarize.py；目录code_e102s、results/e102s_20261115_01。结果/结论：'+json.dumps(verdict,ensure_ascii=False)+'；GPUh='+str(gpu/3600)+'，新增字节='+str(disk)+'。状态COMPLETE；详见notes/e102s_temporal_predicates_20261115.md。\n'),(T/'notes/worklog.md','\n- '+now()+' E102-S COMPLETE：H0-H3因果历史与S2双初态，60训练五折/20dev一次；详见e102s_temporal_predicates_20261115.md。\n')]:
  with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text);f.flush();fcntl.flock(f,fcntl.LOCK_UN)
 status('COMPLETE',gpuh=gpu/3600,new_disk_bytes=disk,history_effective=any(v['historical_information_effective'] for v in verdict.values()))
if __name__=='__main__':
 try:main()
 except Exception:put('SUMMARY_FAILURE_'+str(int(time.time()))+'.json',dict(brt=now(),traceback=traceback.format_exc()));raise
