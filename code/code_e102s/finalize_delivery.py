from common import *
import numpy as np,fcntl
assert read('FINAL_STATUS.json')['status']=='COMPLETE'
pm=read('predicate_metrics.json');sm=read('stage_metrics.json');lock=read('CONTROLLER_LOCK.json');verdict=read('PREREGISTERED_VERDICT.json');hl=read('HEADS_LOCK.json')
lines=['## 最终摘要','本轮预注册判定：**未通过“历史信息有效”**。所有 dev 参考标签均为模型生成，非人工；本轮结论只针对当前数据、头架构与训练方案。','60 train parent五折完整逐帧袋外、20 dev parent一次冻结评估；全187641帧。H2/H3训练侧均选1秒历史、stride3。','|方法|dev宏AUROC|训练预选初态|S2 recall@±30|距0.7差距|','|---|---|---|---|---|']
for m in ['H0','H1','H2','H3']:
 initial=lock['preferred_initial'][m];r=sm['methods'][m+'__'+initial]['metrics']['recall30']['value'];lines.append(f'|{m}|{pm["methods"][m]["all"]["macro"]["AUROC"]["value"]:.4f}|{initial}|{r:.4f}|{.7-r:.4f}|')
lines+=['','H2−H0宏AUROC=-0.1021，95%配对CI [-0.1483,-0.0673]；H3−H0=-0.0383，CI [-0.0719,-0.0049]。相对E101-S切换recall差：H2=-0.1009，CI [-0.1452,-0.0598]；H3=-0.1729，CI [-0.2265,-0.0903]。两项均未满足预注册条件。','补齐训练轨迹并重调H0后recall=0.4150，相对E101-S的0.3660提高0.0490，配对CI [0.0281,0.0690]。这项改善来自本轮完整训练轨迹调参/初态方案，H0谓词分数与旧E99 dev逐项最大差0；尚距0.7差0.2850。','固定S01与均匀初态的边界recall在每种头上完全相同；均匀先验未改善本轮边界召回，部分逐帧指标下降。保留训练预选结果，未按dev切换初态。','资源：GPU含清理余量0.4769h；新增落盘约1.08GB，保守上界低于1.18GB。25485个保护文件、标签哈希与源HDF/视频元数据零变化。指定实验已完成，无需用户决定；未进行人工标注或实机部署。','']
text=REPORT.read_text();assert '## 最终摘要' not in text;first,rest=text.split('\n',1);REPORT.write_text(first+'\n\n'+'\n'.join(lines)+'\n'+rest)
report('## 失败分析：训练—dev差异与转移\nH2训练袋外AUROC 0.8019→dev 0.6926；H3 0.8455→0.7564，H0则0.7861→0.7947。历史模型的训练排序收益没有迁移到因果dev主表。训练监督主要为回顾式模型答案、dev主表为首遍因果答案，存在语义差异；小样本共享时序主干的泛化、分数校准及架构容量变化也可能影响结果。本轮不能仅凭这些差值确定单一原因，更不能推断历史信息在所有方案中无效。\n\n相对E101-S的S04→S05/S05→S06/S06→S07漏检：39/33/17 → H2 38/43/25，H3 40/41/32。H2仅S04→S05少漏1个，后两项分别多漏10/8个；H3三项分别多漏1/8/15个。S06/S07原定义差异保持记录，未改定义或移动标签。\n\n本轮明确使用E99-B3方法b的冻结SigLIP2视觉合同，不计算世界O，故没有调用默认E100-C C2世界模型checkpoint；旧代码与ckpt均未修改。完成后的报告整理只读冻结指标，未重新运行dev。')
# Audit counts, folds and completeness from frozen artifacts without inference.
assert len(read('folds.json')['parent_to_fold'])==60
src=json.loads((T/'results/e99b3_20261115_01/sources.json').read_text());assert len(src)==80 and all(s['cid']!=EXCLUDE for s in src)
for s in src:
 if s['pool']=='train':
  for m in ['H0','H1','H2','H3']:
   z=np.load(R/(s['parent']+'_'+m+'_oof.npz'));assert z['prob'].shape==(s['n'],21) and np.isfinite(z['prob']).all()
 else:
  z=np.load(R/(s['parent']+'_dev_frozen_scores.npz'));assert all(z[m+'_prob'].shape==(s['n'],21) for m in ['H0','H1','H2','H3'])
assert read('DEV_FROZEN_COMPLETE.json')['evaluations']==1
before=read('protected_before.json');changed=[p for p,v in before.items() if not Path(p).exists() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v];assert not changed
hashchanges=[p for p,h in read('input_hashes.json').items() if sha(Path(p))!=h];assert not hashchanges
bytecount=sum(p.stat().st_size for d in [R,T/'code_e102s'] for p in d.rglob('*') if p.is_file())+REPORT.stat().st_size
put('DELIVERY_AUDIT.json',dict(brt=now(),status='COMPLETE',gpu_hours=read('FINAL_STATUS.json')['gpuh'],measured_new_bytes=bytecount,conservative_upper_bytes=bytecount+100_000_000,complete_train_oof_trajectories=240,dev_parents=20,dev_evaluations=1,protected_changes=changed,input_hash_changes=hashchanges,report=str(REPORT),postfreeze_actions='read frozen statistics and append interpretation only; no model, threshold or dev sequence rerun'))
with (T/'notes/worklog.md').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);f.write('\n- '+now()+' E102-S交付核验：预注册未通过，H0/H1/H2/H3 dev AUROC=.7947/.7724/.6926/.7564；训练预选初态S2 R@30=.4150/.3314/.2651/.1931；GPU .4769h，新增约1.08GB，无遗留执行项。\n');f.flush();fcntl.flock(f,fcntl.LOCK_UN)
print(dict(status='COMPLETE',measured_new_bytes=bytecount,upper_bytes=bytecount+100_000_000,protected_files=len(before)))
