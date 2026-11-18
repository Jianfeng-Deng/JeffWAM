from evaluate import *
import fcntl
metric=json.loads((R/'metrics.json').read_text());sm=metric['models'];status0=json.loads((R/'FINAL_STATUS.json').read_text());lock=json.loads((R/'CONTROLLER_LOCK.json').read_text());a=sm['S2']['point'];ci=sm['S2']['ci95']
assert status0['dev_evaluations']==1 and not status0['deploy_ready']
assert len(json.loads((R/'OOF_COMPLETE.json').read_text())['audit'])==10
assert all(set(k['fit_parents']).isdisjoint(k['heldout_parents']) for k in json.loads((R/'OOF_COMPLETE.json').read_text())['audit'])
assert json.loads((R/'online_interface_contract.json').read_text())['batch_online_exact']
assert len(list(R.glob('*_frozen_sequences.npz')))==20
# No dev controller or inference is executed here.
before=json.loads((R/'protected_snapshot_before.json').read_text());after=snapshot();assert before==after
for p,h in json.loads((R/'input_hashes_before.json').read_text()).items():assert sha(Path(p))==h
for s in ['15','30']:
 print('S2 tolerance',s,{k:v for k,v in a.items() if k in ['precision'+s,'recall'+s,'f1_'+s,'error_median'+s,'error_p90'+s,'signed_delay_median'+s]})
summary=f'''\n## 最终摘要\n完成 S0原样回放、S1因果迟滞、S2因果HMM和S3离线Viterbi；60 train parent五折，12条完整袋外训练轨迹调参；20 dev parent、55,754帧一次冻结评估、parent bootstrap 2000次。dev参考为模型标注，非人工。\n\n训练预选为 S2+b，温度0.1、最短驻留45帧；dev准确率{a['accuracy']:.4%}、宏F1={a['macro_f1']:.4f}、recall@±30={a['recall30']:.4%} [95%CI {ci['recall30'][0]:.4%}, {ci['recall30'][1]:.4%}]、F1@±30={a['f1_30']:.4f}。边界匹配127/347，仍漏检220个。虚假切换S0→S2：{sm['S0']['point']['false_per_min30']:.2f}→{a['false_per_min30']:.2f}次/分钟。S3离线recall@30={sm['S3']['point']['recall30']:.4%}，不是部署结果。\n\n推荐S2为后续因果控制器研究基线，**不建议直接实机部署**；recall距0.7还差{(.7-a['recall30'])*100:.2f}个百分点。优先补因果谓词监督/区分度、处理截断初态，并统一S06/S07边界定义。GPUh=0。其余48训练parent没有重提全帧特征；未跑可选融合，普通HMM未扩展为HSMM。\n'''
text=REPORT.read_text();title,rest=text.split('\n',1);REPORT.write_text(title+'\n'+summary+rest)
report('## 最终交付核验 '+now()+'\n新增 `online_controller.py` 提供逐帧 `step(prob21)` / `reset()` CPU接口；与冻结批量因果HMM在3001训练帧逐项完全一致，reset后300帧前缀一致。统计合同核验一一匹配、容差包含端点、方向阶段对必须一致、unknown参考间隙不生成边界、BATCH_WAIT计错。3张带图例时间线已生成并检查布局，未重新评dev。\n\n本轮按任务要求复用E99-B3的b/d缓存和原特征定义，因此没有用默认E100-C C2 checkpoint重算世界O；E100-C代码/ckpt只读。S2当前选择b，不依赖世界O分数。首段CPU启动映射异常和后处理制图异常均保留；制图恢复只读冻结产物，未再次评dev。无需用户决定；若扩展到全60训练轨迹/人工边界验收，应另立预算。')
readme='''# E101-S\n\n冻结研究结果：../results/e101s_20261115_01/，报告：../notes/e101s_stage_controller_20261115.md。\n\n- run.py fit：60-parent 5折CPU谓词头与训练调参，已有锁定产物时拒绝重跑。\n- evaluate.py：dev一次评估，DEV_ONCE_STARTED.json存在即拒绝重跑。原评估在制图步骤报错；finish_frozen.py仅从冻结统计/序列恢复后处理，不调用dev控制器。\n- online_controller.py：锁定S2+b的CPU逐帧接口；输入为E99-B3 b的21维当前帧谓词概率，顺序与旧spec一致，P073索引13固定弃权。每个新parent/cycle的初始化策略应自行定义；当前reset固定S01，截断片段可能失败。\n\n```python\nfrom online_controller import OnlineStageController\nc = OnlineStageController('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e101s_20261115_01')\nresult = c.step(prob21)  # 当前帧概率，不能输入未来分数或标签\nprint(result['stage'])\n```\n\n因果研究候选，未达到recall@30>=0.7，不是实机部署许可。dev参考标签来自模型，非人工。仅12/60训练parent具有完整袋外轨迹用于控制器调参，60条标注点均有袋外分数。普通HMM自环/最短驻留不等于显式HSMM。S3为非因果附录。\n'''
(T/'code_e101s/README.md').write_text(readme)
# Save final actual disk use and conservative reserve for small external caches, local QA copies and shared-log lines.
disk=sum(p.stat().st_size for d in [R,T/'code_e101s'] for p in d.rglob('*') if p.is_file())+REPORT.stat().st_size
status0.update(finalized_brt=now(),new_disk_bytes=disk,new_disk_upper_bytes=disk+100_000_000,online_interface_verified=True,dev_rerun=False,processes_finished=True,protected_metadata_files=len(before))
write('FINAL_STATUS.json',status0)
report('最终新增自有目录/报告实计 '+str(disk)+' 字节；加100MB保守预留后 '+str(disk+100_000_000)+' 字节（包含小缓存、共享日志与本地QA副本余量），小于5GB。保护快照'+str(len(before))+'文件零变化，5个输入SHA256零变化。')
entry=f'''\n## E101-S | {now()} | COMPLETE_WITH_DECLARED_LIMITATIONS\n- 目标：Piper阶段控制器时间平滑，替代E99-B3 B4硬切换失败回放。\n- 改动：code_e101s独立实现S0/S1/S2/S3、60parent五折CPU谓词头、在线接口、冻结统计与3例时间线；仅12训练parent全帧OOF轨迹用于调参。\n- 命令：ssh -o BatchMode=yes <gpu-server>；CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 setsid nohup e29/bin/python -u code_e101s/run.py fit；evaluate.py；制图异常后finish_frozen.py只读冻结产物恢复；check_online.py；finalize.py。\n- 目录：code_e101s/；results/e101s_20261115_01/；notes/e101s_stage_controller_20261115.md。\n- 结果：dev20parent/55754帧一次评估、bootstrap2000；S2+b acc={a['accuracy']:.6f}, macroF1={a['macro_f1']:.6f}, recall30={a['recall30']:.6f}, F1_30={a['f1_30']:.6f}, matched30=127/347, false/min={a['false_per_min30']:.6f}；S0 false/min={sm['S0']['point']['false_per_min30']:.6f}；S3离线recall30={sm['S3']['point']['recall30']:.6f}。\n- 结论：S2为因果研究推荐但recall低于0.7，不可直接实机部署；优先改善谓词监督、截断初态与S06/S07定义。dev真值为用户许可的非人工模型标签。无dev选型/重跑。\n- 资源/审计：GPUh=0；新增实计{disk}字节，保守上界{disk+100_000_000}字节；{len(before)}个保护文件元数据零变化、输入SHA零变化。旧目录不写、无大文件复制/下载、未碰他人进程。\n- 未完成/限制：其余48训练parent全帧特征未重提；未做可选融合或显式HSMM；无视频复审/人工真值/实机验收。CPU启动与制图异常日志保留。状态：COMPLETE_WITH_DECLARED_LIMITATIONS；无需用户决定。\n'''
for path,txt in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',entry),(T/'notes/worklog.md',f"\n- {now()} BRT | E101-S 完成（已披露限制）：S2+b因果HMM，dev acc63.56%/macroF1 .558/recall@30 36.60% [27.69,43.29]，未达0.7；60parent五折、仅12全帧训练轨迹调参；dev20一次+bootstrap2000；GPU0h，新增<0.12GB；报告notes/e101s_stage_controller_20261115.md，无需决策。\n")]:
 with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(txt);f.flush();os.fsync(f.fileno())
manifest={str(p.relative_to(T)):dict(bytes=p.stat().st_size,sha256=sha(p)) for d in [R,T/'code_e101s'] for p in d.rglob('*') if p.is_file() and p.name not in ['delivery_manifest.json','status.json']}
write('delivery_manifest.json',dict(brt=now(),files=manifest,report_sha256=sha(REPORT),dev_evaluations=1))
status('COMPLETE_WITH_DECLARED_LIMITATIONS',gpu_hours=0,new_disk_upper_bytes=disk+100_000_000,report=str(REPORT))
