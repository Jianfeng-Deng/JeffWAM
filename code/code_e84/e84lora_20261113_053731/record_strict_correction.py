from common import *
import fcntl
oldP=R/'probes'/PILOT
if not (oldP/'superseded_logged.json').exists():
 text=f'\n\n## E84-P {PILOT} — E83同源审计后作废主结论（{brt()} BRT）\n- 目标：LoRA真实/打乱同步数对照。\n- 改动/命令：双塔末4层Q/V rank8、各768步；launch.py driver_gpu.py；严格审计e83_pair_audit.py。\n- 目录：{R/"lora"/PILOT}、{oldP}、{T/"code_e84"/PILOT}。\n- 结果：331 E83配对仅11条同锚点，320条近邻≤15帧；为严格执行不平移标签，旧adapter/指标只诊断，不作为正式证据；GPU{PRIOR_LORA_GPU_HOURS:.6f}h均记入本LoRA阶段4h预算。仅停止自己的driver/extract/等待进程，未OOM、未动他人进程/原权重。\n- 结论：不按该先导的val分数挑checkpoint；新run {RUN}仅保留11条同源E83，FK1483对/原selection/split不变，重新固定768步真/乱两组。\n- 状态：superseded_E83_nearest_pairs_not_primary；所有旧结果保留。\n'
 for path,value in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',text),(T/'notes/worklog.md',f'\n- {brt()} BRT `{PILOT}`：E83近邻配对不作正式结论，严格仅留11同锚点caption，转 `{RUN}`；先导GPU{PRIOR_LORA_GPU_HOURS:.6f}h计入4h总预算，旧结果全保留。\n')]:
  with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(value);f.flush()
 with (T/'notes'/f'e84_P_{PILOT}.md').open('a') as f:f.write('\n## 同源审计修正\n\n'+text)
 atomic(oldP/'superseded_logged.json',dict(timestamp_brt=brt(),next_run=RUN,prior_lora_gpu_hours=PRIOR_LORA_GPU_HOURS))
p=P/'e83_training_pair_provenance.json';j=json.loads(p.read_text());assert j['pairs']==j['exact']==11;j['note']='All11 E83 positives retain original source_frame_index==locked anchor_source_row. No nearest-frame reassignment. Actual camera rows still follow original E82 causal exposure/receipt contract. Original action/FK/pixel labels are unchanged.';atomic(p,j)
print('strict_correction_logged')
