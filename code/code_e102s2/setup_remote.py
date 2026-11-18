from pathlib import Path
import json,datetime,hashlib,fcntl
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e102s_20261115_01';C=T/'code_e102s'
old=T/'results/e99b3_20261115_01';sources=json.loads((old/'sources.json').read_text());n=sum(s['n'] for s in sources)
meta={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for d in ['code_e82','code_e86','code_e88','code_e99b3','code_e101s','results/e99b3_20261115_01','results/e101s_20261115_01','data/e97b_stages_v1','data/e98_predicate_labels_train','data/e98_predicate_labels_dev'] for p in (T/d).rglob('*') if p.is_file()}
(R/'protected_before.json').write_text(json.dumps(meta))
inputs=[T/'data/e97b_stages_v1/stages_predicates_v1.json',*[T/f'data/e98_predicate_labels_{pool}/{fn}.jsonl' for pool in ['train','dev'] for fn in ['predicates','stages']]]
(R/'input_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}))
est=dict(train_frames=sum(s['n'] for s in sources if s['pool']=='train'),dev_frames=sum(s['n'] for s in sources if s['pool']=='dev'),pooled_visual_bytes=n*2304*2,proprio_bytes=n*33*4,estimated_total_upper_bytes=6_000_000_000,raw_grid_not_copied=True,gpu_limit_seconds=7200)
(R/'RESOURCE_ESTIMATE.json').write_text(json.dumps(est,indent=2))
with (T/'notes/e102s_temporal_predicates_20261115.md').open('a') as f:f.write('\n## 特征与资源事前估算\n'+json.dumps(est,ensure_ascii=False)+'\n视觉全量约0.865GB，本体约0.025GB；临时内存序列不落盘。总新增保守预留6GB，低于20GB。方法b仍按原fp16网格转float32求均值再转fp16。\n')
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
for path,text in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',f'\n## E102-S | {now}\n目标：60 train/20 dev因果历史谓词，H0/H1/H2/H3 + E101-S S2；目录code_e102s、results/e102s_20261115_01；命令ssh BatchMode + setsid nohup e29/bin/python pipeline.py。状态：执行中，dev未开始。改动/结果/结论见notes/e102s_temporal_predicates_20261115.md。\n'),(T/'notes/worklog.md',f'\n- {now} E102-S 启动：因果历史谓词、全60训练轨迹袋外与S2；run=e102s_20261115_01，dev未开始。\n')]:
 with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text);f.flush();fcntl.flock(f,fcntl.LOCK_UN)
common=(T/'code_e99b3/common.py').read_text().replace("R=T/'results/e99b3_20261115_01'; REPORT=T/'notes/e99b3_predicates_20261115.md'","R=T/'results/e102s_20261115_01'; REPORT=T/'notes/e102s_temporal_predicates_20261115.md'").replace('10800','7140').replace('10700','7020').replace('10750','7080')
(C/'common.py').write_text(common)
(C/'linear.py').write_text((T/'code_e99b3/linear.py').read_text())
print(est)
