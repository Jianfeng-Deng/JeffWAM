from common import *
import collections
ready=json.loads((L/'READY_training_data.json').read_text());items=json.loads((L/'training_items.json').read_text());n=len({r['parent'] for r in items});assert n==ready['parents']==189
sel=json.loads((P/'selection_full_v4.json').read_text());coh={s['canonical_id']:s['cohort'] for s in sel['samples']};pc=collections.defaultdict(set)
for r in items:pc[r['parent']].add(coh[r['canonical_id']])
atomic(P/'training_parent_deduplication.json',dict(timestamp_brt=brt(),unique_train_parents=n,cross_cohort_parents={p:sorted(c) for p,c in pc.items() if len(c)>1},source='training_items.json; exact parent dedup, no relabel/sample change'))
p=C/'finalize.py';s=p.read_text().replace('训练1483FK+11同锚点E83/191parent','训练1483FK+11同锚点E83/189parent');p.write_text(s)
p=C/'record_training.py';s=p.read_text().replace('191个train parent','189个train parent');p.write_text(s)
p=C/'report.py';s=p.read_text().replace('按parent均衡并不等于按方向类别均衡，native/early训练parent仅9/8，外推受限。','各分层parent可能交叉，不可直接加总；去重共189parent，详见training_parent_deduplication.json。按parent均衡并不等于按方向类别均衡，native/early训练parent仅9/8，外推受限。');p.write_text(s)
append_report('## 训练parent去重计数更正\n\n'+brt()+' 正式训练库去重189个parent（READY_training_data与训练manifest原本即为189）；分层计数存在交叉，不能把FK分层113+61+9+8直接当全局191。此前训练小节191为文字计数错误，现统一按189；样本、模型、指标与CI均未改变。证据training_parent_deduplication.json。')
update(actual_unique_lora_train_parents=189)
print('train_parent_count',n,'cross_cohort',sum(len(c)>1 for c in pc.values()))
