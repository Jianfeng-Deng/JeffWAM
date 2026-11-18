import json,pathlib,hashlib
from common import *
from contract import contracts,sha
state=json.loads((P/'progress.json').read_text());path=R/'READY_fk_gt.json';payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest();ready=json.loads(payload)
split,_,_=contracts();assert ready['dev_test_data_read'] is False
assert ready['probe_split']['sha256']==sha(R/'split/probe_split.json')
for r in ready['files']:assert sha(r['path'])==r['sha256'],r['path']
manifest=R/'fk/ARTIFACT_SHA256.jsonl';entries={r['path']:r for r in map(json.loads,manifest.open())}
eligibility=R/'fk/ELIGIBILITY.json';assert str(eligibility) in entries;assert sha(eligibility)==entries[str(eligibility)]['sha256']
eligible=json.loads(eligibility.read_text());assert set(eligible['excluded_parents']).issubset(split['parents'])
assert sha(path)==digest,'receipt changed during acceptance; stop'
snap=P/('READY_fk_gt_accepted_'+digest[:16]+'.json')
if not snap.exists():snap.write_bytes(payload)
old=state.get('fk_gt_ready');update(fk_gt_ready={'path':str(path),'sha256':digest,'accepted_snapshot':str(snap),'detected_brt':brt()},gt_eligibility=eligible)
append_report('## FK回执变更接收（任何模型结果前）\n\n'+brt()+' 旧回执 '+str(old)+'；新回执 SHA256='+digest+'。入口文件及资格规则全部哈希核验通过，probe split未改变；新增明确排除诊断parent '+str(eligible['excluded_parents'])+'。原主表诊断712行仅审计，不训练/评分。保存不可变接收快照 `'+str(snap)+'`。')
print(json.dumps({'accepted_sha256':digest,'eligibility':eligible},ensure_ascii=False))
