import json
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e93_dev_diag_20261114_01'
assert json.loads((R/'diag_status.json').read_text())['status']=='completed'
changed=[]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2));changed.append(p.name)
for name in ('dev_grid_records.json','dev_grid_records_partial.json','dev_joint_records.json','train64_grid_records.json','train64_grid_records_partial.json','train64_joint_records.json','linear_records.json'):
 p=R/name
 if not p.exists():continue
 d=json.loads(p.read_text())
 for r in d:r.pop('linear',None)
 write(p,d)
def strip(a):
 if a.get('values'):
  for k in ('linear','skill_linear'):a['values'].pop(k,None);a['ci95'].pop(k,None)
for name in ('dev_grid_summary.json','train64_grid_summary.json'):
 p=R/name;d=json.loads(p.read_text())
 for hh in d.values():
  for a in hh.values():strip(a)
 write(p,d)
p=R/'linear_summary.json';d=json.loads(p.read_text())
for a in d.values():strip(a)
write(p,d)
p=R/'train_language_results.json';d=json.loads(p.read_text())
for a in d['original_instruction'].values():strip(a)
for r in d['records']:r.pop('linear',None)
write(p,d)
write(R/'metric_contract.json',dict(v1_linear='Only dev_v1 and train64_v1 records/summaries contain the actual causal velocity-extrapolation baseline.',diagnostic_and_ridge='Only model/copy metrics are reported; unused internal dummy linear fields removed. Core MSE/copy/skill/cosine values unchanged.',cleaned_files=changed))
print('metric schemas cleaned',len(changed))
