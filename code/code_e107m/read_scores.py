import json
from pathlib import Path
C=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e107m_control_20261116_01')
p=json.loads((C/'PAIRED.json').read_text());s=json.loads((C/'PROPER_SCORES.json').read_text());o=json.loads((C/'OPENLOOP.json').read_text())
for r in p:
 if r['subset']=='all27' and r['condition']=='no_traj':
  v=next(v for v in s if v['subset']=='all27' and v['condition']=='no_traj' and v['group']=='M1' and v['h']==r['h']);print(json.dumps(dict(h=r['h'],ES_gain=r['energy_improvement'],CRPS_gain=r['crps_improvement'],skill_gain=r['mean_skill_gain'],spread_skill=v['spread_skill'],coverage=v['coverage']),ensure_ascii=False))
print('OPENLOOP',json.dumps([x for x in o['paired'] if x['subset']=='all27'],ensure_ascii=False))
