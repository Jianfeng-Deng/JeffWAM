from common import *
from contract import sha
source=T/'code_e84/e84p_20261112_153338/run_all.py'
results=[]
for arm in ['left','right']:
    p=P/'frozen'/f'motion_robot_{arm}_h50.json';r=json.loads(p.read_text())
    z=r['zero_shot']['zh']['difference']['direction_top1'];l=r['learned']['linear']['metrics']['direction_top1'];b=r['baseline']['random']['direction_top1']
    yes=z['estimate']<=.25 and l['ci95'][0]>b['ci95'][1]
    results.append({'arm':arm,'rule_satisfied':bool(yes),'zero_direction':z,'linear_direction':l,'random_direction':b})
atomic(P/'lora_trigger_evidence.json',{'timestamp_brt':brt(),'legacy_rule_source':str(source),'legacy_rule_source_sha256':sha(source),'rule':'robot H50 ZH difference zero direction<=25%; linear direction CI lower>random direction CI upper; no pixel truth trigger','legacy_rule_satisfied':any(r['rule_satisfied'] for r in results),'arms':results,'pixel_laterality':'inconclusive_7_disjoint_validation_parents','execution':'NOT RUN; latest user revision2 requires explicit confirmation even if legacy condition holds','auto_launch':False})
print(json.dumps({'legacy_rule_satisfied':any(r['rule_satisfied'] for r in results),'arms':[(r['arm'],r['rule_satisfied']) for r in results],'lora_executed':False}))
