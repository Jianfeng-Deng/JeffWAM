from common import *
for v in ['lora','shuffled']:
 root=P/v
 if (root/'pixel_results.json').exists():
  a=json.loads((root/'pixel_results.json').read_text());print('PIXEL',v)
  for r in a['summaries']:
   if r['split']=='probe-val' and r['quality']=='main' and r['radius_multiplier']==1 and r['condition']=='correct':print(r['word'],'point',r['pointing'],'auc',r['auroc'])
  print('COLOR',[(r['condition'],r['correct_minus_control_pointing'],r['correct_minus_control_auroc']) for r in a['summaries'] if r['split']=='probe-val' and r['quality']=='main' and r['radius_multiplier']==1 and r['word']=='orange zip tie' and r['condition'] in ['blue','green']])
  print('LR',[r for r in a['left_right_switch'] if r['split']=='probe-val' and r['scope']=='gripper' and r['radius_multiplier']==1 and r['disjoint_targets']])
 for arm in ['left','right']:
  p=root/f'motion_robot_{arm}_h50.json'
  if p.exists():
   a=json.loads(p.read_text());print('MOTION',v,arm,'zero',a['zero_shot']['zh']['difference']['direction_top1'],'linear',a['learned']['linear']['metrics']['direction_top1'],'contrastive',a['learned']['contrastive_zh']['metrics']['direction_top1'])
  p=root/f'behavior_{arm}.json'
  if p.exists():
   a=json.loads(p.read_text());print('BEHAVIOR',v,arm,a['counterfactual']['language'])
