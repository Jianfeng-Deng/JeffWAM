from common import *
for v in ['lora','shuffled']:
 for arm in ['left','right']:
  p=P/v/f'motion_robot_{arm}_h50.json'
  if p.exists():
   a=json.loads(p.read_text());print(v,arm,{k:m for k,m in [('zero_dir',a['zero_shot']['zh']['difference']['direction_top1']),('linear_dir',a['learned']['linear']['metrics']['direction_top1']),('contrast_dir',a['learned']['contrastive_zh']['metrics']['direction_top1'])]})
  p=P/v/f'behavior_{arm}.json'
  if p.exists():
   a=json.loads(p.read_text());print('BEHAVIOR',v,arm,a['counterfactual']['language']['follow_rate'],a['counterfactual']['language']['prediction_instruction_cosine'])
