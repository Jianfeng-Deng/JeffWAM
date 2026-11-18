from e98w_evaluate import get_data,augment,put
import torch
tr,dev=get_data();receipt={}
for pool,data in [('train64',augment(tr)),('dev',augment(dev))]:
 counts={h:0 for h in (10,25,50,100)};parents={h:set() for h in counts}
 for i,row in enumerate(data.rows):
  b=data.get(i,(10,25,50,100),'short','P1')
  for h in counts:
   if b['world_valid'][b['world_horizons']==h].all():counts[h]+=1;parents[h].add(row['parent_component'])
 receipt[pool]={h:dict(windows=counts[h],parents=len(parents[h])) for h in counts}
assert not torch.cuda.is_initialized();put('evaluation_preflight.json',dict(status='passed',coverage=receipt,no_forward=True,no_GPU=True))
print(receipt)
