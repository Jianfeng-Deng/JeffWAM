from e98w_evaluate import aggregate,put
import torch,json
records=[]
for p,n,copy,values in [('p',100,1.,{'E96a':.8,'A':.6,'B':.5}),('q',1,9.,{'E96a':8.,'A':7.,'B':7.5})]:
 for i in range(n):
  for m,error in values.items():records.append(dict(pool='synthetic',h=25,camera='all',section='full',method=m,parent=p,index=i,model=error,copy=copy,cosine=0.))
s,paired=aggregate(records);z=paired['synthetic|25|all|full|A-E96a'];assert abs(z['delta_skill']-.12)<1e-12
# Identical predictions must produce an exactly zero paired interval.
for r in records:r['model']={'p':.8,'q':8.}[r['parent']]
s,p=aggregate(records);assert all(v['delta_skill']==0 and v['ci95']==[0.,0.] for v in p.values());assert not torch.cuda.is_initialized()
put('evaluation_contracts.json',dict(status='passed',parent_weighting_imbalanced_100_to_1_delta=.12,identical_predictions_pair_ci=[0.,0.],bootstrap=2000,seed=90,no_dev=True,GPUh=0))
print('evaluation contracts passed; parent imbalance 100:1; zero paired CI; no GPU/dev')
