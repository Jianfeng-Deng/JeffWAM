import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,numpy as np
from common import *
from metrics import paired_ci
root=P/'frozen'
for path in sorted(root.glob('motion_robot_*.json')):
    obj=json.loads(path.read_text())
    if obj.get('status')!='completed':continue
    f=np.load(path.with_name(path.stem+'_predictions.npz'));y=f['labels'];parents=f['parents'];random=f['random'];majority=f['majority'];gap={}
    for key in ['zero_difference_zh','linear','contrastive_zh']:
        p=f[key];gap[key]={}
        for name,b in [('random',random),('majority',majority)]:
            gap[key]['top1_vs_'+name]=paired_ci(p.argmax(-1)==y,b.argmax(-1)==y,parents)
            gap[key]['direction_vs_'+name]=paired_ci(p.argmax(-1)//3==y//3,b.argmax(-1)//3==y//3,parents)
            gap[key]['top3_vs_'+name]=paired_ci((np.argsort(p,axis=-1)[:,-3:]==y[:,None]).any(-1),(np.argsort(b,axis=-1)[:,-3:]==y[:,None]).any(-1),parents)
    obj['paired_gaps']=gap;atomic(path,obj)
print('MOTION_PAIRED_GAPS_READY')
