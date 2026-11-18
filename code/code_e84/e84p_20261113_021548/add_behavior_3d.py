import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,numpy as np
from common import *
from metrics import normalize,parent_ci
root=P/'frozen'
requested=np.stack([np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4),np.zeros(8)],-1)
for arm in ['left','right']:
 p=root/f'behavior_{arm}.json';obj=json.loads(p.read_text());f=np.load(root/f'behavior_{arm}_predictions.npz');truth=f['true_displacement'];parents=f['parents'];pp=np.repeat(parents,8);out={}
 for condition in ['language','no_language','shuffled_language_training']:
  bank=f['cf_'+condition];good=np.linalg.norm(bank[:,:,:2],axis=-1)>.001
  reqcos=(normalize(bank)*requested[None]).sum(-1);truecos=(normalize(bank)*normalize(truth)[:,None]).sum(-1)
  out[condition]={'prediction_instruction_cosine_3d':parent_ci(np.where(good,reqcos,np.nan).ravel(),pp),'prediction_true_cosine_3d':parent_ci(np.where(good,truecos,np.nan).ravel(),pp),'output_3d_norm_m':parent_ci(np.linalg.norm(bank,axis=-1).ravel(),pp),'note':'Requests lie in XY plane (zero z), model still outputs3D. Primary follow criterion remains XY>1mm plus same45deg sector.'}
 obj['counterfactual_3d']=out;atomic(p,obj)
print('BEHAVIOR_3D_COSINES_READY')
