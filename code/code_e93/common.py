import os,sys,json,time,collections
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e93_dev_diag_20261114_01';sys.path.insert(0,str(T/'code_e90'))
import numpy as np,torch
from e86.common import atomic,sha,brt
from e90.data import load_data,Data
from e90.model import WorldModel
from e90.train import configure
from e86.data import collate,move
from evaluate_cpu_tail import aggregate,record_identity
H=(10,25,50,100);CK=T/'code_e90/artifacts/e90_p1_20261114_01/ckpt_05000.pt'
def get_data():
 p,ready,train,old,_=load_data();caps=json.loads((R/'dev/captures_full.json').read_text());rows=json.loads((R/'dev/windows_full.json').read_text());stats=train.base.stats
 dev=Data(rows,caps,R/'features',stats,{'scale':train.norm},{'__human_default__':old.base.text_cache['__human_default__']})
 by=collections.defaultdict(list)
 for i,r in enumerate(train.rows):by[r['parent_component']].append(i)
 parents=sorted(by);rng=np.random.default_rng(93);chosen=sorted(rng.choice(parents,64,replace=False).tolist());selected=[]
 for parent in chosen:
  ids=by[parent];selected.extend(ids[j] for j in np.linspace(0,len(ids)-1,min(8,len(ids))).astype(int))
 tr=Data([train.rows[i] for i in selected],p.captures,train.base.cache,stats,{'scale':train.norm},{'__human_default__':old.base.text_cache['__human_default__']})
 assert not set(chosen)&{r['parent_component'] for r in dev.rows};atomic(R/'train64_selection.json',dict(seed=93,parents=chosen,windows=[dict(canonical_id=r['canonical_id'],anchor=r['anchor'],parent=r['parent_component']) for r in tr.rows],language='fixed human short instruction',selection='64 parents random without replacement, up to8 evenly spaced original train anchors each'))
 return train,tr,dev

def model_cpu():
 configure();m=WorldModel();d=torch.load(CK,map_location='cpu',weights_only=False);assert d['step']==5000;assert sha(CK)=='45164cfb86483709892d5c59a55fb830484cb5923eb7c018579595c826abbc49';m.load_state_dict(d['model']);m.eval();return m

def rec(row,h,pred,b,**extra):
 y=b['world'][0];d=(pred-b['world_current'])[0]/b['world_scale'][0];err=(d-y).square();cp=y.square();nt=8 if h<=50 else 1
 cos=float(torch.nn.functional.cosine_similarity(d.flatten(),y.flatten(),dim=0));rawcos=float(torch.nn.functional.cosine_similarity((pred-b['world_current']).flatten(),(b['world_raw']-b['world_current']).flatten(),dim=0))
 return dict(**record_identity(row),canonical_id=row['canonical_id'],anchor=row['anchor'],h=h,model=float(err.mean()),copy=float(cp.mean()),cosine=cos,cosine_raw=rawcos,model_camera=err.reshape(3,nt,768).mean((1,2)).tolist(),copy_camera=cp.reshape(3,nt,768).mean((1,2)).tolist(),model_token=err.mean(1).tolist(),copy_token=cp.mean(1).tolist(),change=float(cp.mean()),prediction_energy=float(d.square().mean()),prediction_truth_dot=float((d*y).mean()),**extra)

def summary(records,keys=('model','copy','linear','cosine')):
 result={};has_linear=all('linear' in r for r in records)
 for h in H:
  rr=[r if 'linear' in r else dict(r,linear=r['copy']) for r in records if r['h']==h]
  a=aggregate(rr,list(keys),resamples=2000)
  if not has_linear and a['values'] is not None:
   for k in ('linear','skill_linear'):a['values'].pop(k,None);a['ci95'].pop(k,None)
  result[str(h)]=a
 return result
