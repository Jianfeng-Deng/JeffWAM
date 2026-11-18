"""Fixed5000 endpoints only. E93 lists/noise/bootstrap, all caches read-only."""
import os,json,time,collections,traceback,argparse
from pathlib import Path
import numpy as np
import torch
from e94a.common import T,ART,atomic,brt,sha
from e94a.model import WorldModel
from e94a.train import configure
from e90.data import Data,load_data
from e86.data import move
from evaluate_cpu_tail import aggregate,record_identity
H=(10,25,50,100);E93=T/'results/e93_dev_diag_20261114_01'

def get_data():
 p,ready,train,old,_=load_data();stats=train.base.stats
 caps=json.loads((E93/'dev/captures_full.json').read_text());rows=json.loads((E93/'dev/windows_full.json').read_text())
 text={'__human_default__':old.base.text_cache['__human_default__']}
 dev=Data(rows,caps,E93/'features',stats,{'scale':train.norm},text)
 sel=json.loads((E93/'train64_selection.json').read_text());by={(r['canonical_id'],r['anchor'],r['parent_component']):r for r in train.rows}
 rows=[by[w['canonical_id'],w['anchor'],w['parent']] for w in sel['windows']]
 tr=Data(rows,p.captures,train.base.cache,stats,{'scale':train.norm},text)
 assert len(dev.rows)==211 and len(tr.rows)==507
 return tr,dev

def rec(row,h,pred,b,reference,index):
 y=b['world'][0];d=(pred-b['world_current'])[0]/b['world_scale'][0];err=(d-y).square();cp=y.square();nt=8 if h<=50 else 1
 cos=float(torch.nn.functional.cosine_similarity(d.flatten(),y.flatten(),dim=0))
 r=dict(**record_identity(row),canonical_id=row['canonical_id'],anchor=row['anchor'],h=h,index=index,model=float(err.mean()),copy=float(cp.mean()),linear=reference['linear'],cosine=cos,model_camera=err.reshape(3,nt,768).mean((1,2)).tolist(),copy_camera=cp.reshape(3,nt,768).mean((1,2)).tolist(),change=float(cp.mean()))
 assert r['parent']==reference['parent'] and abs(r['copy']-reference['copy'])<1e-5
 return r

def summary(rs):return {str(h):aggregate([r for r in rs if r['h']==h],['model','copy','linear','cosine'],2000) for h in H}

def decompose(rs):
 result={}
 for h in H:
  rr=[r for r in rs if r['h']==h];q=np.quantile([r['change'] for r in rr],[.25,.5,.75]);out={}
  for j,cam in enumerate(('top','left_wrist','right_wrist')):
   cr=[dict(r,model=r['model_camera'][j],copy=r['copy_camera'][j]) for r in rr]
   a=aggregate(cr,['model','copy','linear'],2000)
   # Extrapolation reference is only global; do not label it as camera-specific.
   for k in ('linear','skill_linear'):a['values'].pop(k,None);a['ci95'].pop(k,None)
   out[cam]=a
  for k in range(4):out['change_Q'+str(k+1)]=aggregate([r for r in rr if np.searchsorted(q,r['change'],side='right')==k],['model','copy','linear','cosine'],2000)
  result[str(h)]=dict(change_quartile_bounds=q.tolist(),groups=out)
 return result

def repeat(b,n):return {k:v.repeat((n,)+(1,)*(v.ndim-1)) if isinstance(v,torch.Tensor) and k!='world_positions' else v for k,v in b.items()}

def run(recipe,runid):
 guard=None;dest=T/'results'/runid
 assert dest.exists() and not (dest/'evaluation_status.json').exists()
 try:
  cfg=json.loads((dest/'config.json').read_text());cp=dest/'ckpt_05000.pt';assert cfg['recipe']==recipe
  d=torch.load(cp,map_location='cpu',weights_only=False);assert d['step']==5000
  configure();tr,dev=get_data()
  from e94a.resource import Guard
  guard=Guard('E94a_eval_'+recipe,3000);m=WorldModel(recipe=recipe).to('cuda');m.load_state_dict(d['model']);m.eval()
  with torch.inference_mode():
   for pool,data in [('train64',tr),('dev',dev)]:
    refs=json.loads((E93/(pool+'_v1_records.json')).read_text());lookup={(r['index'],r['h']):r for r in refs};rs=[]
    for i,row in enumerate(data.rows):
     guard.check()
     for h in H:
      if (i,h) not in lookup:continue
      cpu=data.batch([i],(h,),'short','P1');b=move(cpu,'cuda');pred=m.mean_rollout(b).cpu()
      rs.append(dict(rec(row,h,pred,cpu,lookup[i,h],i),method=recipe+'_mean',N=0))
      if recipe=='M2':
       bb=repeat(b,16);w=torch.cat([torch.randn(cpu['world'].shape,generator=torch.Generator().manual_seed(9001+i+s*100000)) for s in range(16)]).to('cuda')
       for j in range(4):w=w+m(bb,w=w,tw=torch.full((16,),j/4,device='cuda'))['world']/4
       preds=(bb['world_current']+w*bb['world_scale']).cpu()
       for n in (1,16):rs.append(dict(rec(row,h,preds[:n].mean(0,keepdim=True),cpu,lookup[i,h],i),method='M2_FM_N'+str(n),N=n))
     if i%16==0:
      atomic(dest/'evaluation_status.json',dict(status='evaluating',pool=pool,done=i+1,total=len(data.rows),elapsed_s=guard.check(),brt=brt()));print(pool,i+1,flush=True)
    atomic(dest/(pool+'_records.json'),rs)
    methods=sorted({r['method'] for r in rs});atomic(dest/(pool+'_summary.json'),{a:summary([r for r in rs if r['method']==a]) for a in methods})
    atomic(dest/(pool+'_decomposition.json'),{a:decompose([r for r in rs if r['method']==a]) for a in methods})
  atomic(dest/'evaluation_status.json',dict(status='completed',checkpoint=str(cp),checkpoint_sha256=sha(cp),steps=5000,flow_Euler=4,FM_samples=[1,16] if recipe=='M2' else [],language='fixed human short instruction',bootstrap=2000,bootstrap_seed=90,elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,brt=brt()))
 except BaseException as e:
  atomic(dest/'evaluation_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc(),brt=brt()));raise
 finally:
  if guard:guard.finish()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('recipe');ap.add_argument('runid');a=ap.parse_args();run(a.recipe,a.runid)
