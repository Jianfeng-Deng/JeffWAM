import sys,json,collections,fcntl
sys.dont_write_bytecode=True
from e102x_setup import *
import numpy as np
from e102a_finalize import action_summary,quant
GROUPS=('A0','A1','A2');METHODS=('gt_action','pred_N1','no_traj','oracle')
def bootstrap(n):return np.random.default_rng(90).integers(n,size=(2000,n))
def worlds(rows):
 out={}
 for h in (10,25,50,100):
  rr=[r for r in rows if r['h']==h];parents=sorted({r['parent'] for r in rr});idx=bootstrap(len(parents));a={m:np.array([[np.mean([r[k] for r in rr if r['parent']==p and r['method']==m]) for k in ('model','copy')] for p in parents]) for m in METHODS};copy=a['no_traj'][:,1];den=copy.mean();delta=a['no_traj'][:,0]-a['pred_N1'][:,0];od=a['no_traj'][:,0]-a['oracle'][:,0]
  with np.errstate(divide='ignore',invalid='ignore'):ret=delta[idx].mean(1)/od[idx].mean(1)
  out[str(h)]=dict(parents=len(parents),windows=len(rr)//4,skill={m:float(1-v[:,0].mean()/den) for m,v in a.items()},skill_ci95={m:quant(1-v[idx,0].mean(1)/copy[idx].mean(1)) for m,v in a.items()},delta_skill=float(delta.mean()/den),delta_ci95=quant(delta[idx].mean(1)/copy[idx].mean(1)),retention=float(delta.mean()/od.mean()) if od.mean()>0 else None,retention_ci95=quant(ret[np.isfinite(ret)]) if np.isfinite(ret).any() else None)
 return out
