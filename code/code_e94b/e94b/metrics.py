from .common import *
import numpy as np

def aggregate(rr,keys=('model','copy','linear','cosine'),resamples=2000):
 if not rr:return dict(windows=0,parents=0,values=None,ci95=None)
 by=collections.defaultdict(list)
 for r in rr:by[r['parent']].append(r)
 vals=np.array([[np.mean([r[k] for r in rows]) for k in keys] for _,rows in sorted(by.items())]);mean=vals.mean(0)
 g=np.random.default_rng(90);boot=vals[g.integers(len(vals),size=(resamples,len(vals)))].mean(1);out=dict(zip(keys,map(float,mean)));ci={k:np.quantile(boot[:,j],[.025,.975]).tolist() for j,k in enumerate(keys)}
 for base in ('copy','linear'):
  if base not in keys:continue
  j=keys.index(base);i=keys.index('model');out['skill_'+base]=float(1-mean[i]/max(mean[j],1e-20));ci['skill_'+base]=np.quantile(1-boot[:,i]/np.maximum(boot[:,j],1e-20),[.025,.975]).tolist()
 return dict(windows=len(rr),parents=len(by),values=out,ci95=ci)
def rec(row,h,d,y,linear,**kw):
 err=(d-y)**2;cp=y**2;nt=y.shape[0]//3
 return dict(parent=row['parent_component'],canonical_id=row['canonical_id'],anchor=row['anchor'],h=h,model=float(err.mean()),copy=float(cp.mean()),linear=float(((linear-y)**2).mean()),cosine=float(np.dot(d.ravel(),y.ravel())/max(np.linalg.norm(d)*np.linalg.norm(y),1e-12)),model_camera=err.reshape(3,nt,768).mean((1,2)).tolist(),copy_camera=cp.reshape(3,nt,768).mean((1,2)).tolist(),linear_camera=((linear-y)**2).reshape(3,nt,768).mean((1,2)).tolist(),cosine_camera=[float(np.dot(a.ravel(),b.ravel())/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)) for a,b in zip(d.reshape(3,nt,768),y.reshape(3,nt,768))],change=float(cp.mean()),**kw)
def summary(records):return {str(h):aggregate([r for r in records if r['h']==h]) for h in H}
def decomposition(records):
 out={}
 for h in H:
  rr=[r for r in records if r['h']==h];cuts=np.quantile([r['change'] for r in rr],[.25,.5,.75]);groups={}
  for k,cam in enumerate(('top','left_wrist','right_wrist')):
   cr=[dict(r,model=r['model_camera'][k],copy=r['copy_camera'][k],linear=r['linear_camera'][k],cosine=r['cosine_camera'][k]) for r in rr];groups[cam]=aggregate(cr)
  for q in range(4):groups['change_Q'+str(q+1)]=aggregate([r for r in rr if np.searchsorted(cuts,r['change'],side='right')==q])
  out[str(h)]=dict(cuts=cuts.tolist(),groups=groups,quartiles_descriptive_only=True)
 return out
