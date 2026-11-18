import sys
sys.dont_write_bytecode=True
sys.path.insert(0,'/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e101a')
from diag import *
tr=np.load(OUT/'train.npz');ds={k:np.load(OUT/(k+'.npz')) for k in ('dev','train64')};pars=tr['parents'];hold=np.array([sorted(set(pars)).index(p)%5==0 for p in pars]);yt=(tr['y']-tr['q'][:,None]).reshape(-1,700);res={};cfg={}
for version in ('current_processed14','current_signed14','history70'):
 def get(d):
  a=d['q'].copy() if version!='history70' else d['x'][:,:70].copy()
  if version=='current_processed14':a[:,GRIPS]=a[:,GRIPS].clip(0)
  return a.astype('f8')
 xx=get(tr);mu=xx[~hold].mean(0);sd=xx[~hold].std(0).clip(1e-5);scale=lambda x:np.c_[(x-mu)/sd,np.ones(len(x))];a=scale(xx[~hold]);v=scale(xx[hold]);scores={}
 for al in (.1,1.,10.,100.,1000.):
  reg=np.eye(a.shape[1])*al;reg[-1,-1]=0;co=np.linalg.solve(a.T@a+reg,a.T@yt[~hold]);scores[al]=float(np.mean(((v@co-yt[hold])/yt[~hold].std(0).clip(.0001))**2))
 al=min(scores,key=scores.get);mu=xx.mean(0);sd=xx.std(0).clip(1e-5);a=scale(xx);reg=np.eye(a.shape[1])*al;reg[-1,-1]=0;co=np.linalg.solve(a.T@a+reg,a.T@yt)
 for k,d in ds.items():res[version+'_'+k]=metrics((scale(get(d))@co).reshape(-1,50,14)+d['q'][:,None],d['y'],d['q'],d['parents'])
 cfg[version]=dict(alpha=al,train_parent_validation=scores)
save('input_ablation_metrics.json',res);save('input_ablation_config.json',cfg);status('complete')
