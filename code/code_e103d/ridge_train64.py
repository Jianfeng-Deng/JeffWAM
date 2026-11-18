"""Reuse unchanged E93 target's frozen ridge fit. Additional training-side evaluation only."""
from eval_e94a import *
started=time.monotonic();dest=ART/'ridge_train64_status.json'
try:
 configure();tr,dev=get_data();rs=[];refs=json.loads((E93/'train64_v1_records.json').read_text());lookup={(r['index'],r['h']):r for r in refs};fits={h:np.load(E93/f'ridge_h{h}_coefficients.npz') for h in H}
 with torch.inference_mode():
  for i,row in enumerate(tr.rows):
   for h in H:
    if (i,h) not in lookup:continue
    b=tr.batch([i],(h,),'short','P1');obs=b['obs'][0].numpy();hv=b['history_valid'][0].numpy();nt=8 if h<=50 else 1;n=3*nt;z=fits[h]
    zz=obs if h<=50 else obs.mean(2,keepdims=True);cur=zz[4].reshape(n,768);hist=zz[:4].reshape(4,n,768);delta=hist-cur[None];delta[~hv]=0
    X=np.concatenate([(cur-z['pca_mean'])@z['pca'],*[delta[j]@z['pca'] for j in range(4)],np.broadcast_to(hv.astype('f4'),(n,4)),np.eye(n,dtype='f4')],1)
    y=((X-z['x_mean'])/z['x_scale'])@z['coef']+z['intercept'];pred=b['world_current']+torch.tensor(y,dtype=torch.float32)[None]*b['world_scale']
    rs.append(rec(row,h,pred,b,lookup[i,h],i))
 atomic(ART/'ridge_train64_records.json',rs);atomic(ART/'ridge_train64_summary.json',summary(rs));atomic(dest,dict(status='completed',seconds=time.monotonic()-started,GPUh=0,method='Read-only E93 B5 coefficients; unchanged target, no refit or alpha/dev selection; train64 is in-sample parent reference',coeff_sha={str(h):sha(E93/f'ridge_h{h}_coefficients.npz') for h in H}))
except BaseException as e:atomic(dest,dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
