import torch,json
from e88.train import make_data,make_chain
from e88.model import WorldModel
from e88.objective import loss
from e86.common import atomic
from e88.common import ART
p,r,stats,full,small,lc=make_data();m=WorldModel().eval();out={};torch.set_num_threads(2)
for stage in ('P1','P3'):
 ds=small[stage];b=ds.batch([0,1]);v,d=loss(m,b,stage,0);v.backward();out[stage]={'loss':float(v.detach()),'shape':list(b['world'].shape),'finite':bool(torch.isfinite(v))}
for k in (25,50,100):
 fn=make_chain(small['P3'],range(32),k,'long','P3',192,'cpu')
 if fn:
  v,d=fn(m);v.backward();out[str(k)]={'loss':float(v.detach()),**d}
 else:out[str(k)]={'status':'no_candidates'}
out['cuda_initialized']=torch.cuda.is_initialized();atomic(ART/'real_pipeline_CPU.json',out);print(json.dumps(out))
