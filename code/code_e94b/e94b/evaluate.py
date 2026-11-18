from .common import *
import torch,numpy as np,traceback
from .data import load
from .model import WorldModel
from .resource import Guard
from .metrics import rec,summary,decomposition
from .ridge import linear_delta
from e86.data import move
from e90.train import configure

def evaluate(branch):
 dest=R/('e94b_'+branch+'_20261114_01');out=dest/'evaluation';out.mkdir(exist_ok=False);g=None
 try:
  _,tr,dev=load();configure();m=WorldModel()
  if branch=='b2':
   from .readout import make_model
   m=make_model()
  ck=dest/'ckpt_05000.pt';d=torch.load(ck,map_location='cpu',weights_only=False);assert d['step']==5000;m.load_state_dict(d['model']);m.eval()
  cfg=json.loads((dest/'config.json').read_text());det=cfg.get('readout_recipe')=='M1';hasmean=branch=='b2'
  g=Guard('eval_'+branch);m.cuda()
  with torch.inference_mode():
   for name,data in [('dev',dev),('train64',tr)]:
    records=[]
    for i,row in enumerate(data.rows):
     g.check()
     for h in H:
      if data.index.resolve(row,h)[0] is None:continue
      cpu=data.batch([i],(h,),'short','P1');b=move(cpu,'cuda:0');y=cpu['world'][0].numpy();small={k:(v[0] if isinstance(v,torch.Tensor) and k!='world_positions' else v) for k,v in cpu.items()};lin=linear_delta(data,row,small,h)
      if hasmean:
       from .readout import predict_mean
       pred=predict_mean(m,b).cpu()[0].numpy();dn=(pred-cpu['world_current'][0].numpy())/cpu['world_scale'][0].numpy();records.append(rec(row,h,dn,y,lin,index=i,branch='mean',N=0))
      if not det:
       bb={k:(v.repeat((16,)+(1,)*(v.ndim-1)) if isinstance(v,torch.Tensor) and k!='world_positions' else v) for k,v in b.items()}
       w=torch.cat([torch.randn(cpu['world'].shape,generator=torch.Generator().manual_seed(9001+i+s*100000)) for s in range(16)]).cuda()
       for j in range(4):
        tau=torch.full((16,),j/4,device='cuda');w=w+m(bb,w=w,tw=tau)['world']/4
       for n in (1,16):records.append(rec(row,h,w[:n].mean(0).cpu().numpy(),y,lin,index=i,branch='FM',N=n,Euler=4))
     if i%8==0:atomic(out/'status.json',dict(status='evaluating',pool=name,done=i+1,total=len(data.rows),seconds=g.check()))
    atomic(out/(name+'_records.json'),records);summaries={};decomps={}
    for label,n in ([('mean',0)] if hasmean else [])+([] if det else [('FM',1),('FM',16)]):
     rr=[r for r in records if r['branch']==label and r['N']==n];key=label+str(n);summaries[key]=summary(rr);decomps[key]=decomposition(rr)
    atomic(out/(name+'_summary.json'),summaries);atomic(out/(name+'_decomposition.json'),decomps)
  atomic(out/'receipt.json',dict(checkpoint=str(ck),sha256=sha(ck),fixed_step=5000,flow_steps=4,noise='CPU per window seed9001+index+100000*sample_index, same pool prefix N1,N16',language='fixed human short instruction only',bootstrap=2000,bootstrap_seed=90,estimator='equal parents, mean windows within parent, paired bootstrap ratio of MSE means',target_contract_sha256=sha(CACHE/'normalization_contract.json'),peak_mib=torch.cuda.max_memory_allocated()/2**20));atomic(out/'status.json',dict(status='completed',brt=brt()));report('## 6 '+branch+' dev/train64 统一评估完成\n'+(out/'dev_summary.json').read_text());status(branch+'_evaluated')
 except BaseException as e:atomic(out/'status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
 finally:
  if g:g.finish()
if __name__=='__main__':evaluate(sys.argv[1])
