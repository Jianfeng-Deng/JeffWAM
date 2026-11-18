from common import *
import traceback
started=time.monotonic()
try:
 m=model_cpu();_,tr,dev=get_data();rs=[]
 with torch.inference_mode():
  for name,data in [('dev',dev),('train64',tr)]:
   for i,row in enumerate(data.rows):
    for h in (10,25,50):
     if data.index.resolve(row,h)[0] is None:continue
     b=data.batch([i],(h,),'short','P1');out=m(b,w=torch.zeros_like(b['world']),tw=torch.zeros(1));p=out['world_cos'][0];y=b['world'][0];cos=float(torch.nn.functional.cosine_similarity(p.flatten(),y.flatten(),dim=0));rs.append(dict(**record_identity(row),pool=name,h=h,cosine=cos,head_norm=float(p.norm()),truth_norm=float(y.norm())))
    if i%32==0:atomic(R/'aux_cos_status.json',dict(status='evaluating',pool=name,done=i,total=len(data.rows)))
 result={name:{str(h):aggregate([r for r in rs if r['pool']==name and r['h']==h],['cosine'],2000) for h in (10,25,50)} for name in ('dev','train64')};atomic(R/'aux_cos_results.json',result);atomic(R/'aux_cos_records.json',rs);atomic(R/'aux_cos_status.json',dict(status='completed',seconds=time.monotonic()-started,GPUh=0));print(json.dumps({p:{h:r['values']['cosine'] for h,r in v.items()} for p,v in result.items()}))
except BaseException as e:atomic(R/'aux_cos_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
