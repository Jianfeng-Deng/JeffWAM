from common import *
import traceback
started=time.monotonic()
try:
 assert json.loads((R/'extract_status.json').read_text())['status']=='completed'
 m=model_cpu();_,tr,dev=get_data();records={};torch.set_num_threads(2)
 with torch.inference_mode():
  for name,data in [('dev',dev),('train64',tr)]:
   rs=[]
   for i,row in enumerate(data.rows):
    for h in H:
     ids,why=data.index.resolve(row,h)
     if ids is None:continue
     b=data.batch([i],(h,),'short','P1');pred=m.rollout(b,steps=4,seed=9001+i);r=rec(row,h,pred,b,index=i)
     # Preserve E90 v1 extrapolation baseline exactly.
     js=b['history_valid'][0].nonzero().flatten();j=int(js[-1]) if len(js) else 4;previous=b['obs'][0,j];z,_=data.index.get(row['canonical_id']);ct=z['camera_exposure_timestamp'];ratio=[]
     for cam in range(3):
      now=row['obs_rows'][4][cam];past=row['obs_rows'][j][cam];dt=ct[now,cam]-ct[past,cam];ratio.append((ct[ids[cam],cam]-ct[now,cam])/dt if dt>1e-7 else 0.)
     nt=8 if h<=50 else 1
     if h>50:previous=previous.mean(1,keepdim=True)
     ratio=torch.tensor(ratio,dtype=b['world_current'].dtype)[:,None].expand(3,nt).reshape(1,-1,1);linear=b['world_current']+(b['world_current']-previous.reshape(1,-1,768))*ratio
     r['linear']=float(((linear-b['world_raw'])/b['world_scale']).square().mean());rs.append(r)
    if i%8==0:atomic(R/'cpu_status.json',dict(status='evaluating',pool=name,done=i+1,total=len(data.rows),seconds=time.monotonic()-started))
   atomic(R/(name+'_v1_records.json'),rs);atomic(R/(name+'_v1_summary.json'),summary(rs));records[name]=rs
 # Original 48 windows keep their original index/seeds; exact reproduction check.
 old=json.loads((T/'results/e90_p1_20261114_01/dev_records.json').read_text())['world'];new=[r for r in records['dev'] if r['index']<48];assert len(old)==len(new)
 diffs={k:max(abs(a[k]-b[k]) for a,b in zip(old,new)) for k in ('model','copy','linear')};atomic(R/'v1_replay.json',dict(max_abs_difference=diffs,records=len(new),backend='CPU float32',Euler=4,seed=9001));assert max(diffs.values())<1e-5
 atomic(R/'cpu_status.json',dict(status='completed',seconds=time.monotonic()-started,GPUh=0))
except BaseException as e:atomic(R/'cpu_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
