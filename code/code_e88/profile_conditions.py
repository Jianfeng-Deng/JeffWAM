import os,json,sys,time,torch,numpy as np,traceback
from e88.common import *
from e88.train import make_data,configure,calibrate,update
from e88.model import WorldModel
from e88.resource import Guard
from e86.data import CappedParentSampler
run=ART/'smoke_03';dest=run/'profile_conditions_01';dest.mkdir(exist_ok=False);p,r,stats,full,small,lc=make_data();configure();guard=Guard('E88_batch32_none_long_conditions',1000);results={}
try:
 for language in ('none','long'):
  m=WorldModel().to('cuda:0');m.load_state_dict(torch.load(run/('P3_'+language+'_end.pt'),map_location='cpu',weights_only=False)['model']);cal=dest/('adapter_'+language);cal.mkdir();held=calibrate(m,small['P3'],guard,cal);sampler=CappedParentSampler(full['P3'].rows,seed=881);opt=torch.optim.AdamW(m.parameters(),lr=1e-4);times=[];chain_times=[];normal=[];m.train();recipe=json.loads((run/'contract.json').read_text())['recipe']
  for i in range(48):
   v,d=update(m,opt,sampler,full['P3'],'P3',5000+i,language,guard,batch=32,recipe=recipe)
   if i>=8:times.append(d['seconds']);(chain_times if d['chain_weight']>0 else normal).append(d['seconds'])
  results[language]=dict(pool_windows=len(full['P3'].rows),batch=32,measured_steps=len(times),mean_s=float(np.mean(times)),p50_s=float(np.median(times)),p99_s=float(np.quantile(times,.99)),chain_steps=len(chain_times),mean_chain_s=float(np.mean(chain_times)) if chain_times else None,mean_normal_s=float(np.mean(normal)),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,adapter_gate=bool(m.adapter_ready));atomic(dest/'results.json',results)
 atomic(dest/'READY.json',dict(status='ready',results=results,elapsed_s=guard.check(),formal_training_started=False))
except BaseException as e:atomic(dest/'status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
finally:guard.finish()
