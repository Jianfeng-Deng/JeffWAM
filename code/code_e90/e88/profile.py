"""Batch32 full immutable pool profile, includes stochastic chain overhead, bounded."""
import sys,json,time,torch,numpy as np
from .common import *
from .train import make_data,configure,calibrate,update
from .resource import Guard
from .model import WorldModel
from e86.data import CappedParentSampler

def main(runid):
    run=ART/runid;dest=run/'profile_01';dest.mkdir(exist_ok=False);p,r,stats,full,small,lc=make_data();configure();guard=Guard('E88_batch32_full_pool_profile',1200);results={}
    try:
        for stage,name in [('P1','P1'),('P3','P3_short')]:
            m=WorldModel().to('cuda:0');m.load_state_dict(torch.load(run/(name+'_end.pt'),map_location='cpu',weights_only=False)['model']);cal=dest/('adapter_'+stage);cal.mkdir();held=calibrate(m,small[stage],guard,cal);sampler=CappedParentSampler(full[stage].rows,seed=881);opt=torch.optim.AdamW(m.parameters(),lr=1e-4);times=[];chain_times=[];normal_times=[];m.train()
            for i in range(72):
                v,d=update(m,opt,sampler,full[stage],stage,5000+i,'short',guard,batch=32,recipe=json.loads((run/'contract.json').read_text())['recipe'])
                if i>=8:
                    times.append(d['seconds']);(chain_times if d['chain_weight']>0 else normal_times).append(d['seconds'])
            results[stage]=dict(batch=32,pool_windows=len(full[stage].rows),measured_steps=len(times),mean_s=float(np.mean(times)),p50_s=float(np.median(times)),p99_s=float(np.quantile(times,.99)),chain_steps=len(chain_times),mean_chain_s=float(np.mean(chain_times)) if chain_times else None,normal_steps=len(normal_times),mean_normal_s=float(np.mean(normal_times)),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,adapter_gate=bool(m.adapter_ready));atomic(dest/'results.json',results)
        t1=results['P1']['mean_s'];t3=results['P3']['mean_s'];raw=(t1*5000+t3*10000)/3600
        atomic(dest/'READY.json',dict(status='ready',results=results,estimate=dict(P1_5000_GPUh=t1*5000/3600,P3_10000_GPUh=t3*10000/3600,total_GPUh=raw,with_25pct_margin_GPUh=raw*1.25,paired_P3_GPUh=(t1*5000+t3*20000)/3600*1.25,three_language_conditions_one_seed_GPUh=(t1*5000+t3*30000)/3600*1.25,scope='cached full pools batch32 all six horizons + random; chain frequency 1/4, two Euler steps, gated real adapter, no cache rebuild/queue/evaluation'),elapsed_s=guard.check()))
    except BaseException as e:
        import traceback
        atomic(dest/'status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
    finally:guard.finish()
if __name__=='__main__':main(sys.argv[1])
