"""Fixed 32 effective-batch timing, 100 updates/stage, no selection or formal run."""
import time,json,sys
from .common import *
def main(run_id):
    from .resource import Guard
    guard=Guard('batch32_training_profile',600)
    run=C/'artifacts'/run_id;dest=run/'profile32_v1';dest.mkdir(exist_ok=False)
    try:
        import torch,numpy as np
        from .data import Package,CachedDataset,CappedParentSampler,collate,move
        from .language import load as load_language
        from .model import WAM
        from .objective import loss
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
        torch.set_num_threads(2);p=Package();stats=json.loads((C/'artifacts/stats_v1.json').read_text());contract=json.loads((run/'contract.json').read_text());results={}
        for stage in ('P1','P3'):
            ws=contract[stage+'_subset'];lc=load_language(ws,C/'artifacts/language_v1');ds=CachedDataset(ws,p.captures,C/'artifacts/features_v3',stats,lc);sampler=CappedParentSampler(ws,seed=861)
            m=WAM().to('cuda:0');m.load_state_dict(torch.load(run/(stage+'_end.pt'),map_location='cpu',weights_only=False)['model']);m.train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4);times=[];mode_times={}
            for step in range(108):
                torch.cuda.synchronize();t=time.perf_counter();guard.check();b=move(collate([ds[i] for i in sampler.sample(32)]),'cuda:0');opt.zero_grad(set_to_none=True);v,d=loss(m,b,stage,step,seed=86,clean_aux_probability=contract['clean_aux_probability']);v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();torch.cuda.synchronize();dt=time.perf_counter()-t
                if not torch.isfinite(v):raise RuntimeError('NaN profile loss')
                if step>=8:times.append(dt);mode_times.setdefault(d['mode'],[]).append(dt)
            results[stage]=dict(steps_measured=len(times),effective_batch=32,mean_step_s=float(np.mean(times)),p50_s=float(np.median(times)),p99_s=float(np.quantile(times,.99)),mode_mean_s={k:float(np.mean(v)) for k,v in mode_times.items()},peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            atomic(dest/'results.json',results)
        t1=results['P1']['mean_step_s'];t3=results['P3']['mean_step_s'];estimate=dict(P1_5000_GPUh=5000*t1/3600,P3_10000_GPUh=10000*t3/3600,single_P1_P3_GPUh=(5000*t1+10000*t3)/3600,single_P1_P3_with_25pct_margin_GPUh=1.25*(5000*t1+10000*t3)/3600,paired_R_P_with_25pct_margin_GPUh=1.25*(5000*t1+20000*t3)/3600,scope='measured cached-features effective batch32; 100 measured updates/stage; P3 mixture used as conservative warm+joint proxy; excludes full cache rebuild, scheduler, storage tail variability; no formal launch')
        atomic(dest/'READY_profile.json',dict(status='ready',results=results,formal_estimate=estimate,elapsed_s=guard.check()));print(json.dumps(estimate),flush=True)
    finally:guard.finish()
if __name__=='__main__':main(sys.argv[1])
