from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from .config import Config
from .runtime import Budget,LatencyRecorder,atomic_json,idle_gpu,brt

def main():
    import signal
    def stop(signum,frame):raise RuntimeError(f'Profile interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--output',required=True);p.add_argument('--fixture-cpu',action='store_true');p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    c=Config.load(args.config) if args.config else Config();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    gpu,snapshot=idle_gpu()
    if gpu is None and not args.fixture_cpu:
        atomic_json(out/'status.json',dict(status='blocked_no_idle_gpu',brt=brt(),gpu_snapshot=snapshot,selection='memory<512MiB AND utilization<5%; never operate others processes'))
        return 75
    if not args.fixture_cpu:
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    # Import torch only after choosing one idle physical card.
    import torch,numpy as np
    from .model import PiperWAM
    from .data import DataPackageDataset,SyntheticDataset,StatContract,move,sha256
    from .randomness import Streams,seed_all
    from .objective import objective
    from .train import optimizer
    from .inference import infer
    torch.set_num_threads(args.threads);seed_all(c.seed)
    device='cpu' if args.fixture_cpu else 'cuda:0'
    budget=Budget(out/'budget.jsonl',0 if args.fixture_cpu else 1,c.profile_budget_gpuh,str(out))
    started=time.monotonic();stats=StatContract.fixture();timings={};evidence={};latency={}
    # Profile coordinates are explicitly provisional, never used by formal launcher.
    atomic_json(out/'status.json',dict(status='running',brt=brt(),formal_training=False,gpu=gpu))
    try:
        model=PiperWAM(c,fixture=args.fixture_cpu).to(device)
        for stage,count in [('P1',30),('P3_warm',20),('P3_joint',50)]:
            budget.check();ds=SyntheticDataset(n=3) if args.fixture_cpu else DataPackageDataset(c.data_root,stage,profile=True)
            # Decode two real train windows once. CPU decode timing is reported independently.
            cached=[];decode=[]
            if args.fixture_cpu: sample_ids=list(range(min(2,len(ds))))
            else:
                groups={}
                for j,r in enumerate(ds.rows):groups.setdefault(r['canonical_id'],[]).append(j)
                pools=[[ids for cid,ids in sorted(groups.items()) if ('demo6' in ds.rows[ids[0]]['source_path'])==flag] for flag in (True,False)]
                sample_ids=[pool[0][len(pool[0])//2] for pool in pools if pool]
                if len(sample_ids)<2:
                    candidates=[ids for ids in groups.values() if ids[len(ids)//2] not in sample_ids]
                    if candidates:sample_ids.append(candidates[0][len(candidates[0])//2])
            for j in sample_ids:
                ts=time.monotonic();cached.append(move(ds[j],device));decode.append(time.monotonic()-ts)
            evidence[stage]=dict(windows=[b['window_id'][0] for b in cached],parents=[b['parent_component'][0] for b in cached],decode_seconds=decode,refs=getattr(ds,'refs',{'fixture':True}),provisional_stats=True)
            model.set_stage(stage);model.train();opt=optimizer(model,3e-4 if stage=='P1' else 2e-4 if stage=='P3_warm' else 1e-4)
            measured=[]
            for k in range(count):
                budget.check();opt.zero_grad(set_to_none=True)
                if device.startswith('cuda'):torch.cuda.synchronize()
                ts=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
                    loss,d=objective(model,cached[k%len(cached)],stats,Streams(c.seed,stage,k),500+k,stage,forced_modes=torch.tensor([k%10//3 if k%10<9 else 3],device=device) if stage=='P3_joint' else None)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite profile loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                if device.startswith('cuda'):torch.cuda.synchronize()
                measured.append(time.monotonic()-ts);budget.tick()
            steady=measured[2:];mean=float(np.mean(steady));decode_mean=float(np.mean(decode))
            timings[stage]=dict(microbatch=1,measured_updates=count,mean_seconds=mean,p95_seconds=float(np.quantile(steady,.95)),decode_seconds=decode_mean,effective32_seconds_estimate=32*(mean+decode_mean),method='32x per-microbatch compute + measured raw decode; conservative optimizer repeated; small train windows; targets uncached')
            atomic_json(out/'partial.json',dict(timings=timings,evidence=evidence,gpuh=budget.used()))
        model.set_stage('P3_joint');model.eval()
        for mode in ('action_only','joint'):
            rec=LatencyRecorder()
            for k in range(10):
                budget.check();torch.cuda.synchronize() if device.startswith('cuda') else None
                ts=time.monotonic();b=move(ds[sample_ids[k%len(sample_ids)]],device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):infer(model,b,stats,mode)
                torch.cuda.synchronize() if device.startswith('cuda') else None;rec.add(time.monotonic()-ts);budget.tick()
            latency[mode]=rec.summary()
        gpu_hours={s:timings[s]['effective32_seconds_estimate']*n/3600 for s,n in [('P1',c.p1_steps),('P3_warm',c.warm_steps),('P3_joint',c.joint_steps)]}
        paired=gpu_hours['P1']+2*(gpu_hours['P3_warm']+gpu_hours['P3_joint'])
        result=dict(config_hash=c.digest(),status='complete',brt=brt(),fixture=args.fixture_cpu,profile_updates=100,profile_gpuh=budget.tick(),stage_gpuh_estimate=gpu_hours,paired_training_gpuh_estimate=paired,with_preparation4_gpuh=paired+4,within48=paired+4<=48,timings=timings,latency=latency,evidence=evidence,parameters=model.parameter_counts(),peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,limitations=['100 microbatch=1 updates, not effective32 optimizer throughput','Provisional identity scales for timing only, formal stat fit gated on P3 quality','10 latency calls/mode; below 1000 acceptance; no latency qualification','No formal training checkpoint saved'])
        atomic_json(out/'profile.json',result);atomic_json(out/'status.json',dict(status='complete',brt=brt(),gpuh=budget.used()))
    except BaseException as e:
        atomic_json(out/'status.json',dict(status='stopped',reason=str(e),brt=brt(),gpuh=budget.tick()));raise
    return 0
if __name__=='__main__':raise SystemExit(main())
