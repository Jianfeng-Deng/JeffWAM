from __future__ import annotations
import argparse,hashlib,json,math,os,random,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .data import SyntheticDataset,ManifestDataset,DataPackageDataset,ParentSampler,StatContract,collate,move,read_ready,sha256
from .model import PiperWAM
from .objective import objective
from .randomness import Streams,seed_all
from .runtime import Budget,atomic_json,brt,cache_guard

STAGES={'P1':('p1_steps',3e-4),'P3_warm':('warm_steps',2e-4),'P3_joint':('joint_steps',1e-4)}

def optimizer(model,lr):
    visual=[p for p in model.vision.online.parameters() if p.requires_grad]
    visual_ids={id(p) for p in visual}
    rest=[p for p in model.parameters() if p.requires_grad and id(p) not in visual_ids]
    groups=[dict(params=rest,lr=lr,weight_decay=1e-4,peak_lr=lr)]
    if visual: groups.append(dict(params=visual,lr=3e-6,weight_decay=0.,peak_lr=3e-6))
    return torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)

def schedule(opt,step,total):
    warm=max(1,int(total*.05))
    factor=(step+1)/warm if step<warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))
    for group in opt.param_groups: group['lr']=group['peak_lr']*factor

def save(path,model,opt,step,metadata):
    state=model.state_dict()
    if not model.vision.fixture:
        state={k:v for k,v in state.items() if not k.startswith('vision.') or any(n in k for n in ('q_a','q_b','v_a','v_b'))}
    d=dict(model=state,optimizer=opt.state_dict() if opt else None,step=step,metadata=metadata,
           rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),cuda=torch.cuda.get_rng_state(next(model.parameters()).device) if next(model.parameters()).is_cuda else None))
    tmp=Path(str(path)+'.tmp'); torch.save(d,tmp); os.replace(tmp,path)

def load(path,model,opt=None,device='cpu',restore_rng=False):
    d=torch.load(path,map_location=device,weights_only=False)
    saved=d['model']; expected=model.state_dict()
    omitted=set(expected)-set(saved)
    if any(not k.startswith('vision.') or any(n in k for n in ('q_a','q_b','v_a','v_b')) for k in omitted) or set(saved)-set(expected): raise ValueError('Checkpoint module keys mismatch')
    expected.update(saved); model.load_state_dict(expected,strict=True)
    if opt is not None:
        if d['optimizer'] is None: raise ValueError('Cannot resume weight-only checkpoint')
        opt.load_state_dict(d['optimizer'])
    if restore_rng:
        torch.set_rng_state(d['rng']['torch'].cpu()); np.random.set_state(d['rng']['numpy']); random.setstate(d['rng']['python'])
        if d['rng']['cuda'] is not None and next(model.parameters()).is_cuda:
            torch.cuda.set_rng_state(d['rng']['cuda'].cpu(),next(model.parameters()).device)
    return d

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--run',required=True); p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--fixture',action='store_true'); p.add_argument('--steps',type=int); p.add_argument('--device',default='cpu'); p.add_argument('--manifest'); p.add_argument('--stats'); p.add_argument('--init'); p.add_argument('--resume',action='store_true'); p.add_argument('--ledger'); p.add_argument('--arm',choices=['R','P'],default='R'); p.add_argument('--monitor'); p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(); c=Config.load(args.config) if args.config else Config(); torch.set_num_threads(args.threads)
    if not args.fixture and args.device=='cpu': raise ValueError('Formal DINO training requires an explicitly allocated GPU')
    if not args.fixture and args.steps is not None: raise ValueError('Do not silently shorten formal recipe; preregister a new config')
    if not args.fixture and not (args.stats and args.init): raise ValueError('Formal stages require locked manifest/stat/S0 or stage-init')
    if not args.fixture and args.arm=='R' and args.stage=='P1': raise ValueError('R starts directly at P3')
    run=Path(args.run)
    if args.resume:
        if not (run/'latest.pt').exists(): raise FileNotFoundError('Resume checkpoint missing')
    else: run.mkdir(parents=True,exist_ok=False)
    budget=Budget(args.ledger or run/'budget.jsonl',int(args.device.startswith('cuda')),c.budget_gpuh-4.,str(run))
    atomic_json(run/'status.json',dict(status='initializing',brt=brt(),stage=args.stage,formal=not args.fixture))
    try:
        seed_all(c.seed)
        if args.fixture: ds=SyntheticDataset(); stats=StatContract.fixture(); refs={'fixture':True}
        else:
            ready=read_ready(c.data_root); ds=ManifestDataset(args.manifest,args.stage) if args.manifest else DataPackageDataset(c.data_root,args.stage); stats=StatContract.load(args.stats)
            if stats.d.get('fixture'): raise ValueError('Synthetic statistics forbidden for real training')
            refs=dict(manifest_hash=sha256(args.manifest or ds.captures_path),stat_hash=sha256(args.stats),parent_ready_hash=sha256(Path(c.data_root)/'READY_parent_split.json'),data_ready_hash=sha256(Path(c.data_root)/'READY_data.json'))
        source_hash=hashlib.sha256(''.join(sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest()
        if not args.fixture:
            if stats.d.get('data_refs')!={'parent_ready':refs['parent_ready_hash'],'data_ready':refs['data_ready_hash']}: raise ValueError('Statistics data lock differs from training')
            if stats.d.get('encoder_sha256')!=sha256(c.dino_weights): raise ValueError('Statistics E* differs')
        meta=dict(encoder_sha256=None if args.fixture else sha256(c.dino_weights),code_hash=source_hash,config=asdict(c),config_hash=c.digest(),stage=args.stage,arm=args.arm,refs=refs)
        atomic_json(run/'config.json',meta)
        model=PiperWAM(c,fixture=args.fixture).to(args.device); model.set_stage(args.stage)
        total=args.steps if args.steps is not None else getattr(c,STAGES[args.stage][0]); opt=optimizer(model,STAGES[args.stage][1]); begin=0
        if args.resume:
            d=load(run/'latest.pt',model,opt,args.device,True)
            if d['metadata']!=meta: raise ValueError('Resume config/data/stage changed')
            begin=d['step']
        elif args.init:
            d=load(args.init,model,device=args.device)
            if not args.fixture:
                expected='S0' if args.stage=='P3_warm' and args.arm=='R' or args.stage=='P1' else 'P1' if args.stage=='P3_warm' else 'P3_warm'
                if d['metadata'].get('stage')!=expected: raise ValueError(f'Expected init stage {expected}')
                if d['metadata'].get('fixture'): raise ValueError('Fixture checkpoint forbidden')
                if d['metadata'].get('encoder_sha256')!=meta['encoder_sha256'] or d['metadata'].get('config_hash')!=c.digest():raise ValueError('Initialization encoder/config lock changed')
            if d['metadata'].get('seed',d['metadata'].get('config',{}).get('seed'))!=c.seed: raise ValueError('S0 seed mismatch')
            if d['metadata'].get('stage') in STAGES and d['metadata'].get('refs')!=refs: raise ValueError('Stage changed frozen data/stat references')
        sampler=ParentSampler(ds.rows,c.seed,args.stage); atomic_json(run/'sampling.json',dict(strategy=sampler.strategy,parents=len({r['parent_component'] for r in ds.rows}),windows=len(ds))); model.train(); tic=time.monotonic()
        accumulation=c.batch_size//c.microbatch
        for step in range(begin,total):
            budget.check(); schedule(opt,step,total); opt.zero_grad(set_to_none=True)
            ids=sampler.sample(step,c.batch_size); logs=[]
            full_modes=None; world_count=0
            if args.stage=='P3_joint':
                u=Streams(c.seed,args.stage,step).uniform('mode',(c.batch_size,),args.device)
                full_modes=torch.bucketize(u,torch.tensor([.3,.6,.9],device=args.device)); world_count=int((full_modes!=2).sum())
            for start in range(0,len(ids),c.microbatch):
                b=move(collate([ds[i] for i in ids[start:start+c.microbatch]]),args.device)
                streams=Streams(c.seed,args.stage,step,start)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                    modes=full_modes[start:start+c.microbatch] if full_modes is not None else None
                    world_scale=accumulation*int((modes!=2).sum())/max(1,world_count) if modes is not None else 1.
                    loss,metrics=objective(model,b,stats,streams,step,args.stage,forced_modes=modes,world_scale=world_scale)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                (loss/accumulation).backward(); logs.append(metrics)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            opt.step(); used=budget.tick()
            entry=dict(step=step+1,brt=brt(),gradient_norm=float(norm),gpuh=used,exposures=(step+1)*c.batch_size,metrics=logs)
            with (run/'metrics.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
            if step%100==0 or step+1==total:
                cache_guard(run.parent,15.);  save(run/'latest.pt',model,opt,step+1,meta)
                atomic_json(run/'status.json',dict(status='running',brt=brt(),step=step+1,total=total,gpuh=used,wall_seconds=time.monotonic()-tic))
            if args.stage=='P3_joint' and (step+1)%500==0:
                from .diagnostics import gradient_diagnostic
                diagnostic=gradient_diagnostic(model,move(ds[sampler.sample(0,1)[0]],args.device),stats)
                with (run/'gradient_diagnostics.jsonl').open('a') as f: f.write(json.dumps(dict(step=step+1,groups=diagnostic))+'\n')
            if args.monitor and Path(args.monitor).is_file():
                from .diagnostics import stoploss_monitor
                reason=stoploss_monitor(json.loads(Path(args.monitor).read_text()))
                if reason: raise RuntimeError('Preregistered stoploss: '+reason)
            if args.stage=='P3_joint' and step+1 in (3000,6000,9000): save(run/f'grid_{step+1}.pt',model,None,step+1,meta)
            if args.stage=='P1' and step+1 in (1000,2500): save(run/f'diagnostic_{step+1}.pt',model,None,step+1,meta)
        save(run/'stage_end.pt',model,opt,total,meta)
        atomic_json(run/'status.json',dict(status='complete',brt=brt(),step=total,gpuh=budget.tick()))
    except BaseException as exc:
        atomic_json(run/'status.json',dict(status='stopped',brt=brt(),reason=str(exc),gpuh=budget.tick()))
        raise
if __name__=='__main__': main()
