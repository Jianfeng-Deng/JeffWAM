"""Resume a smoke checkpoint into a new directory; maximum128 additional updates."""
import argparse,json,time,os
from pathlib import Path
from .common import *
def main(checkpoint,new_run,extra_steps=1,cache=None):
    if not 1<=extra_steps<=128:raise ValueError('Smoke resume allows 1..128 updates only')
    old=Path(checkpoint).resolve();source=old.parent/'contract.json';dest=C/'artifacts'/new_run;dest.mkdir(exist_ok=False)
    from .resource import Guard
    guard=Guard('bounded_resume_'+new_run,180)
    try:
        import torch,numpy as np
        from .checkpoint import load,save
        from .data import Package,CachedDataset,CappedParentSampler,collate,move
        from .language import load as load_language
        from .model import WAM,Config
        from .objective import loss
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(2)
        contract=json.loads(source.read_text());p=Package();assert contract['package']==p.refs;assert contract['stats']==sha(C/'artifacts/stats_v1.json');assert contract['language']==sha(C/'artifacts/language_v1/READY_language.json')
        cache=Path(cache) if cache else C/'artifacts/features_v3';assert contract['cache_contract']==sha(cache/'contract.json')
        d=torch.load(old,map_location='cpu',weights_only=False);stage=old.name.split('_resume')[0].split('_end')[0];assert stage in ('P1','P3','P3_no_language')
        target=192 if stage=='P1' else 384
        if d['step']+extra_steps>target:raise ValueError('Cannot extend beyond original smoke stage target')
        dsstage='P1' if stage=='P1' else 'P3';ws=contract[dsstage+'_subset'];stats=json.loads((C/'artifacts/stats_v1.json').read_text());lc=load_language(ws,C/'artifacts/language_v1');ds=CachedDataset(ws,p.captures,cache,stats,lc);sampler=CappedParentSampler(ws,seed=d['sampler']['seed'],cap=d['sampler']['cap']);m=WAM(Config(**d['config'])).train().to('cuda:0');lr=3e-4 if stage=='P1' else 1e-4;opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
        refs=dict(contract=sha(source));(dest/'contract.json').write_bytes(source.read_bytes());step=load(old,m,opt,sampler,refs);losses=[];expected={}
        for line in (old.parent/'updates.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['stage']==stage:expected[r['step']]=r['loss']
        for i in range(step,step+extra_steps):
            guard.check();b=move(collate([ds[j] for j in sampler.sample(8)]),'cuda:0')
            if stage=='P3_no_language':b=dict(b,text_valid=torch.zeros_like(b['text_valid']),align_valid=torch.zeros_like(b['align_valid']))
            warm=min(1.,(i+1)/max(1,int(.05*target)));cos=.1+.9*(1+np.cos(np.pi*i/target))/2
            for group in opt.param_groups:group['lr']=lr*warm*cos
            opt.zero_grad(set_to_none=True);v,metrics=loss(m,b,dsstage,i,seed=86,clean_aux_probability=contract['clean_aux_probability']);assert torch.isfinite(v);v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();value=float(v.detach());losses.append(dict(step=i+1,loss=value,expected_original_loss=expected.get(i+1),exact_match=value==expected.get(i+1)))
            if i+1 in expected and value!=expected[i+1]:raise RuntimeError('External resume replay differs from original smoke')
        save(dest/'resumed.pt',m,opt,step+extra_steps,sampler,refs,extra=dict(resumed_from=str(old),source_checkpoint_sha256=sha(old)))
        atomic(dest/'status.json',dict(status='resume_smoke_completed',stage=stage,from_step=step,extra_steps=extra_steps,results=losses,source_checkpoint_sha256=sha(old),elapsed_s=guard.check(),formal_training_started=False));print(json.dumps(losses),flush=True)
    finally:guard.finish()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('new_run');p.add_argument('--extra-steps',type=int,default=1);p.add_argument('--cache');a=p.parse_args();main(a.checkpoint,a.new_run,a.extra_steps,a.cache)
