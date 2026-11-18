from __future__ import annotations
import torch
from .randomness import generator

@torch.no_grad()
def infer(model,batch,stats,mode='action_only',cache=True,diagnostics=None):
    if mode not in ('joint','action_only'): raise ValueError(mode)
    if model.training: raise RuntimeError('Call eval() before deployment inference')
    if diagnostics is not None:
        task=batch.get('task_id'); diagnostics['UNK_count']=batch['q'].shape[0] if task is None else int(((task<=0)|(task>=model.c.tasks)).sum())
        diagnostics['mode']=mode
    b=stats.norm_state(batch); o,ov=model.observation(b)
    oc=model.core.cache(o,ov) if cache else None
    n=batch['q'].shape[0]; device=o.device; results=[]
    for seed in model.c.noise_seeds:
        ids=batch.get('window_id',[str(i) for i in range(n)])
        a=torch.stack([torch.randn(50,14,generator=generator(seed,'A',wid,device=device),device=device) for wid in ids])
        w=torch.stack([torch.randn(3,91,768,generator=generator(seed,'W',wid,device=device),device=device) for wid in ids]) if mode=='joint' else None
        for i in range(model.c.euler_steps):
            tau=torch.full((n,),i/model.c.euler_steps,device=device)
            av,wv,_=model.predict(o,ov,a,w,tau,tau if w is not None else None,w_valid=batch.get('world_structure_valid') if w is not None else None,cache=oc)
            a=a+av/model.c.euler_steps
            if w is not None: w=w+wv/model.c.euler_steps
        results.append(a)
    return stats.denorm_action(torch.stack(results).mean(0),batch['q'])
