from __future__ import annotations
import torch
from .config import GROUPS
from .randomness import Streams

def action_loss(pred,target):
    weights=pred.new_tensor([4]*10+[2]*15+[1]*25)
    return (((pred.float()-target.float())**2).mean(-1)*weights).sum(-1)/weights.sum()

def world_loss(pred,target,valid):
    err=(pred.float()-target.float()).square().mean(-1)
    values=[]; available=[]
    for start,end,_ in GROUPS:
        mask=valid[:,:,start:end].float(); count=mask.sum(-1)
        values.append((err[:,:,start:end]*mask).sum(-1)/count.clamp_min(1))
        available.append(count>0)
    camera_values=[];camera_valid=[]
    for ids in ((0,),(1,2),(3,4)):
        den=sum(available[j].float() for j in ids)
        camera_values.append(sum(values[j]*available[j] for j in ids)/den.clamp_min(1))
        camera_valid.append(den>0)
    den=sum(v.float() for v in camera_valid)
    per_endpoint=sum(v*m for v,m in zip(camera_values,camera_valid))/den.clamp_min(1)
    endpoint=err.new_tensor([.5,.3,.2])[None]*(den>0)
    if not (endpoint.sum(-1)>0).all(): raise ValueError('Empty world target forbidden')
    return (per_endpoint*endpoint).sum(-1)/endpoint.sum(-1)

def augment(batch,streams,stage):
    b=dict(batch); n=b['q'].shape[0]; dev=b['q'].device
    factors=1+(streams.uniform('augmentation',(n,1,1,1,1),dev)-.5)*.1
    b['obs_images']=[(x*factors).clamp(0,1) for x in b['obs_images']]
    taskdrop=streams.uniform('task_dropout',(n,),dev)<(.1 if stage=='P1' else .05)
    b['task_id']=torch.where(taskdrop,0,b['task_id'])
    if stage!='P1':
        hdrop=streams.uniform('history_dropout',(n,),dev)<.1
        b['state_valid']=b['state_valid'].clone(); b['state_valid'][:,3]&=~hdrop
        b['history']=b['history'].clone(); b['history'][hdrop]=0
    return b

def objective(model,batch,stats,streams,step,stage,forced_modes=None,world_scale=1.):
    n=batch['q'].shape[0]; device=batch['q'].device
    b=augment(stats.norm_state(batch),streams,stage)
    o,ov=model.observation(b)
    ta=streams.normal('tauA',(n,),device).sigmoid()
    tw=streams.normal('tauW',(n,),device).sigmoid()
    ea=streams.normal('epsilonA',(n,50,14),device)
    ya=stats.norm_action(batch['action'],batch['q']).detach()
    a=(1-ta[:,None,None])*ea+ta[:,None,None]*ya; at=ya-ea
    if stage=='P3_warm':
        av,_,_=model.predict(o,ov,a=a,ta=ta)
        la=action_loss(av,at).mean()
        return la,dict(LA=float(la.detach()),LW=None,lambda_world=0.,modes=[0,0,n,0])
    world_structure=batch.get('world_structure_valid')
    yw,target_valid=model.world_target(batch); yw=stats.norm_world(yw).detach()
    ew=streams.normal('epsilonW',yw.shape,device)
    if stage=='P1':
        w=(1-tw[:,None,None,None])*ew+tw[:,None,None,None]*yw
        _,wv,_=model.predict(o,ov,w=w,tw=tw,w_valid=world_structure)
        lw=world_loss(wv,yw-ew,target_valid).mean()
        return lw,dict(LA=None,LW=float(lw.detach()),lambda_world=1.,modes=[0,0,0,0])
    u=streams.uniform('mode',(n,),device)
    modes=torch.bucketize(u,torch.tensor([.3,.6,.9],device=device)) if forced_modes is None else forced_modes.to(device)
    asum=o.new_zeros(()); wsum=o.new_zeros(()); nw=0; counts=[]
    for mode in range(4):
        ids=(modes==mode).nonzero().flatten(); counts.append(len(ids))
        if not len(ids): continue
        ai=a[ids]; ti=ta[ids]; wi=None; wt=None
        if mode!=2:
            wt=ta[ids] if mode==0 else tw[ids]
            wi=(1-wt[:,None,None,None])*ew[ids]+wt[:,None,None,None]*yw[ids]
        if mode==3: ai=ya[ids]; ti=torch.ones_like(ti)
        av,wv,_=model.predict(o[ids],ov[ids],a=ai,w=wi,ta=ti,tw=wt,w_valid=world_structure[ids] if world_structure is not None and wi is not None else None)
        if mode==3:
            # Extra 10% action-only call: every action sample receives one main LA.
            av,_,_=model.predict(o[ids],ov[ids],a=a[ids],ta=ta[ids])
        asum+=action_loss(av,at[ids]).sum()
        if mode!=2:
            wsum+=world_loss(wv,yw[ids]-ew[ids],target_valid[ids]).sum(); nw+=len(ids)
    la=asum/n; lw=wsum/max(nw,1)
    lam=model.c.lambda_world*min(1.,step/model.c.lambda_ramp)
    loss=la+lam*world_scale*lw
    return loss,dict(LA=float(la.detach()),LW=float(lw.detach()) if nw else None,lambda_world=lam,modes=counts)
