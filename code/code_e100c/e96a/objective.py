"""P1 direct conditional mean MSE; M2 adds original FM with weight 1:1."""
import torch
import torch.nn.functional as F
from e86.common import generator
from e90.objective import weighted_mean,chain_weight

def loss(model,b,stage,step,seed=90,chain=None,chain_warmup=3000,chain_ramp=1000,chain_maximum=.1,p1_converged=False,engineering_chain_probe=False,phase_weight=0.,adapter_weight=.1,alignment_weight=0.,cosine_weight=.1,**kwargs):
    if stage!='P1' or phase_weight or alignment_weight:raise ValueError('E94 readout objective is P1 only, phase/alignment weights zero')
    y=b['world'].detach(); out=model.forward_mean(b)
    mean=weighted_mean((out['mean']-y).square().mean(-1),b['world_valid'],b['world_horizons'])
    cosine=weighted_mean(1-F.cosine_similarity(out['world_cos'],y,dim=-1),b['world_valid'],b['world_horizons'],True)
    adapter,_=model.adapter_loss(b['obs'][:,4]);zero=mean*0;fm=zero
    if model.recipe=='M2':
        device=y.device;n=len(y);g=lambda k:generator(seed,stage,step,k,device=device)
        tw=torch.sigmoid(torch.randn(n,device=device,generator=g('tw')))
        ew=torch.randn(y.shape,device=device,generator=g('ew'))
        w=(1-tw[:,None,None])*ew+tw[:,None,None]*y
        velocity=model(b,w=w,tw=tw)['world']
        fm=weighted_mean((velocity-(y-ew)).square().mean(-1),b['world_valid'],b['world_horizons'])
    cw=chain_weight(stage,step,bool(model.adapter_ready) and chain is not None and (p1_converged or engineering_chain_probe),chain_warmup,chain_ramp,chain_maximum)
    cl=zero;cc={}
    if cw:cl,cc=chain(model)
    total=mean+fm+cosine_weight*cosine+adapter_weight*adapter+cw*cl
    err=(out['mean']-y).square().mean(-1)
    per={str(int(h)):float(err[b['world_valid'] & (b['world_horizons']==h)].mean().detach()) for h in b['world_horizons'][0].unique() if (b['world_valid'] & (b['world_horizons']==h)).any()}
    return total,dict(per_horizon=per,mean=float(mean.detach()),fm=float(fm.detach()),world=float((mean+fm).detach()),cosine_short=float(cosine.detach()),adapter=float(adapter.detach()),phase=0.,action=0.,chain=float(cl.detach()),chain_weight=cw,mode='world_only',clean_aux=False,**cc)

def chain_loss(model,first,shifted,direct,k,use_predicted=True,steps=2,seed=88):
    """W-only both hops; no q/A. Stop first and one direct consistency endpoint."""
    with torch.no_grad():
        z=model.mean_rollout(first,steps,seed).detach()
        n=8 if k<=50 else 1;z=z.reshape(len(z),3,n,-1)
        if n==1:z=z.expand(-1,-1,8,-1)
    shifted=dict(shifted)
    if use_predicted:
        # First-hop is detached; adapter remains trainable on reentry.
        shifted['reentry_tokens']=model.reentry(z)
        cur=z if k<=50 else z.mean(2,keepdim=True)
        shifted['world_current']=cur.reshape(len(z),-1,z.shape[-1])
    else:
        # Teacher reentry also uses the adapter, with a genuinely available causal frame.
        shifted['reentry_tokens']=model.reentry(shifted['obs'][:,4].detach())
    pred=model.mean_rollout(shifted,steps,seed+1)
    # Shifted target must be the exact original +2k endpoint, not a nearby reselected row.
    truth=direct['world_raw'];shift_valid=shifted['world_valid']
    if shift_valid.shape!=direct['world_valid'].shape:shift_valid=shift_valid.reshape(len(pred),3,-1).all(-1)
    valid=direct['world_valid'] & shift_valid
    if k<=50 and 2*k>50:pred=pred.reshape(len(pred),3,8,-1).mean(2)
    if pred.shape[1]!=truth.shape[1]:raise ValueError('Chain resolution mismatch')
    # Valid masks are reduced to camera if coarse direct target.
    if valid.shape!=truth.shape[:2]:valid=direct['world_valid'] & shifted['world_valid'].reshape(len(pred),3,-1).all(-1)
    scale=direct['world_scale'];supervised=(((pred-truth)/scale).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
    with torch.no_grad():direct_pred=model.mean_rollout(direct,steps,seed+2).detach()
    consistency=(((pred-direct_pred)/scale).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
    return supervised+consistency,dict(chain_supervised=float(supervised.detach()),chain_gap=float(consistency.detach()),predicted_reentry=use_predicted)
