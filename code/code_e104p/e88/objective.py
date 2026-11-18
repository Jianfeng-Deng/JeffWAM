import torch,torch.nn.functional as F
from e86.common import generator
from e86.objective import mode_for,contrastive

def weighted_mean(err,valid,horizons,short_only=False):
    vals=[]
    for h in horizons[0].unique():
        if short_only and h>50:continue
        mask=valid & (horizons==h);weight=mask.sum()
        if weight:vals.append((err*mask).sum()/weight)
    return torch.stack(vals).mean() if vals else err.sum()*0

def chain_weight(stage,step,enabled,warmup=128,ramp=128,maximum=.1):
    if not enabled:return 0.
    # P3 starts after completed P1; P1 ramp begins only at explicit convergence boundary.
    start=0 if stage=='P3' else warmup
    return maximum*max(0.,min(1.,(step-start)/max(1,ramp)))
def loss(model,b,stage,step,seed=88,mode=None,clean_aux_probability=.25,mode_probabilities=(.3,.3,.3,.1),chain=None,chain_warmup=128,chain_ramp=128,chain_maximum=.1,p1_converged=False,engineering_chain_probe=False):
    device=b['obs'].device;n=len(b['obs']);g=lambda k:generator(seed,stage,step,k,device=device)
    ta=torch.sigmoid(torch.randn(n,device=device,generator=g('ta')));tw=torch.sigmoid(torch.randn(n,device=device,generator=g('tw')))
    y=b['world'].detach();ew=torch.randn(y.shape,device=device,generator=g('ew'));yw=y-ew;mode=mode or ('world_only' if stage=='P1' else mode_for(step,seed,mode_probabilities))
    a=None;ya=None;ea=None
    if stage!='P1':
        ya=(b['action']-b['q_anchor'][:,None]-b['action_mean'])/b['action_std'];ea=torch.randn(ya.shape,device=device,generator=g('ea'))
        if mode=='sync':tw=ta
        if mode=='clean':ta=torch.ones_like(ta)
        a=(1-ta[:,None,None])*ea+ta[:,None,None]*ya
    w=None if mode=='action_only' else (1-tw[:,None,None])*ew+tw[:,None,None]*y
    out=model(b,a,w,ta,tw);zero=out['observation'].sum()*0;la=zero;lw=zero;lc=zero;clean_aux=False;adapter=zero;phase=zero;chainloss=zero;cc={};cw=0.
    def world_losses(o):
        fm=weighted_mean((o['world']-yw).square().mean(-1),b['world_valid'],b['world_horizons'])
        cos=weighted_mean(1-F.cosine_similarity(o['world_cos'],y,dim=-1),b['world_valid'],b['world_horizons'],True)
        return fm,cos
    if w is not None:
        lw,lc=world_losses(out);adapter,adapter_cos=model.adapter_loss(b['obs'][:,4]);pv=b['phase_valid']
        if pv.any():phase=F.cross_entropy(out['phase'][pv],b['phase_target'][pv])+F.smooth_l1_loss(out['progress'][pv],b['progress_target'][pv])
        cw=chain_weight(stage,step, bool(model.adapter_ready) and chain is not None and (p1_converged or engineering_chain_probe),chain_warmup,chain_ramp,maximum=chain_maximum)
        if cw:chainloss,cc=chain(model)
    if stage!='P1':
        if mode=='clean':action_pred=model(b,.5*ea+.5*ya,None,ta*0+.5,tw)['action']
        else:action_pred=out['action']
        wt=torch.tensor([4.]*10+[2.]*15+[1.]*25,device=device)[None,:,None];la=((action_pred-(ya-ea)).square()*wt).sum()/(wt.sum()*14*n)
        if w is not None and mode!='clean' and torch.rand((),device=device,generator=g('clean_aux')).item()<clean_aux_probability:
            clean_aux=True;aux=model(b,ya,(1-tw[:,None,None])*ew+tw[:,None,None]*y,torch.ones_like(ta),tw);aw,ac=world_losses(aux);lw=(lw+aw)/2;lc=(lc+ac)/2
    align=contrastive(model,out,b);world_total=lw+.1*lc+.1*adapter+.01*phase+cw*chainloss
    total=world_total+.01*align if stage=='P1' else la+.1*world_total+.01*align
    return total,dict(action=float(la.detach()),world=float(lw.detach()),cosine_short=float(lc.detach()),adapter=float(adapter.detach()),phase=float(phase.detach()),chain=float(chainloss.detach()),chain_weight=cw,mode=mode,clean_aux=clean_aux,**cc)

def chain_loss(model,first,shifted,direct,k,use_predicted=True,steps=2,seed=88):
    """W-only both hops; no q/A. Stop first and one direct consistency endpoint."""
    with torch.no_grad():
        z=model.rollout(first,steps,seed).detach()
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
    pred=model.rollout(shifted,steps,seed+1)
    # Shifted target must be the exact original +2k endpoint, not a nearby reselected row.
    truth=direct['world_raw'];shift_valid=shifted['world_valid']
    if shift_valid.shape!=direct['world_valid'].shape:shift_valid=shift_valid.reshape(len(pred),3,-1).all(-1)
    valid=direct['world_valid'] & shift_valid
    if k<=50 and 2*k>50:pred=pred.reshape(len(pred),3,8,-1).mean(2)
    if pred.shape[1]!=truth.shape[1]:raise ValueError('Chain resolution mismatch')
    # Valid masks are reduced to camera if coarse direct target.
    if valid.shape!=truth.shape[:2]:valid=direct['world_valid'] & shifted['world_valid'].reshape(len(pred),3,-1).all(-1)
    scale=direct['world_scale'];supervised=(((pred-truth)/scale).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
    with torch.no_grad():direct_pred=model.rollout(direct,steps,seed+2).detach()
    consistency=(((pred-direct_pred)/scale).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
    return supervised+consistency,dict(chain_supervised=float(supervised.detach()),chain_gap=float(consistency.detach()),predicted_reentry=use_predicted)
