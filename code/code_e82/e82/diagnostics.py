import torch
from .objective import action_loss,world_loss
from .randomness import Streams

def gradient_diagnostic(model,batch,stats):
    o,ov=model.observation(stats.norm_state(batch)); w,valid=model.world_target(batch); w=stats.norm_world(w)
    a=stats.norm_action(batch['action'],batch['q']); n=a.shape[0]; device=a.device
    s=Streams(model.c.seed,'fixed_train_diagnostic',0)
    ea=s.normal('epsilonA',a.shape,device);ew=s.normal('epsilonW',w.shape,device);t=torch.full((n,),.5,device=device)
    av,wv,_=model.predict(o,ov,.5*(a+ea),.5*(w+ew),t,t,w_valid=batch.get('world_structure_valid'))
    la=action_loss(av,a-ea).mean();lw=world_loss(wv,w-ew,valid).mean()
    groups={'core':[p for p in model.core.parameters() if p.requires_grad],'lora':[p for p in model.vision.online.parameters() if p.requires_grad]};out={}
    for name,params in groups.items():
        if not params:continue
        ga=torch.autograd.grad(la,params,retain_graph=True,allow_unused=True);gw=torch.autograd.grad(lw,params,retain_graph=True,allow_unused=True)
        a2=sum((g.float().square().sum() for g in ga if g is not None),la.new_zeros(()));w2=sum((g.float().square().sum() for g in gw if g is not None),la.new_zeros(()))
        dot=sum(((a.float()*w.float()).sum() for a,w in zip(ga,gw) if a is not None and w is not None),la.new_zeros(()))
        out[name]=dict(LA_norm=float(a2.sqrt()),LW_norm=float(w2.sqrt()),cosine=float(dot/(a2*w2).sqrt().clamp_min(1e-12)),lambda_gW_over_gA=float(model.c.lambda_world*w2.sqrt()/a2.sqrt().clamp_min(1e-12)))
    return out

def stoploss_monitor(record):
    """Recorded-feedback/human dev metrics only; no world-loss or generated label selection."""
    if record.get('truth_source') not in ('human','recorded_feedback'):raise ValueError('Invalid stop-loss truth source')
    if record.get('hard_physical_violation'):return 'raw action/known hardware-limit anomaly'
    points=record.get('aligned_points',[])
    if len(points)>=2 and all(p['joint_step']>=1000 and p['P_over_R_K8']>1.05 for p in points[-2:]):return 'two aligned dev checkpoints: P K8 degradation >5%'
    ratios=record.get('lambda_grad_ratio',[])
    if len(ratios)>=3 and all(x>1 for x in ratios[-3:]) and record.get('action_dev_worsening'):return 'three diagnostics world gradient dominance with action dev worsening'
    return None
