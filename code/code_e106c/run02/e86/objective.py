import torch
import torch.nn.functional as F
from .common import generator
MODES=('sync','independent','action_only','clean')
def mode_for(step,seed=86,probabilities=(.3,.3,.3,.1)):
    if len(probabilities)!=4 or any(p<0 for p in probabilities) or abs(sum(probabilities)-1)>1e-8:raise ValueError("Mode probabilities must sum to one")
    u=torch.rand((),generator=generator(seed,step,'mode')).item()
    total=0.
    for name,p in zip(MODES,probabilities):
        total+=p
        if u<total:return name
    return MODES[-1]
def contrastive(model,out,b):
    base_valid=b.get('align_valid',b['text_valid'].any(1));losses=[]
    for head,r in enumerate(out['align']):
        valid=base_valid & out['alignment_available'][:,head];idx=valid.nonzero().flatten()
        if len(idx)<2:continue
        frozen=b.get('align_text',b['text'][:,0]).detach()[idx];target=F.normalize(model.align_text(frozen),dim=-1)
        # Fixed exact frozen-text identity defines positives. Trainable projection cannot relabel pairs.
        positives=(frozen[:,None,:]==frozen[None,:,:]).all(-1);allowed=torch.ones_like(positives)
        if 'parent_index' in b:
            same_parent=b['parent_index'][idx,None]==b['parent_index'][None,idx]
            allowed=~same_parent | positives  # Different motions of one parent are neither positives nor negatives.
        scores=r[idx]@target.T/.1;positive=scores.masked_fill(~positives,-torch.inf);denominator=scores.masked_fill(~allowed,-torch.inf)
        forward=-(torch.logsumexp(positive,1)-torch.logsumexp(denominator,1)).mean();reverse=-(torch.logsumexp(positive,0)-torch.logsumexp(denominator,0)).mean()
        losses.append((forward+reverse)*.5)
    return sum(losses)/len(losses) if losses else out['observation'].sum()*0

def loss(model,b,stage,step,seed=86,mode=None,clean_aux_probability=.25,mode_probabilities=(.3,.3,.3,.1)):
    if not 0<=clean_aux_probability<=1:raise ValueError("Invalid clean aux probability")
    device=b['obs'].device;n=len(b['obs']);g=lambda k:generator(seed,stage,step,k,device=device)
    ta=torch.sigmoid(torch.randn(n,device=device,generator=g('ta')));tw=torch.sigmoid(torch.randn(n,device=device,generator=g('tw')))
    y=b['world'].detach();ew=torch.randn(y.shape,device=device,generator=g('ew'));yw=y-ew
    mode=mode or ('world_only' if stage=='P1' else mode_for(step,seed,mode_probabilities))
    action=None;ea=None;ya=None
    if stage!='P1':
        ya=(b['action']-b['q_anchor'][:,None]-b['action_mean'])/b['action_std'];ea=torch.randn(ya.shape,device=device,generator=g('ea'))
        if mode=='sync':tw=ta
        if mode=='clean':ta=torch.ones_like(ta)
        action=(1-ta[:,None,None])*ea+ta[:,None,None]*ya
    w=None if mode=='action_only' else (1-tw[:,None,None])*ew+tw[:,None,None]*y
    out=model(b,action,w,ta,tw);zero=out['observation'].sum()*0
    la=zero;lw=zero;lc=zero;clean_aux=False
    def world_losses(out):
        v=b['world_valid'];err=(out['world']-yw).square().mean(-1);weights=torch.tensor([.5,.3,.2],device=device).repeat_interleave(24)[None]*v
        fm=(err*weights).sum()/weights.sum().clamp_min(1)
        cos=((1-F.cosine_similarity(out['world_cos'],y,dim=-1))*weights).sum()/weights.sum().clamp_min(1)
        return fm,cos
    if w is not None:lw,lc=world_losses(out)
    if stage!='P1':
        if mode=='clean':
            # A recorded action is a condition, not a velocity target at tau=1.
            clean_action_out=model(b,(1-(ta*.0+.5))[:,None,None]*ea+.5*ya,None,ta*.0+.5,tw)
            action_pred=clean_action_out['action']
        else:action_pred=out['action']
        wt=torch.tensor([4.]*10+[2.]*15+[1.]*25,device=device)[None,:,None]
        per=((action_pred-(ya-ea)).square()*wt).sum((1,2))/(wt.sum()*14)
        # Optional quality loss weights; callers omit it by default.
        weights=b.get('quality_weight',torch.ones(n,device=device));la=(per*weights).sum()/weights.sum().clamp_min(1)
        if w is not None and mode!='clean' and torch.rand((),device=device,generator=g('clean_aux')).item()<clean_aux_probability:
            clean_aux=True
            aux=model(b,ya,(1-tw[:,None,None])*ew+tw[:,None,None]*y,torch.ones_like(ta),tw)
            aw,ac=world_losses(aux);lw=(lw+aw)*.5;lc=(lc+ac)*.5
    align=contrastive(model,out,b)
    total=(lw+.1*lc+.01*align) if stage=='P1' else la+.1*(lw+.1*lc)+.01*align
    return total,dict(action=float(la.detach()),world=float(lw.detach()),cosine=float(lc.detach()),alignment=float(align.detach()),mode=mode,clean_aux=clean_aux)
