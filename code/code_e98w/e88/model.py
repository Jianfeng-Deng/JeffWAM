"""v3.1 residual multi-horizon W, directional attention and causal reentry."""
import math,torch
from torch import nn
import torch.nn.functional as F
from e86.model import WAM,Config,AttentionPool,mask_oa

def horizon_embedding(h,width):
    # Continuous actual source offset, not learned integer horizon IDs.
    freq=torch.exp(torch.arange(width//2,device=h.device,dtype=h.dtype)*(-math.log(10000.)/(width//2-1)))
    phase=(h/300.)[...,None]*freq*2*math.pi
    return torch.cat((phase.sin(),phase.cos()),-1)
def directional_mask(no,na,h):
    m=mask_oa(no,na,len(h),h.device);long=h>50;start=no+na
    # Long W cannot read A either: A can read short W and would be an indirect leak.
    if na:m[start:,:][long,no:start]=True
    m[start:,start:] |= long[:,None] & ~long[None,:]
    return m
class WorldModel(WAM):
    def __init__(self,c=None):
        super().__init__(c);c=self.c
        self.horizon_proj=nn.Sequential(nn.Linear(c.width,c.width),nn.SiLU(),nn.Linear(c.width,c.width))
        self.grid_pos=nn.Parameter(torch.randn(24,c.width)*.02)
        self.reentry=AttentionPool(c.latent,8,c.width)
        self.phase_head=nn.Linear(c.width,32);self.progress_head=nn.Linear(c.width,1)
        self.register_buffer('adapter_ready',torch.tensor(False))
        # Preserve all 768 noise coordinates outside the 256-wide transformer bottleneck.
        nn.init.zeros_(self.world_head.weight);nn.init.zeros_(self.world_head.bias)
    def observation(self,b,training=None):
        o,ov,hv=super().observation(b,training)
        if 'reentry_tokens' in b:
            cur=b['reentry_tokens']+self.camera[None]+self.obs_age(b['obs_meta'][:,4])[:,:,None]
            o=torch.cat((cur.flatten(1,2)+self.kind[0],o[:,24:]),1)
        return o,ov,hv
    def adapter_loss(self,raw):
        target=self.current(raw).detach();pred=self.reentry(raw.detach())
        mse=(pred-target).square().mean()/target.square().mean().clamp_min(1e-6)
        cos=F.cosine_similarity(pred,target,dim=-1).mean()
        return mse,cos
    def forward(self,b,a=None,w=None,ta=None,tw=None):
        # Legacy action-only API keeps physical W deletion.
        o,ov,hv=self.observation(b);parts=[o];valids=[ov];na=0;nw=0
        if a is not None:
            state=b['q_normalized'].clone();sv=b.get('state_valid',torch.ones(len(o),dtype=torch.bool,device=o.device)).clone()
            if self.training:
                state+=torch.randn_like(state)*self.c.state_noise;sv &= torch.rand(len(o),device=o.device)>=self.c.state_drop
            state=self.state_proj(state*sv[:,None])[:,None]+self.kind[1];state*=sv[:,None,None]
            at=self.action_proj(a)+self.a_pos[None]+self.ta(torch.stack((ta,torch.sin(ta*math.pi)),1))[:,None]+self.kind[1]
            parts.extend((state,at));valids.extend((sv[:,None],torch.ones(a.shape[:2],dtype=torch.bool,device=o.device)));na=51
        if w is not None:
            nw=w.shape[1];hs=b['world_horizons'];hs=hs.expand(len(o),-1) if hs.ndim==1 else hs
            he=self.horizon_proj(horizon_embedding(hs.to(o.dtype),self.c.width));pos=self.grid_pos[b['world_positions']]
            clock=self.tw(torch.stack((tw,torch.sin(tw*math.pi)),1))[:,None]
            parts.append(self.world_proj(w)+he+pos[None]+clock+self.kind[2]);valids.append(b['world_valid'])
        x=torch.cat(parts,1);no=o.shape[1]
        m=directional_mask(no,na,hs[0]) if w is not None else mask_oa(no,na,0,o.device)
        if w is not None and not torch.equal(hs,hs[0].expand_as(hs)):raise ValueError('Shared horizon layout required per batch')
        invalid=~torch.cat(valids,1)
        for block in self.blocks:x=block(x,m,invalid)
        x=self.norm(x);ah=x[:,no+1:no+51] if a is not None else None;wh=x[:,-nw:] if nw else None
        def mean(z,v):return (z*v[:,:,None]).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
        image=mean(x[:,:24],ov[:,:24]);history=mean(x[:,24:56],ov[:,24:56]);rep=[image,history,ah.mean(1) if ah is not None else image*0]
        # Aux uses O-only causal state, not noisy W which contains future target mixture.
        causal=mean(x[:,:no],ov)
        world_clean=causal[:,None]+he+pos[None] if wh is not None else None
        cos=self.world_cos_head(world_clean) if wh is not None else None
        aux_rep=world_clean.mean(1) if wh is not None else causal*0
        velocity=(self.world_head(wh)-w)/(1-tw[:,None,None]).clamp_min(.01) if wh is not None else None
        return dict(action=self.action_head(ah) if ah is not None else None,world=velocity,world_cos=cos,observation=x[:,:no],phase=self.phase_head(aux_rep),progress=self.progress_head(aux_rep).sigmoid().squeeze(-1),align=[F.normalize(head(z),dim=-1) for head,z in zip(self.align_heads,rep)],history_kept=hv,alignment_available=torch.stack((b['current_valid'].any(1),hv.any(1),torch.full((len(o),),a is not None,dtype=torch.bool,device=o.device)),1))
    def infer(self,b,mode='action_only',steps=10,seed=86):
        # Joint requires v3.1 target layout. Absolute actions retain e86's scale contract.
        device=b['obs'].device;g=torch.Generator(device=device).manual_seed(seed)
        a=torch.randn(len(b['obs']),50,14,device=device,generator=g)
        w=torch.randn(b['world'].shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+100003)) if mode=='joint' else None
        for i in range(steps):
            tau=torch.full((len(a),),i/steps,device=device);out=self(b,a,w,tau,tau);a=a+out['action']/steps
            if w is not None:w=w+out['world']/steps
        return b['action_mean']+a*b['action_std']+b['q_anchor'][:,None],w
    def rollout(self,b,steps=4,seed=88):
        device=b['obs'].device;w=torch.randn(b['world'].shape,device=device,generator=torch.Generator(device=device).manual_seed(seed))
        for i in range(steps):
            t=torch.full((len(w),),i/steps,device=device);w=w+self(b,w=w,tw=t)['world']/steps
        return b['world_current']+w*b['world_scale']
