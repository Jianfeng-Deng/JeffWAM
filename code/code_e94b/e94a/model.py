"""Frozen E94 unified readout v1. E90 inputs/attention; deterministic and FM MLP heads."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from e90.model import WorldModel as E90WorldModel, adapter_pass, SOURCES
from e86.model import Config, mask_oa
from e88.model import directional_mask, horizon_embedding

class WorldModel(E90WorldModel):
    def __init__(self,c=None,recipe='M1',readout_hidden=1024):
        super().__init__(c)
        if recipe not in ('M1','M2'): raise ValueError(recipe)
        self.recipe=recipe
        self.readout_hidden=readout_hidden
        self.mean_query=nn.Parameter(torch.randn(1,1,self.c.width)*.02)
        self.mean_head=nn.Sequential(nn.Linear(self.c.width,readout_hidden),nn.GELU(),nn.Linear(readout_hidden,self.c.latent))
        self.world_head=nn.Sequential(nn.Linear(self.c.width,readout_hidden),nn.GELU(),nn.Linear(readout_hidden,self.c.latent))
        for head in (self.mean_head,self.world_head):
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def forward_mean(self,b,a=None,ta=None):
        # Only target SHAPE is used; no target values, random noise, or FM clock.
        return self(b,a=a,w=torch.empty_like(b['world']),ta=ta,branch='mean')

    def mean_rollout(self,b,steps=4,seed=90):
        return b['world_current']+self.forward_mean(b)['mean']*b['world_scale']

    def forward(self,b,a=None,w=None,ta=None,tw=None,branch="fm"):
        if branch not in ("mean","fm"): raise ValueError(branch)
        if branch=="fm" and self.recipe=="M1": raise RuntimeError("M1 never calls FM")
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
            if branch=='mean':
                query=self.mean_query.expand(len(o),nw,-1)+he+pos[None]+self.kind[2]
            else:
                clock=self.tw(torch.stack((tw,torch.sin(tw*math.pi)),1))[:,None]
                query=self.world_proj(w)+he+pos[None]+clock+self.kind[2]
            parts.append(query);valids.append(b['world_valid'])
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
        mean=self.mean_head(wh) if wh is not None and branch=='mean' else None
        velocity=(self.world_head(wh)-w)/(1-tw[:,None,None]).clamp_min(.01) if wh is not None and branch=='fm' else None
        return dict(mean=mean,action=self.action_head(ah) if ah is not None else None,world=velocity,world_cos=cos,observation=x[:,:no],phase=self.phase_head(aux_rep),progress=self.progress_head(aux_rep).sigmoid().squeeze(-1),align=[F.normalize(head(z),dim=-1) for head,z in zip(self.align_heads,rep)],history_kept=hv,alignment_available=torch.stack((b['current_valid'].any(1),hv.any(1),torch.full((len(o),),a is not None,dtype=torch.bool,device=o.device)),1))
