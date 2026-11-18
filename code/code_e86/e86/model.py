from dataclasses import dataclass,asdict
import math,torch
from torch import nn
import torch.nn.functional as F
@dataclass
class Config:
    width:int=256
    layers:int=6
    heads:int=8
    ffn:int=1024
    latent:int=768
    horizon:int=50
    history_drop:float=.15
    state_drop:float=.1
    state_noise:float=.02
    quality_condition:bool=False
class AttentionPool(nn.Module):
    def __init__(self,dim,out=8,width=256):
        super().__init__();self.proj=nn.Linear(dim,width);self.query=nn.Parameter(torch.randn(out,width)*.02);self.attn=nn.MultiheadAttention(width,8,batch_first=True)
    def forward(self,x):
        shape=x.shape[:-2];x=self.proj(x.flatten(0,-3));q=self.query[None].expand(x.shape[0],-1,-1)
        y=self.attn(q,x,x,need_weights=False)[0];return y.reshape(*shape,*y.shape[-2:])
def mask_oa(n_o,n_a,n_w,device=None):
    m=torch.zeros(n_o+n_a+n_w,n_o+n_a+n_w,dtype=torch.bool,device=device);m[:n_o,n_o:]=True;return m
class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.n1=nn.LayerNorm(c.width);self.n2=nn.LayerNorm(c.width);self.attn=nn.MultiheadAttention(c.width,c.heads,batch_first=True);self.ff=nn.Sequential(nn.Linear(c.width,c.ffn),nn.GELU(),nn.Linear(c.ffn,c.width))
    def forward(self,x,mask,invalid):
        y=self.n1(x);x=x+self.attn(y,y,y,attn_mask=mask,key_padding_mask=invalid,need_weights=False)[0];return x+self.ff(self.n2(x))
class WAM(nn.Module):
    def __init__(self,c=None):
        super().__init__();self.c=c or Config();c=self.c
        self.pool=AttentionPool(c.latent,8,c.width) # each history frame: 3x8 raw grid -> 8 learned tokens
        self.current=AttentionPool(c.latent,8,c.width) # each camera retains 8 tokens
        self.text_proj=nn.Linear(c.latent,c.width);self.null_text=nn.Parameter(torch.zeros(1,1,c.width))
        self.camera=nn.Parameter(torch.randn(3,1,c.width)*.02);self.obs_age=nn.Linear(2,c.width)
        self.hist_time=nn.Parameter(torch.randn(4,1,c.width)*.02);self.history_camera=nn.Parameter(torch.randn(3,1,c.latent)*.02)
        self.state_proj=nn.Linear(14,c.width);self.action_proj=nn.Linear(14,c.width);self.world_proj=nn.Linear(c.latent,c.width)
        self.a_pos=nn.Parameter(torch.randn(50,c.width)*.02);self.w_pos=nn.Parameter(torch.randn(72,c.width)*.02)
        self.ta=nn.Sequential(nn.Linear(2,c.width),nn.SiLU(),nn.Linear(c.width,c.width));self.tw=nn.Sequential(nn.Linear(2,c.width),nn.SiLU(),nn.Linear(c.width,c.width))
        self.kind=nn.Parameter(torch.randn(3,c.width)*.02);self.quality=nn.Embedding(4,c.width) if c.quality_condition else None
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)]);self.norm=nn.LayerNorm(c.width)
        self.action_head=nn.Linear(c.width,14);self.world_head=nn.Linear(c.width,c.latent);self.world_cos_head=nn.Linear(c.width,c.latent)
        self.align_heads=nn.ModuleList([nn.Linear(c.width,128) for _ in range(3)]);self.align_text=nn.Linear(c.latent,128)
    def observation(self,b,training=None):
        training=self.training if training is None else training
        cur=self.current(b['obs'][:,4])+self.camera[None]
        cur=cur+self.obs_age(b['obs_meta'][:,4])[:,:,None]
        hist=self.pool((b['obs'][:,:4]+self.history_camera[None,None]).flatten(2,3))+self.hist_time[None]
        hist=hist+self.obs_age(b['obs_meta'][:,:4].mean(2))[:,:,None]
        hv=b['history_valid'].clone()
        if training:hv=hv & (torch.rand(len(hv),1,device=hv.device)>=self.c.history_drop)
        hist=hist*hv[:,:,None,None]
        t=self.text_proj(b['text']);tv=b['text_valid']
        pos=torch.arange(t.shape[1],device=t.device,dtype=t.dtype)[:,None];freq=torch.exp(torch.arange(0,self.c.width,2,device=t.device,dtype=t.dtype)*(-math.log(10000.)/self.c.width));phase=pos*freq[None];encoding=torch.stack((phase.sin(),phase.cos()),-1).flatten(1)*.02
        t=t+encoding[None];t=torch.where(tv[:,:,None],t,self.null_text)
        # text has a permanent NULL slot, making fully missing conditions safe.
        ts=torch.cat((self.null_text.expand(len(t),-1,-1),t),1)
        parts=[cur.flatten(1,2),hist.flatten(1,2),ts]
        valid=[b['current_valid'][:,:,None].expand(-1,-1,8).flatten(1),hv[:,:,None].expand(-1,-1,8).flatten(1),torch.cat((torch.ones(len(t),1,dtype=torch.bool,device=t.device),tv),1)]
        if self.quality is not None:parts.append(self.quality(b.get('quality_flag',torch.zeros(len(t),dtype=torch.long,device=t.device)))[:,None]);valid.append(torch.ones(len(t),1,dtype=torch.bool,device=t.device))
        return torch.cat(parts,1)+self.kind[0],torch.cat(valid,1),hv
    def forward(self,b,a=None,w=None,ta=None,tw=None):
        o,ov,hv=self.observation(b);parts=[o];valids=[ov];na=0;nw=0;state=None
        if a is not None:
            state=b['q_normalized'].clone();sv=b.get('state_valid',torch.ones(len(o),dtype=torch.bool,device=o.device)).clone()
            if self.training:
                state=state+torch.randn_like(state)*self.c.state_noise;sv &= torch.rand(len(o),device=o.device)>=self.c.state_drop
            state=self.state_proj(state*sv[:,None])[:,None]+self.kind[1]
            state=state*sv[:,None,None]
            clock=self.ta(torch.stack((ta,torch.sin(ta*math.pi)),1))[:,None]
            at=self.action_proj(a)+self.a_pos[None]+clock+self.kind[1]
            parts.extend((state,at));valids.extend((sv[:,None],torch.ones(a.shape[:2],dtype=torch.bool,device=o.device)));na=51
        if w is not None:
            nw=w.shape[1];clock=self.tw(torch.stack((tw,torch.sin(tw*math.pi)),1))[:,None]
            parts.append(self.world_proj(w)+self.w_pos[None,:nw]+clock+self.kind[2]);valids.append(b.get('world_valid',torch.ones(w.shape[:2],dtype=torch.bool,device=o.device)))
        x=torch.cat(parts,1);m=mask_oa(len(o[0]),na,nw,o.device);invalid=~torch.cat(valids,1)
        for block in self.blocks:x=block(x,m,invalid)
        x=self.norm(x);no=o.shape[1]
        ah=x[:,no+1:no+51] if a is not None else None;wh=x[:,-nw:] if nw else None
        # Masked means avoid padded language/history being used as alignment targets.
        def mean(z,v):return (z*v[:,:,None]).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
        image=mean(x[:,:24],ov[:,:24]);history=mean(x[:,24:56],ov[:,24:56])
        rep=[image,history,ah.mean(1) if ah is not None else image*0]
        return dict(action=self.action_head(ah) if ah is not None else None,world=self.world_head(wh) if wh is not None else None,world_cos=self.world_cos_head((x[:,:no]*ov[:,:,None]).sum(1,keepdim=True)/ov.sum(1,keepdim=True)[:,:,None].clamp_min(1)+self.w_pos[None,:nw]) if wh is not None else None,observation=x[:,:no],align=[F.normalize(h(z),dim=-1) for h,z in zip(self.align_heads,rep)],history_kept=hv,alignment_available=torch.stack((b['current_valid'].any(1),hv.any(1),torch.full((len(o),),a is not None,dtype=torch.bool,device=o.device)),1))
    @torch.no_grad()
    def infer(self,b,mode='action_only',steps=10,seed=86):
        if mode not in ('action_only','joint'):raise ValueError(mode)
        device=b['obs'].device;g=torch.Generator(device=device).manual_seed(seed)
        a=torch.randn(len(b['obs']),50,14,device=device,generator=g)
        w=torch.randn(len(a),72,self.c.latent,device=device,generator=torch.Generator(device=device).manual_seed(seed+100003)) if mode=='joint' else None
        for i in range(steps):
            tau=torch.full((len(a),),i/steps,device=device);out=self(b,a,w,tau,tau);a=a+out['action']/steps
            if w is not None:w=w+out['world']/steps
        return b['action_mean']+a*b['action_std']+b['q_anchor'][:,None],w
