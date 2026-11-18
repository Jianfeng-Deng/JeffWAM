from __future__ import annotations
import math
import torch
from torch import nn
from .config import Config,CAMERA_IDS,ENDPOINTS
from .core import UnifiedCore
from .vision import VisionPair,POOL,CAMERAS

class TimeEmbedding(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(32,d),nn.SiLU(),nn.Linear(d,d))
        self.register_buffer('freq',torch.exp(torch.linspace(0,math.log(1000),16)))
    def forward(self,t):
        f=t.float()[...,None]*self.freq; return self.net(torch.cat((f.sin(),f.cos()),-1))

def spatial_metadata():
    rows=[]
    for i,(h,w,glob) in enumerate(POOL):
        rows.extend([(i,(x+.5)/w,(y+.5)/h) for y in range(h) for x in range(w)])
        if glob: rows.append((i,.5,.5))
    return torch.tensor(rows)

class PiperWAM(nn.Module):
    def __init__(self,c=Config(),fixture=False,inference_only=False):
        super().__init__(); self.c=c; self.vision=VisionPair(c.dino_weights,fixture,target_enabled=not inference_only)
        d=c.width; self.core=UnifiedCore(c)
        self.o_proj=nn.Linear(768,d); self.a_proj=nn.Linear(14,d); self.w_proj=nn.Linear(768,d)
        self.action_head=nn.Linear(d,14); self.world_head=nn.Linear(d,768)
        self.types=nn.ParameterList([nn.Parameter(torch.randn(d)*.02) for _ in range(3)]); self.task=nn.Embedding(c.tasks,d); self.camera=nn.Embedding(3,d)
        # x/y, source box, age, physical relative time, present, full-vs-ROI
        self.visual_meta=nn.Linear(10,d); self.camera_status=nn.Linear(30,d)
        self.position=nn.Embedding(91,d); self.state_type=nn.Embedding(4,d)
        self.state_missing=nn.Embedding(4,d)
        self.q_proj=nn.Linear(14,d); self.dq_proj=nn.Linear(29,d)
        self.geo_proj=nn.Linear(1,d)  # reserved geometry slot; no unverified FK
        self.hist_proj=nn.Linear(8*16,d)
        self.clock=TimeEmbedding(d); self.physical_time=TimeEmbedding(d)
        self.register_buffer('spatial',spatial_metadata()); self.register_buffer('camera_ids',torch.tensor(CAMERA_IDS))
        self.register_buffer('end_seconds',torch.tensor(ENDPOINTS)/30.)
        self.set_stage('P3_joint')
    def set_stage(self,stage):
        self.stage=stage; self.vision.set_stage(stage)
        self.types[1].requires_grad_(stage!='P1')
        # A/state tensors remain bit-identical to S0 throughout P1.
        for module in (self.a_proj,self.action_head,self.q_proj,self.dq_proj,self.geo_proj,self.hist_proj,self.state_type,self.state_missing):
            for p in module.parameters(): p.requires_grad_(stage!='P1')
    def observation(self,batch):
        z,valid=self.vision(batch['obs_images'],batch['obs_pixel_valid'],batch['obs_crop_valid'])
        b,t,n,_=z.shape; assert (t,n)==(2,91)
        idx=self.spatial[:,0].long()
        boxes=batch['obs_boxes'][:,:,idx]; age=batch['obs_age'][:,:,idx,None]
        present=batch['obs_crop_valid'][:,:,idx,None].float()
        xy=self.spatial[None,None,:,1:].expand(b,t,-1,-1)
        seconds=batch['obs_offsets'][:,:,None,None].expand(b,t,n,1)
        roi=((idx==2)|(idx==4)).float()[None,None,:,None].expand(b,t,-1,-1)
        meta=torch.cat((xy,boxes,age,seconds,present,roi),-1)
        o=self.o_proj(z)+self.types[0]+self.camera(self.camera_ids)[None,None]+self.position.weight[None,None]+self.visual_meta(meta)
        o=o.flatten(1,2); ov=valid.flatten(1,2)
        task_id=batch.get('task_id',torch.zeros(b,dtype=torch.long,device=z.device))
        task_id=torch.where((task_id>=0)&(task_id<self.c.tasks),task_id,0)
        # Per-crop camera/age/missing signals survive even when all image patches are masked.
        status=torch.stack((batch['obs_age'],batch['obs_crop_valid'].float(),batch['obs_offsets'][:,:,None].expand(b,2,5)),-1).flatten(1)
        task=self.task(task_id)+self.camera_status(status)+self.types[0]
        if self.stage=='P1':
            state=z.new_zeros(b,4,self.c.width); sv=torch.zeros(b,4,dtype=torch.bool,device=z.device)
        else:
            sv=batch['state_valid'].bool()
            values=(self.q_proj(batch['q']),self.dq_proj(torch.cat((batch['dq'],batch['q_valid'].float(),batch['dq_valid'].float()[:,None]),-1)),self.geo_proj(batch['geo']),self.hist_proj(batch['history'].flatten(1)))
            state=torch.stack(values,1)+self.state_type.weight[None]+self.types[0]
            state=torch.where(sv[:,:,None],state,self.state_missing.weight[None])
        return torch.cat((o,state,task[:,None]),1),torch.cat((ov,sv,torch.ones(b,1,dtype=torch.bool,device=z.device)),1)
    def generated(self,a,w,ta,tw,w_valid=None):
        parts=[]; valids=[]; na=0
        if a is not None:
            assert self.stage!='P1'; na=a.shape[1]
            seconds=torch.arange(1,na+1,device=a.device).float()/30
            parts.append(self.a_proj(a)+self.types[1]+self.clock(ta)[:,None]+self.physical_time(seconds)[None])
            valids.append(torch.ones(a.shape[:2],dtype=torch.bool,device=a.device))
        if w is not None:
            assert tw is not None
            b=w.shape[0]; x=self.w_proj(w.reshape(b,273,768))
            x=x+self.types[2]+self.clock(tw)[:,None]
            x=x+self.position.weight.repeat(3,1)[None]+self.camera(self.camera_ids).repeat(3,1)[None]
            x=x+self.physical_time(self.end_seconds).repeat_interleave(91,0)[None]
            parts.append(x); valids.append(torch.ones(b,273,dtype=torch.bool,device=x.device) if w_valid is None else w_valid.reshape(b,273))
        if not parts: raise ValueError('At least one generated modality required')
        return torch.cat(parts,1),torch.cat(valids,1),na
    def predict(self,o,ov,a=None,w=None,ta=None,tw=None,w_valid=None,cache=None):
        g,gv,na=self.generated(a,w,ta,tw,w_valid)
        oh,gh=self.core(o,g,ov,gv) if cache is None else self.core.cached(g,gv,cache)
        av=self.action_head(gh[:,:na]) if a is not None else None
        wv=self.world_head(gh[:,na:]).reshape(-1,3,91,768) if w is not None else None
        return av,wv,oh
    @torch.no_grad()
    def world_target(self,batch):
        return self.vision(batch['future_images'],batch['future_pixel_valid'],batch['future_crop_valid'],target=True)
    def parameter_counts(self):
        return {'total':sum(p.numel() for p in self.parameters()),'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),'core':sum(p.numel() for p in self.core.parameters()),'online_visual':sum(p.numel() for p in self.vision.online.parameters()),'lora':sum(p.numel() for p in self.vision.online.parameters() if p.requires_grad),'target':sum(p.numel() for p in self.vision.target.parameters()) if self.vision.target is not None else 0}
