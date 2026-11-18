from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

CROPS=('top','left','left_roi','right','right_roi')
CAMERAS=(0,1,1,2,2)
POOL=((4,6,True),(4,4,True),(4,4,False),(4,4,True),(4,4,False))

class LoRAQKV(nn.Module):
    """Only Q/V increments of the fused QKV. K/base parameters stay frozen."""
    def __init__(self,base,rank=8):
        super().__init__(); self.base=base; d=base.in_features
        self.q_a=nn.Linear(d,rank,bias=False); self.q_b=nn.Linear(rank,d,bias=False)
        self.v_a=nn.Linear(d,rank,bias=False); self.v_b=nn.Linear(rank,d,bias=False)
        nn.init.zeros_(self.q_b.weight); nn.init.zeros_(self.v_b.weight)
        for p in base.parameters(): p.requires_grad_(False)
    def forward(self,x):
        q,k,v=self.base(x).chunk(3,-1)
        return torch.cat((q+self.q_b(self.q_a(x)),k,v+self.v_b(self.v_a(x))),-1)

def load_dino(weights):
    """Local original Meta checkpoint only. timm never downloads pretrained assets."""
    import timm
    if not Path(weights).is_file(): raise FileNotFoundError(f'Local DINO missing; no download allowed: {weights}')
    m=timm.create_model('vit_base_patch14_dinov2',pretrained=False,dynamic_img_size=True,num_classes=0)
    sd=torch.load(weights,map_location='cpu',weights_only=True)
    r=m.load_state_dict(sd,strict=False)
    if r.missing_keys or set(r.unexpected_keys)-{'mask_token'}: raise ValueError(str(r))
    return m

class DinoEncoder(nn.Module):
    def __init__(self,model):
        super().__init__(); self.model=model
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self,x):
        h,w=x.shape[-2:]
        assert h%14==0 and w%14==0
        x=(x-self.mean)/self.std
        t=self.model.forward_features(x)[:,self.model.num_prefix_tokens:]
        return t.transpose(1,2).reshape(x.shape[0],768,h//14,w//14)

class FixtureEncoder(nn.Module):
    """CPU contract fixture, explicitly not DINO or a production weight substitute."""
    def __init__(self):
        super().__init__(); self.stem=nn.Conv2d(3,32,14,14)
        self.qkv=LoRAQKV(nn.Linear(32,96),8); self.out=nn.Linear(32,768)
        for n,p in self.named_parameters(): p.requires_grad_('q_a' in n or 'q_b' in n or 'v_a' in n or 'v_b' in n)
    def forward(self,x):
        z=self.stem(x).permute(0,2,3,1); q,k,v=self.qkv(z).chunk(3,-1)
        return self.out(q+v+k).permute(0,3,1,2)

class VisionPair(nn.Module):
    def __init__(self,weights=None,fixture=False,target_enabled=True):
        super().__init__()
        if fixture:
            self.online=FixtureEncoder(); self.target=deepcopy(self.online) if target_enabled else None
        else:
            base=load_dino(weights)
            for p in base.parameters(): p.requires_grad_(False)
            self.target=DinoEncoder(deepcopy(base)) if target_enabled else None
            for block in base.blocks[-2:]: block.attn.qkv=LoRAQKV(block.attn.qkv,8)
            self.online=DinoEncoder(base)
        if self.target is not None:
            for p in self.target.parameters():p.requires_grad_(False)
            self.target.eval()
        self.fixture=fixture
    def train(self,mode=True):
        super().train(mode)
        if self.target is not None:self.target.eval()
        return self
    def set_stage(self,stage):
        for n,p in self.online.named_parameters():
            p.requires_grad_(stage!='P1' and any(s in n for s in ('q_a','q_b','v_a','v_b')))
    @staticmethod
    def pool(grid,pixel_valid,hw,global_token):
        coverage=F.avg_pool2d(pixel_valid.float(),14,14)
        num=F.adaptive_avg_pool2d(grid*coverage,hw)
        den=F.adaptive_avg_pool2d(coverage,hw)
        z=(num/den.clamp_min(1e-6)).flatten(2).transpose(1,2)
        valid=den.flatten(1)>0
        if global_token:
            glob=(grid*coverage).sum((2,3))/coverage.sum((2,3)).clamp_min(1e-6)
            z=torch.cat((z,glob[:,None]),1)
            valid=torch.cat((valid,coverage.sum((1,2,3))[:,None]>0),1)
        return z,valid
    def encode(self,images,pixel_valid,crop_valid,target=False):
        encoder=self.target if target else self.online
        if encoder is None:raise RuntimeError('Deployment policy contains no target encoder')
        b,t=images[0].shape[:2]; zs=[]; vs=[]
        # Each crop is encoded from its own native source transform.
        for i,(image,mask,pool) in enumerate(zip(images,pixel_valid,POOL)):
            flat=image.flatten(0,1); mask=mask.flatten(0,1)
            g=encoder(flat); z,v=self.pool(g,mask,pool[:2],pool[2])
            v=v.reshape(b,t,-1)&crop_valid[:,:,i,None]
            zs.append(z.reshape(b,t,-1,768)); vs.append(v)
        return torch.cat(zs,2),torch.cat(vs,2)
    def forward(self,images,pixel_valid,crop_valid,target=False):
        if target:
            with torch.no_grad(): return self.encode(images,pixel_valid,crop_valid,True)
        return self.encode(images,pixel_valid,crop_valid,False)
