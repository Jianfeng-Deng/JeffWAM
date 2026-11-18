from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); self.heads=c.heads; self.d=c.width//c.heads
        self.n1=nn.LayerNorm(c.width); self.qkv=nn.Linear(c.width,3*c.width)
        self.out=nn.Linear(c.width,c.width); self.n2=nn.LayerNorm(c.width)
        self.ff=nn.Sequential(nn.Linear(c.width,c.ffn),nn.GELU(),nn.Linear(c.ffn,c.width))
    def project(self,x):
        b,n,d=x.shape
        return self.qkv(self.n1(x)).reshape(b,n,3,self.heads,self.d).permute(2,0,3,1,4).unbind(0)
    def attend(self,q,k,v,valid):
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:],dropout_p=0.0)
        return self.out(a.transpose(1,2).flatten(2))
    def finish(self,x,att,valid):
        x=x+att; x=x+self.ff(self.n2(x)); return x*valid[...,None]
    def obs(self,o,valid):
        q,k,v=self.project(o)
        return self.finish(o,self.attend(q,k,v,valid),valid),(k,v)
    def generated(self,g,gv,kv,ov):
        if g.shape[1]==0: return g
        q,k,v=self.project(g); ok,ovalue=kv
        return self.finish(g,self.attend(q,torch.cat((ok,k),2),torch.cat((ovalue,v),2),torch.cat((ov,gv),1)),gv)

class UnifiedCore(nn.Module):
    def __init__(self,c=Config()):
        super().__init__(); self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm=nn.LayerNorm(c.width)
    def forward(self,o,g,ov,gv):
        for b in self.blocks:
            o,kv=b.obs(o,ov); g=b.generated(g,gv,kv,ov)
        return self.norm(o)*ov[...,None],self.norm(g)*gv[...,None]
    def cache(self,o,ov):
        if self.training: raise RuntimeError('O cache is eval-only; training must retain graph')
        cache=[]
        for b in self.blocks: o,kv=b.obs(o,ov); cache.append(kv)
        return (tuple(cache),ov,self.norm(o)*ov[...,None])
    def cached(self,g,gv,cache):
        pairs,ov,o=cache
        for b,kv in zip(self.blocks,pairs): g=b.generated(g,gv,kv,ov)
        return o,self.norm(g)*gv[...,None]
