import numpy as np,torch
from torch import nn
import torch.nn.functional as F
class AlignmentHead(nn.Module):
    def __init__(self,input_dim=6912,text_dim=768):
        super().__init__();self.net=nn.Sequential(nn.Linear(input_dim,96),nn.GELU(),nn.Linear(96,text_dim))
        assert sum(p.numel() for p in self.parameters())<=1_000_000
    def forward(self,x):return F.normalize(self.net(x),dim=-1)
class DirectionHead(nn.Module):
    def __init__(self,input_dim=6912,n_classes=24):super().__init__();self.net=nn.Linear(input_dim,n_classes)
    def forward(self,x):return self.net(x)
class DisplacementHead(nn.Module):
    def __init__(self,input_dim=6912,state_dim=14,text_dim=768,use_language=True):
        super().__init__();self.use_language=use_language
        self.obs=nn.Sequential(nn.Linear(input_dim+state_dim,64),nn.GELU())
        self.lang=nn.Linear(text_dim,64) if use_language else None
        self.out=nn.Sequential(nn.Linear(64*(2 if use_language else 1),64),nn.GELU(),nn.Linear(64,3))
    def forward(self,x,state,text=None):
        z=self.obs(torch.cat([x,state],-1))
        if self.use_language:z=torch.cat([z,self.lang(text)],-1)
        return self.out(z)
def observations(global_features,valid):
    g=np.asarray(global_features,dtype=np.float32);v=np.asarray(valid,dtype=bool)
    assert g.ndim==4 and v.shape==g.shape[:-1]
    now=g[:,-1];history=(g[:,:-1]*v[:,:-1,:,None]).sum(1)
    count=np.maximum(v[:,:-1].sum(1),1).astype('f4')[...,None];history=history/count
    # Missing history is represented by current features, so diff is zero.
    history=np.where((v[:,:-1].sum(1)>0)[...,None],history,now)
    return np.concatenate([now.reshape(len(g),-1),history.reshape(len(g),-1),(now-history).reshape(len(g),-1)],-1).astype('f4')
