import torch
from .data import collate

def batch(n=2):
    g=torch.Generator().manual_seed(86)
    rand=lambda *s:torch.randn(*s,generator=g)
    q=rand(n,14)*.1;anchor=q.clone();anchor[:,[6,13]]=0
    return dict(obs=rand(n,5,3,8,768),obs_meta=torch.zeros(n,5,3,2),history_valid=torch.ones(n,4,dtype=torch.bool),current_valid=torch.ones(n,3,dtype=torch.bool),world=rand(n,72,768),world_valid=torch.ones(n,72,dtype=torch.bool),text=rand(n,2,768),text_valid=torch.ones(n,2,dtype=torch.bool),q_normalized=q,q_anchor=anchor,action=anchor[:,None]+rand(n,50,14)*.01,action_mean=torch.zeros(n,50,14),action_std=torch.ones(n,50,14)*.1,parent_index=torch.arange(n),align_text=rand(n,768),align_valid=torch.ones(n,dtype=torch.bool),state_valid=torch.ones(n,dtype=torch.bool))
