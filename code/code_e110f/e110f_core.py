"""E110-F action-head comparison; imports A2 implementation read-only."""
import hashlib
import torch
import torch.nn.functional as F
from e99w2_model import JOINTS
from e102a_contract import contract

TRAINABLE_PREFIXES=("state_proj.","action_proj.","proprio_proj.","delta_head.","grip_head.")
WEIGHT=torch.tensor([4.]*10+[2.]*15+[1.]*25)

def set_trainables(model):
    names=[]
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
        if p.requires_grad:names.append(name)
    assert names and all(any(n.startswith(x) for x in TRAINABLE_PREFIXES) for n in names)
    return tuple(names)

def tensor_sha(model, names=None):
    h=hashlib.sha256()
    allowed=set(names) if names is not None else None
    for n,p in model.named_parameters():
        if allowed is None or n in allowed:
            h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def target_delta(b):
    j=torch.as_tensor(JOINTS,device=b["action"].device)
    return (b["action"][:,:,j]-b["q_current"][:,None,j])/b["action_std"][:,:,j]

def weights(device,dtype):
    return WEIGHT.to(device=device,dtype=dtype)[None,:,None]

def deterministic_loss(model,b):
    n=len(b["obs"]);t=torch.zeros(n,device=b["obs"].device)
    a=torch.zeros_like(b["action"])
    out=model(b,a,None,t,t)
    target=target_delta(b);w=weights(target.device,target.dtype)
    delta=((out["delta"]-target).square()*w).sum()/(w.sum()*target.shape[-1]*n)
    grip=F.binary_cross_entropy_with_logits(out["grip_logits"],b["grip_label"])
    return delta+grip,{"delta":delta.detach(),"grip":grip.detach()}

def flow_loss(model,b,generator):
    n=len(b["obs"]);dev=b["obs"].device;y=target_delta(b)
    eps=torch.randn(y.shape,device=dev,dtype=y.dtype,generator=generator)
    t=torch.sigmoid(torch.randn(n,device=dev,generator=generator))
    x=(1-t[:,None,None])*eps+t[:,None,None]*y
    a=torch.zeros_like(b["action"]);a[:,:,torch.as_tensor(JOINTS,device=dev)]=x
    out=model(b,a,None,t,t);w=weights(dev,y.dtype)
    fm=((out["delta"]-(y-eps)).square()*w).sum()/(w.sum()*y.shape[-1]*n)
    z=torch.zeros(n,device=dev);gripout=model(b,torch.zeros_like(a),None,z,z)
    grip=F.binary_cross_entropy_with_logits(gripout["grip_logits"],b["grip_label"])
    return fm+grip,{"flow":fm.detach(),"grip":grip.detach()}

@torch.no_grad()
def infer_d(model,b,steps=4,seed=90):
    # The source A2 method is the locked deterministic implementation.
    return model.infer(b,"action_only",steps,seed)[0]

@torch.no_grad()
def infer_f(model,b,steps=4,seed=90,initial_noise=None):
    assert steps in (1,2,4,8)
    device=b["obs"].device;n=len(b["obs"])
    gen=torch.Generator(device=device).manual_seed(seed)
    x=(torch.randn(n,50,12,device=device,dtype=b["obs"].dtype,generator=gen) if initial_noise is None else initial_noise.to(device=device,dtype=b["obs"].dtype).clone())
    assert x.shape==(n,50,12)
    ji=torch.as_tensor(JOINTS,device=device)
    for i in range(steps):
        a=torch.zeros_like(b["action"],dtype=x.dtype);a[:,:,ji]=x
        t=torch.full((n,),i/steps,device=device,dtype=x.dtype)
        out=model(b,a,None,t,t);x=x+out["delta"]/steps
        assert torch.isfinite(x).all(),"nonfinite flow sample"
    absolute=b["q_current"][:,None].expand(-1,50,-1).clone()
    absolute[:,:,ji]=b["q_current"][:,None,ji]+x*b["action_std"][:,:,ji]
    z=torch.zeros(n,device=device);grip=model(b,torch.zeros_like(b["action"],dtype=b["obs"].dtype),None,z,z)["grip_logits"]
    c=contract();gidx=torch.as_tensor([6,13],device=device)
    absolute[:,:,gidx]=torch.where(grip>0,absolute.new_tensor(c["open_m"]),absolute.new_tensor(c["closed_m"]))
    return absolute
