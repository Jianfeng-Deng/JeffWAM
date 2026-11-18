"""Corrected endpoint-only chain. Predicted hop uses only original observation and plan."""
import torch
from e99w2_model import trajectory,pose_torch

def open_second(first,k):
    # k25/50: all four shifted historical slots are strictly after original t.
    assert k in (25,50)
    b=dict(first)
    b['obs']=torch.zeros_like(first['obs'])
    b['history_valid']=torch.zeros_like(first['history_valid'])
    b['obs_meta']=torch.zeros_like(first['obs_meta'])
    for j,off in enumerate((-24,-18,-12,-6,0)):b['obs_meta'][:,j,:,0]=off/30.
    q=first['action'][:,k-1];pose=pose_torch(q)
    ctx=torch.zeros_like(first['context']);ctx[:,:14]=q;ctx[:,84:102]=pose
    suffix=torch.cat([first['action'][:,k:],first['action'][:,-1:].expand(-1,k,-1)],1)
    b.update(context=ctx,action=suffix,robot=trajectory(ctx,suffix),robot_valid=torch.ones_like(first['robot_valid']),state_valid=torch.zeros_like(first['state_valid']),q_normalized=torch.zeros_like(first['q_normalized']),q_current=q,q_anchor=q)
    return b

def reentered(model,first,teacher,k,predicted,steps=2,seed=90):
    with torch.no_grad():
        z=model.mean_rollout(first,steps,seed).detach().reshape(len(first['obs']),3,8,-1)
    b=open_second(first,k) if predicted else dict(teacher)
    raw=z if predicted else teacher['obs'][:,4].detach()
    b['reentry_tokens']=model.reentry(raw)
    if predicted:b['world_current']=z.reshape(len(z),24,-1)
    return b,z

def corrected_chain(model,first,teacher,direct,k,use_predicted=True,steps=2,seed=90):
    b,z=reentered(model,first,teacher,k,use_predicted,steps,seed)
    pred=model.mean_rollout(b,steps,seed+1)
    if k==50:pred=pred.reshape(len(pred),3,8,-1).mean(2)
    valid=direct['world_valid'] & (teacher['world_valid'] if k==25 else teacher['world_valid'].reshape(len(pred),3,-1).all(-1))
    supervised=(((pred-direct['world_raw'])/direct['world_scale']).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
    # Direct endpoint is no longer computed: no consistency loss in either mode.
    return supervised,dict(chain_supervised=float(supervised.detach()),chain_gap=0.,predicted_reentry=use_predicted,chain_k=k)
