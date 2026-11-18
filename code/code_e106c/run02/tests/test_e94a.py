import copy,random
import numpy as np
import pytest
import torch
from e86.model import Config
from test_e88 import fixture
from e94a.model import WorldModel
from e94a.objective import loss,chain_loss
from e94a.checkpoint import save,restore
from e90.sampling import MultiSourceSampler

def model(recipe='M1'):
    return WorldModel(Config(width=32,heads=4,ffn=64,layers=2),recipe=recipe).eval()

@pytest.mark.parametrize('recipe',['M1','M2'])
def test_mean_shape_target_independence_and_no_state(recipe):
    m=model(recipe);b=fixture();x=m.forward_mean(b)['mean'];assert x.shape==b['world'].shape
    c=copy.deepcopy(b);c['world']*=100;c['q_normalized']*=100
    assert torch.equal(x,m.forward_mean(c)['mean'])
    v,d=loss(m,b,'P1',1);v.backward()
    assert m.state_proj.weight.grad is None and m.action_head.weight.grad is None
    assert all(p.grad is None for p in m.phase_head.parameters())

@pytest.mark.parametrize('head',['mean_head','world_head'])
def test_mlp_affine_span_exceeds256(head):
    m=WorldModel();h=getattr(m,head);assert h[0].in_features==256 and h[0].out_features==1024 and h[-1].out_features==768
    with torch.no_grad():
        h[-1].weight.normal_(std=.03)
        y=h(torch.randn(340,256));rank=int(torch.linalg.matrix_rank(y[1:]-y[:1],rtol=1e-5))
    assert rank>256

def test_m1_never_calls_FM():
    m=model();m.world_head.forward=lambda *a: (_ for _ in ()).throw(AssertionError('FM called'))
    b=fixture();v,d=loss(m,b,'P1',3500);v.backward();assert d['fm']==0
    m.mean_rollout(b)
    with pytest.raises(RuntimeError,match='never calls FM'):m(b,w=b['world'],tw=torch.zeros(2))

def test_M2_equal_double_loss_and_both_gradients():
    m=model('M2');b=fixture();v,d=loss(m,b,'P1',1,cosine_weight=0,adapter_weight=0)
    assert float(v.detach())==pytest.approx(d['mean']+d['fm'])
    v.backward();assert m.mean_head[-1].weight.grad.norm()>0 and m.world_head[-1].weight.grad.norm()>0

@pytest.mark.parametrize('recipe',['M1','M2'])
@pytest.mark.parametrize('k',[25,50])
def test_chain_mean_stopgradient(recipe,k):
    m=model(recipe)
    def b(h):
        x=fixture();n=24 if h<=50 else 3
        for key in ('world','world_raw','world_current','world_scale','world_valid'):x[key]=x[key][:,:n]
        x['world_horizons']=torch.full((2,n),h);x['world_positions']=torch.arange(24) if n==24 else torch.tensor([0,8,16]);return x
    calls=[];old=m.mean_rollout
    def wrapped(*a,**kw):calls.append(torch.is_grad_enabled());return old(*a,**kw)
    m.mean_rollout=wrapped;m.rollout=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('FM chain'))
    v,d=chain_loss(m,b(k),b(k),b(2*k),k);v.backward()
    assert calls==[False,True,False] and m.reentry.proj.weight.grad is not None
    assert m.mean_head[-1].weight.grad.norm()>0 and m.world_head[-1].weight.grad is None

@pytest.mark.parametrize('recipe',['M1','M2'])
def test_exact_checkpoint_optimizer_sampler_RNG_restore(tmp_path,recipe):
    m=model(recipe).train();s=MultiSourceSampler([dict(source='piper',canonical_id='a',parent_component='a',demo3=False,split='train',stratum='demo4')]);opt=torch.optim.AdamW(m.parameters(),lr=.001);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    cfg=dict(recipe=recipe);b=fixture()
    def step(i):
        opt.zero_grad();s.sample(2);v,_=loss(m,b,'P1',i);v.backward();opt.step();sched.step()
        return float(v.detach()),copy.deepcopy(m.state_dict()),(random.random(),np.random.rand(),torch.rand(1).item())
    step(1);cp=tmp_path/'cp.pt';save(cp,m,opt,sched,s,1,cfg,'frozen');a=step(2)
    other=model(recipe).train();op=torch.optim.AdamW(other.parameters(),lr=.001);sc=torch.optim.lr_scheduler.LambdaLR(op,lambda _:1.)
    ss=MultiSourceSampler(s.rows) if hasattr(s,'rows') else copy.deepcopy(s)
    restore(cp,other,op,sc,ss,cfg,'frozen');m,opt,sched,s=other,op,sc,ss;c=step(2)
    assert a[0]==c[0] and a[2]==c[2] and all(torch.equal(a[1][k],c[1][k]) for k in a[1])

@pytest.mark.parametrize('latent',[768,1024])
def test_configurable_target_dimension(latent):
    m=WorldModel(Config(width=32,heads=4,ffn=64,layers=1,latent=latent),recipe='M2');b=fixture()
    for key in ('obs','text','align_text','world','world_raw','world_current','world_scale'):
        b[key]=torch.randn(*b[key].shape[:-1],latent)
    b['world_scale']=torch.ones_like(b['world_scale']);v,_=loss(m,b,'P1',1);v.backward()
    assert m.forward_mean(b)['mean'].shape[-1]==latent

@pytest.mark.parametrize('branch',['mean','fm'])
def test_new_head_directional_masks(branch):
    m=model('M2');b=fixture();t=torch.ones(2)*.5
    with torch.no_grad():
        m.mean_head[-1].weight.normal_(std=.02);m.world_head[-1].weight.normal_(std=.02)
    c=copy.deepcopy(b);c['world_positions'][:72]=torch.flip(c['world_positions'][:72],[0])
    a=torch.randn(2,50,14);w=torch.randn_like(b['world']);changed=w.clone();changed[:,:72]+=100
    r=m(b,a,w,t,t,branch=branch);s=m(c,a*100,changed,t,t,branch=branch);key='mean' if branch=='mean' else 'world'
    assert torch.allclose(r[key][:,72:],s[key][:,72:],atol=4e-6)
    assert torch.allclose(r['world_cos'],m(b,a,changed,t,t,branch=branch)['world_cos'],atol=4e-6)
