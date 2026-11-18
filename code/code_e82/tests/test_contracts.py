import copy
import numpy as np
import pytest
import torch
from e82.config import Config,TOKEN_TABLE
from e82.core import UnifiedCore
from e82.data import synthetic_batch,StatContract,causal_dq,ParentSampler
from e82.model import PiperWAM
from e82.objective import objective,action_loss,world_loss
from e82.randomness import Streams,seed_all
from e82.inference import infer
from e82.evaluation import parent_equal,paired_bootstrap,window_metrics
from e82.train import optimizer,save,load,schedule
from e82.runtime import LatencyRecorder

@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(2); seed_all(8101); return PiperWAM(fixture=True)

@pytest.fixture
def batch(): return synthetic_batch(1,42)

def test_exact_tokens(model,batch):
    model.set_stage('P3_joint'); o,ov=model.observation(batch)
    assert o.shape==(1,187,256)
    g,gv,na=model.generated(torch.zeros(1,50,14),torch.zeros(1,3,91,768),torch.zeros(1),torch.ones(1))
    assert g.shape==(1,323,256) and na==50 and o.shape[1]+g.shape[1]==510
    assert TOKEN_TABLE['P1']==460

def test_O_only_reads_O_and_future_invariance(model,batch):
    model.set_stage('P3_joint'); model.eval(); o,ov=model.observation(batch)
    a=torch.randn(1,50,14); w=torch.randn(1,3,91,768); t=torch.zeros(1)
    oh=model.predict(o,ov,a,w,t,t)[2]
    other=model.predict(o,ov,a*100,w*100,t+1,t+.9)[2]
    torch.testing.assert_close(oh,other,rtol=0,atol=0)
    b2=copy.deepcopy(batch); b2['future_images']=[x*100 for x in b2['future_images']]; b2['action']*=100
    o2,ov2=model.observation(b2)
    torch.testing.assert_close(o,o2,rtol=0,atol=0)
    # Explicit derivatives: generated future cannot influence any O output.
    w.requires_grad_(); oh=model.predict(o,ov,a,w,t,t)[2]
    grad=torch.autograd.grad(oh.sum()+0*w.sum(),w)[0]; assert grad.abs().max()==0

def test_W_physically_absent_no_hidden_condition(model,batch):
    model.eval(); o,ov=model.observation(batch); a=torch.randn(1,50,14); t=torch.zeros(1)
    x=model.predict(o,ov,a=a,ta=t,tw=torch.tensor([float('nan')]))[0]
    y=model.predict(o,ov,a=a,ta=t,tw=torch.ones(1))[0]
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    calls=[]
    hook=model.w_proj.register_forward_hook(lambda *a:calls.append(1))
    model.predict(o,ov,a=a,ta=t); hook.remove(); assert calls==[]

def test_bidirectional_A_W_influence(model,batch):
    model.eval(); o,ov=model.observation(batch); a=torch.randn(1,50,14); w=torch.randn(1,3,91,768); t=torch.zeros(1)
    x,y,_=model.predict(o,ov,a,w,t,t)
    x2,y2,_=model.predict(o,ov,a+3,w+7,t,t)
    assert not torch.allclose(x,x2) and not torch.allclose(y,y2)
    xx=model.predict(o,ov,a,w+7,t,t)[0]; yy=model.predict(o,ov,a+3,w,t,t)[1]
    assert not torch.allclose(x,xx) and not torch.allclose(y,yy)

@pytest.mark.parametrize('which',['LA','LW'])
def test_losses_reach_core_and_LoRA_target_zero(model,batch,which):
    model.set_stage('P3_joint'); model.train(); model.zero_grad(set_to_none=True)
    o,ov=model.observation(batch); a=torch.randn(1,50,14); w,valid=model.world_target(batch); t=torch.ones(1)*.4
    assert not w.requires_grad
    av,wv,_=model.predict(o,ov,a,w+.2,t,t)
    loss=action_loss(av,torch.zeros_like(av)).mean() if which=='LA' else world_loss(wv,torch.zeros_like(wv),valid).mean()
    loss.backward()
    core=sum(float(p.grad.abs().sum()) for p in model.core.parameters() if p.grad is not None)
    lora=sum(float(p.grad.abs().sum()) for p in model.vision.online.parameters() if p.requires_grad and p.grad is not None)
    assert core>0 and lora>0
    assert all(p.grad is None for p in model.vision.target.parameters())

def test_missing_modalities_and_states(model,batch):
    model.eval(); batch['obs_crop_valid'][:]=False; batch['state_valid'][:]=False
    batch['q'][:]=0; batch['history'][:]=0
    o,ov=model.observation(batch); assert ov[:,:186].sum()==0 and ov[:,-1].all()
    av,_,_=model.predict(o,ov,a=torch.zeros(1,50,14),ta=torch.zeros(1)); assert torch.isfinite(av).all()
    # Missing data cannot affect outputs even if masked pixels/numerics differ.
    batch['obs_images']=[x+3 for x in batch['obs_images']]; batch['q']+=100
    oo,vv=model.observation(batch); aa=model.predict(oo,vv,a=torch.zeros(1,50,14),ta=torch.zeros(1))[0]
    torch.testing.assert_close(av,aa,rtol=0,atol=0)

def test_cache_equals_uncached_all_paths(model,batch):
    model.set_stage('P3_joint'); model.eval(); o,ov=model.observation(batch); cache=model.core.cache(o,ov)
    for with_w in (False,True):
        a=torch.randn(1,50,14); w=torch.randn(1,3,91,768) if with_w else None; t=torch.ones(1)*.3
        x=model.predict(o,ov,a,w,t,t if with_w else None)
        y=model.predict(o,ov,a,w,t,t if with_w else None,cache=cache)
        for xx,yy in zip(x,y):
            if xx is not None: torch.testing.assert_close(xx,yy,rtol=1e-6,atol=1e-6)

@pytest.mark.parametrize('mode',['action_only','joint'])
def test_inference_ten_euler_four_noise_reproducible(model,batch,mode):
    model.set_stage('P3_joint'); model.eval(); stats=StatContract.fixture()
    x=infer(model,batch,stats,mode,True); y=infer(model,batch,stats,mode,False)
    torch.testing.assert_close(x,y,rtol=1e-6,atol=2e-6); assert x.shape==(1,50,14) and torch.isfinite(x).all()
    z=infer(model,batch,stats,mode,True); torch.testing.assert_close(x,z,rtol=0,atol=0)

def test_P1_no_A_state_or_visual_grad(model,batch):
    model.set_stage('P1'); model.train(); model.zero_grad(set_to_none=True)
    calls=[]; hook=model.a_proj.register_forward_hook(lambda *a:calls.append(1))
    loss,metrics=objective(model,batch,StatContract.fixture(),Streams(8101,'P1',1),1,'P1')
    loss.backward(); hook.remove()
    assert calls==[] and metrics['LA'] is None
    assert all(p.grad is None for p in model.vision.parameters())
    assert all(p.grad is None for p in model.a_proj.parameters())
    o,ov=model.observation(batch); assert not ov[:,182:186].any()
    model.set_stage('P3_joint')

def test_all_four_modes_extra_LA(model):
    model.set_stage('P3_joint'); model.train(); model.zero_grad(set_to_none=True)
    b=synthetic_batch(4,30); calls=[]
    hook=model.a_proj.register_forward_hook(lambda m,a,o:calls.append(a[0].detach().clone()))
    loss,d=objective(model,b,StatContract.fixture(),Streams(8101,'P3_joint',500),500,'P3_joint',torch.arange(4))
    loss.backward(); hook.remove()
    assert d['modes']==[1,1,1,1] and d['lambda_world']==.1 and len(calls)==5
    assert d['LA']>0 and d['LW']>0

def test_dq_real_dt_and_signed_absolute_anchor():
    q=np.arange(14,dtype=np.float32); v,ok=causal_dq(q+.2,q,10.02,10.0)
    assert ok; np.testing.assert_allclose(v,10,rtol=1e-5)
    assert not causal_dq(q,q,0,0)[1]
    st=StatContract.fixture(); q=torch.randn(2,14); a=torch.randn(2,50,14); a[:,:,6]=-.003
    norm=st.norm_action(a,q); assert (norm[:,:,6]<0).all()
    torch.testing.assert_close(st.denorm_action(norm,q),a,rtol=1e-5,atol=1e-6)

def test_seeds_R_P_same_P3_W_does_not_advance_A():
    r=Streams(8101,'P3_joint',9); p=Streams(8101,'P3_joint',9)
    _=p.normal('epsilonW',(99,))
    torch.testing.assert_close(r.normal('epsilonA',(4,50,14)),p.normal('epsilonA',(4,50,14)),rtol=0,atol=0)
    rows=[dict(task_id=1,parent_component=str(i//3)) for i in range(12)]
    assert ParentSampler(rows).sample(1,32)==ParentSampler(rows).sample(1,32)

def test_parent_equal_bootstrap_and_units():
    r=parent_equal([1,1,1,3],['a','a','a','b'],['s1','s1','s1','s2']); assert r['mean']==2
    p=parent_equal([.9,.9,.9,2.7],['a','a','a','b'],['s1','s1','s1','s2'])
    d=paired_bootstrap(r,p,draws=200); np.testing.assert_allclose(d['relative_ci95'],[-.1,-.1],atol=1e-12)
    a=np.zeros((2,50,14)); a[:,:,6]=.001
    m=window_metrics(a,np.zeros_like(a),np.ones((50,14)))
    assert m['left_gripper/K10/MAE_mm'][0]==pytest.approx(1)

def test_resume_next_update_identical(tmp_path):
    torch.set_num_threads(2); seed_all(1); m=PiperWAM(fixture=True); m.set_stage('P3_warm'); opt=optimizer(m,2e-4); b=synthetic_batch()
    def update(model,opt,step):
        opt.zero_grad(); loss,_=objective(model,b,StatContract.fixture(),Streams(8101,'P3_warm',step),step,'P3_warm'); loss.backward(); opt.step(); return float(loss)
    update(m,opt,0); save(tmp_path/'ck.pt',m,opt,1,dict(seed=1))
    expected=update(m,opt,1)
    n=PiperWAM(fixture=True); n.set_stage('P3_warm'); oo=optimizer(n,2e-4); load(tmp_path/'ck.pt',n,oo,restore_rng=True)
    assert update(n,oo,1)==expected
    for a,c in zip(m.parameters(),n.parameters()): torch.testing.assert_close(a,c,rtol=0,atol=0)

def test_config_and_latency_gate():
    with pytest.raises(AssertionError): Config(robot='tianji')
    with pytest.raises(AssertionError): Config(cache_gb=101)
    r=LatencyRecorder()
    for _ in range(1000): r.add(.2)
    assert r.summary()['qualified']
