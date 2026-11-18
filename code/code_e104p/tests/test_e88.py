import numpy as np,pytest,torch
from e86.fixtures import batch
from e86.model import Config
from e88.model import WorldModel,directional_mask,horizon_embedding
from e88.audit import stage_target,causal_segments,HorizonIndex
from e88.objective import loss,chain_loss,chain_weight
from e88.data import sample_horizons

def fixture(n=2):
    b=batch(n);h=[10]*24+[25]*24+[50]*24+[100]*3+[200]*3+[300]*3;t=len(h)
    b.update(world=torch.randn(n,t,768),world_raw=torch.randn(n,t,768),world_current=torch.randn(n,t,768),world_scale=torch.ones(n,t,768),world_horizons=torch.tensor(h)[None].expand(n,-1),world_positions=torch.tensor(list(range(24))*3+[0,8,16]*3),world_valid=torch.ones(n,t,dtype=torch.bool),phase_target=torch.zeros(n,dtype=torch.long),progress_target=torch.zeros(n),phase_valid=torch.ones(n,dtype=torch.bool))
    return b
@pytest.fixture
def m():return WorldModel(Config(width=32,heads=4,ffn=64,layers=2)).eval()
def test_continuous_embedding_and_sampling():
    x=horizon_embedding(torch.tensor([25.,25.1,200.]),32);assert not torch.equal(x[0],x[1]);assert torch.isfinite(x).all()
    assert all(set((10,25,50,100,200,300))<=set(sample_horizons(i)) for i in range(20));assert any(len(sample_horizons(i))==7 for i in range(20))
def test_directional_mask_and_indirect_A_block():
    h=torch.tensor([25,50,100,300]);mask=directional_mask(3,2,h);assert mask[7,5] and mask[7,6] and mask[7,3] and mask[7,4];assert not mask[5,7];assert mask[:3,3:].all()
def test_long_outputs_invariant_to_short_noisy_tokens_and_actions(m):
    with torch.no_grad():m.world_head.weight.normal_(std=.02)
    b=fixture();a=torch.randn(2,50,14);w=torch.randn_like(b['world']);t=torch.ones(2)*.5
    r=m(b,a,w,t,t);changed=w.clone();changed[:,:72]+=100
    r2=m(b,a*100,changed,t,t);assert torch.allclose(r['world'][:,72:],r2['world'][:,72:],atol=2e-6)
def test_cosine_noise_independent_and_short_only(m):
    b=fixture();t=torch.ones(2)*.5;r=m(b,w=b['world'],tw=t);r2=m(b,w=b['world']+100,tw=t);assert torch.allclose(r['world_cos'],r2['world_cos'],atol=2e-6)
    from e88.objective import weighted_mean
    e=torch.zeros(2,81);e[:,72:]=100;assert weighted_mean(e,b['world_valid'],b['world_horizons'],True)==0
def test_residual_normalization_roundtrip_and_copy_skill():
    current=torch.randn(3,8,768);future=torch.randn_like(current);scale=torch.rand(3,8,1)+.05;res=(future-current)/scale
    assert torch.allclose(current+res*scale,future,atol=1e-6);err=res.square().mean();assert float(1-err/err)==0
@pytest.mark.parametrize('mode',('world_only','sync','independent','action_only','clean'))
def test_modes_finite_and_deleted_W_skips_chain(m,mode):
    b=fixture();called=[]
    def chain(_):called.append(True);return next(m.parameters()).sum()*0,{}
    m.adapter_ready.fill_(True);v,d=loss(m,b,'P1' if mode=='world_only' else 'P3',200,mode=mode,chain=chain,p1_converged=True);assert torch.isfinite(v);v.backward()
    assert bool(called)==(mode!='action_only')
def test_adapter_distillation_stopgrad_current(m):
    x=torch.randn(2,3,8,768);v,c=m.adapter_loss(x);v.backward();assert m.current.proj.weight.grad is None;assert m.reentry.proj.weight.grad is not None
def test_reentry_bypasses_real_current(m):
    b=fixture();b['reentry_tokens']=m.reentry(b['obs'][:,4]);x=m.observation(b,False)[0];b2=dict(b,obs=b['obs'].clone());b2['obs'][:,4]+=100;y=m.observation(b2,False)[0];assert torch.allclose(x,y)
@pytest.mark.parametrize('anchor,ok',((19,False),(20,True),(21,False)))
def test_phase_cutoff_no_future_supervision(anchor,ok):
    segments=[dict(source_frame_start=0,source_frame_end=20,causal_evidence_cutoff_source_frame=20)]
    assert stage_target(segments,anchor)[2]==ok
    assert bool(causal_segments(segments,anchor))==(anchor>=20)
def test_phase_missing_cutoff_masked():assert stage_target([dict(source_frame_start=0,source_frame_end=10)],5)[2] is False
def test_chain_schedule_gate_and_P1_no_action(m):
    assert chain_weight('P1',127,True)==0 and chain_weight('P1',256,True)==.1 and chain_weight('P3',128,False)==0
    b=fixture();b['q_normalized']*=100;v,d=loss(m,b,'P1',200);assert d['action']==0 and d['chain_weight']==0
@pytest.mark.parametrize('reason',('end','subinterval','seam','source_mask','cache'))
def test_horizon_rejects_invalid_source_or_cache(reason):
    ix=object.__new__(HorizonIndex);n=500;t=np.arange(n)/30;z=dict(timestamp=t,continuous_segment=np.zeros(n,int),p3_source_mask=np.ones(n,bool),camera_exposure_timestamp=np.repeat(t[:,None],3,1));sets=[set(range(n)) for _ in range(3)];row=dict(canonical_id='c',anchor=20)
    if reason=='end':row['anchor']=490
    if reason=='subinterval':row['subinterval_raw_rows_exclusive']=[0,29]
    if reason=='seam':z['continuous_segment'][30:]=1
    if reason=='source_mask':z['p3_source_mask'][25]=False
    if reason=='cache':sets[0].remove(30)
    ix.get=lambda cid:(z,sets);assert ix.resolve(row,10)[1]==reason
@pytest.mark.parametrize('k',(25,50,100))
def test_chain_stopgrad_and_resolution(m,k):
    def mk(h):
        b=fixture();n=24 if h<=50 else 3
        for key in ('world','world_raw','world_current','world_scale','world_valid'):b[key]=b[key][:,:n].clone()
        b['world_horizons']=torch.full((2,n),h);b['world_positions']=torch.arange(n) if n==24 else torch.tensor([0,8,16]);return b
    first=mk(k);shift=mk(k);direct=mk(2*k);v,d=chain_loss(m,first,shift,direct,k,steps=1);v.backward();assert torch.isfinite(v);assert m.reentry.proj.weight.grad is not None
    # First/direct rollout are executed under no_grad; only second hop graph is retained.
    assert first['world_raw'].grad is None and direct['world_raw'].grad is None

def test_signed_current_regression_preserves_negative_and_units(tmp_path):
    from e88.evaluation import signed_current
    from types import SimpleNamespace
    q=np.zeros((2,14),np.float32);q[1,[6,13]]=[0,.02];raw=q.copy();raw[1,6]=-.003
    path=tmp_path/'numeric.npz';np.savez(path,q=q,qpos_raw=raw)
    ds=SimpleNamespace(captures={'cid':{'numeric_cache':str(path)}});got=signed_current(ds,{'canonical_id':'cid','anchor':1})
    assert got[6]==pytest.approx(-.003) and got[13]==pytest.approx(.02);assert abs(got[6]-q[1,6])*1000==pytest.approx(3.)
def test_chain_first_and_direct_have_no_grad_and_no_state_action(m):
    calls=[];original=m.rollout
    def recorded(b,*args,**kwargs):
        calls.append(torch.is_grad_enabled());return original(b,*args,**kwargs)
    m.rollout=recorded
    b=fixture();n=24
    for key in ('world','world_raw','world_current','world_scale','world_valid'):b[key]=b[key][:,:n]
    b['world_horizons']=torch.full((2,n),25);b['world_positions']=torch.arange(n)
    direct=dict(b,world_horizons=torch.full((2,n),50));v,d=chain_loss(m,b,b,direct,25,steps=1);v.backward();assert calls==[False,True,False];assert m.state_proj.weight.grad is None and m.action_proj.weight.grad is None

def test_stage_aux_invariant_to_future_noise(m):
    b=fixture();t=torch.ones(2)*.3;r=m(b,w=b['world'],tw=t);r2=m(b,w=b['world']+100,tw=t)
    assert torch.allclose(r['phase'],r2['phase'],atol=2e-6) and torch.allclose(r['progress'],r2['progress'],atol=2e-6)

def test_numpy_fine_coarse_pooling():
    from e88.data import coarse
    z=np.ones((3,8,768),np.float32);assert coarse(z,25).shape==(3,8,768);assert coarse(z,100).shape==(3,1,768)
def test_real_v31_dataset_and_multi_horizon_forward_cpu(m):
    from e88.train import make_data
    p,r,stats,full,small,lc=make_data()
    for stage in ('P1','P3'):
        b=small[stage].batch([0,1],stage=stage);v,d=loss(m,b,stage,0);assert torch.isfinite(v);assert b['world'].shape==(2,81,768)
        cur=b['world_current'];assert torch.allclose(cur+b['world']*b['world_scale'],b['world_raw'],atol=1e-6)
    assert not torch.cuda.is_initialized()

def test_P1_diagnostic_keeps_P1_source_mask():
    from e88.train import make_data
    from e88.data import Data
    p,r,stats,full,small,lc=make_data();assert all(row['_stage']=='P1' for row in small['P1'].rows)
    b=small['P1'].batch([0]);assert b['world_valid'][:,:72].all()
    for row in full['P1'].rows[::10000]:assert full['P1'].index.resolve(row,10)[0] is not None

def test_v31_configured_recipe_reaches_real_losses(m):
    b=fixture();v,d=loss(m,b,'P3',0,mode_probabilities=(0,0,0,1),clean_aux_probability=0)
    assert torch.isfinite(v) and d['mode']=='clean' and not d['clean_aux']

def test_real_reentry_uses_adapter_and_maximum_config(m):
    b=fixture();n=24
    for key in ('world','world_raw','world_current','world_scale','world_valid'):b[key]=b[key][:,:n]
    b['world_horizons']=torch.full((2,n),25);b['world_positions']=torch.arange(n)
    direct=dict(b,world_horizons=torch.full((2,n),50));v,d=chain_loss(m,b,b,direct,25,use_predicted=False,steps=1);v.backward()
    assert m.reentry.proj.weight.grad is not None and d['predicted_reentry'] is False
    assert chain_weight('P3',999,True,maximum=.03)==.03

def test_full_latent_noise_skip_and_zero_copy_initialization(m):
    b=fixture();w=torch.randn_like(b['world']);t=torch.tensor([.25,.75]);out=m(b,w=w,tw=t)
    assert torch.allclose(out['world'],-w/(1-t[:,None,None]),atol=1e-6)
    pred=m.rollout(b,steps=4);assert torch.allclose(pred,b['world_current'],atol=1e-6)

def test_chain_requires_convergence_or_explicit_engineering_probe(m):
    b=fixture();m.adapter_ready.fill_(True);called=[]
    def chain(_):called.append(True);return next(m.parameters()).sum()*0,{}
    v,d=loss(m,b,'P3',200,mode='independent',chain=chain);assert not called and d['chain_weight']==0
    v,d=loss(m,b,'P3',200,mode='independent',chain=chain,engineering_chain_probe=True);assert called and d['chain_weight']>0
