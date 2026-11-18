import copy,random
import numpy as np
import pytest
import torch
from e86.model import Config
from e86.data import CappedParentSampler
from e88.audit import HorizonIndex
from test_e88 import fixture
from e90.model import WorldModel,adapter_pass
from e90.objective import loss,chain_loss,chain_weight
from e90.data import sample_horizons,load_data
from e90.sampling import MultiSourceSampler
from e90.checkpoint import save,restore,initialize

@pytest.fixture
def model():return WorldModel(Config(width=32,heads=4,ffn=64,layers=2)).eval()

@pytest.mark.parametrize('step',[0,1,100,2999,3000,3999,4999])
def test_formal_horizons_default_max100(step):
    hs=sample_horizons(step);assert {10,25,50,100}<=set(hs);assert max(hs)<=100
    assert {200,300}<=set(sample_horizons(step,maximum=300))

@pytest.mark.parametrize('nmse,cos,expected',[(.1,.95,True),(.10001,.99,False),(.01,.949,False),(.001,.999,True)])
def test_adapter_gate(nmse,cos,expected):assert adapter_pass(nmse,cos)==expected

def test_source_token_is_O_and_no_state_in_W(model):
    b=fixture();b['source_id']=torch.tensor([0,1]);o,valid,_=model.observation(b)
    assert o.shape[1]==60 and valid[:,-1].all()
    b['source_id']=torch.tensor([2,2]);o2,_,_=model.observation(b)
    assert not torch.equal(o[:,-1],o2[:,-1])
    b['q_normalized']*=1000
    v,_=loss(model,b,'P1',1,phase_weight=0,alignment_weight=0);v.backward()
    assert model.state_proj.weight.grad is None and model.action_head.weight.grad is None

@pytest.mark.parametrize('mode',['world_only','sync','independent','clean','action_only'])
def test_phase_zero_has_no_gradient(model,mode):
    b=fixture();v,d=loss(model,b,'P1' if mode=='world_only' else 'P3',1,mode=mode,phase_weight=0)
    v.backward();assert d['phase']==0
    assert all(p.grad is None for p in model.phase_head.parameters())
    assert all(p.grad is None for p in model.progress_head.parameters())

def test_recipe_B_probabilities_and_no_aux(model):
    from e86.objective import mode_for
    counts={m:0 for m in ('sync','independent','action_only','clean')}
    for i in range(10000):counts[mode_for(i,90,(.25,)*4)]+=1
    assert all(2200<n<2800 for n in counts.values())
    v,d=loss(model,fixture(),'P3',0,mode_probabilities=(0,0,0,1))
    assert d['mode']=='clean' and not d['clean_aux'] and torch.isfinite(v)

def test_exact_formal_chain_ramp():
    assert chain_weight('P1',3000,True,3000,1000)==0
    assert chain_weight('P1',3500,True,3000,1000)==.05
    assert chain_weight('P1',4000,True,3000,1000)==.1
    assert chain_weight('P1',5000,False,3000,1000)==0

@pytest.mark.parametrize('k',[25,50])
def test_e90_chain_stopgradient_reentry(model,k):
    def b(h):
        x=fixture();n=24 if h<=50 else 3
        for key in ('world','world_raw','world_current','world_scale','world_valid'):x[key]=x[key][:,:n]
        x['world_horizons']=torch.full((2,n),h);x['world_positions']=torch.arange(24) if n==24 else torch.tensor([0,8,16]);return x
    calls=[];old=model.rollout
    def wrapped(*a,**kw):calls.append(torch.is_grad_enabled());return old(*a,**kw)
    model.rollout=wrapped
    v,d=chain_loss(model,b(k),b(k),b(2*k),k,steps=1);v.backward()
    assert calls==[False,True,False] and model.reentry.proj.weight.grad is not None
    assert model.state_proj.weight.grad is None

def test_plus100_no_overrun_or_bad_cache():
    ix=object.__new__(HorizonIndex);n=151;t=np.arange(n)/30
    z=dict(timestamp=t,continuous_segment=np.zeros(n,int),p1_source_mask=np.ones(n,bool),camera_exposure_timestamp=np.repeat(t[:,None],3,1))
    sets=[set(range(n)) for _ in range(3)];ix.get=lambda cid:(z,sets)
    r=dict(canonical_id='x',anchor=50,_stage='P1');assert ix.resolve(r,100)[0]==[150]*3
    assert ix.resolve(dict(r,anchor=51),100)[1]=='end'
    sets[1].remove(150);assert ix.resolve(r,100)[1]=='cache'

def sample_rows():
    return [dict(source=s,canonical_id=s+str(i),parent_component=s+str(i),demo3=False,split='train',stratum='demo4') for s in ('piper','abc','yam') for i in range(3)]

def test_multisource_hard_caps_and_resume():
    rows=sample_rows();kwargs=dict(seed=90,weights={'piper':.1,'abc':.6,'yam':.3},caps={'abc':.3,'yam':.2})
    sampler=MultiSourceSampler(rows,**kwargs);counts={s:0 for s in ('piper','abc','yam')}
    for n in range(1,501):
        i=sampler.sample(1)[0];counts[rows[i]['source']]+=1
        assert counts['abc']<=int(n*.3+1e-9) and counts['yam']<=int(n*.2+1e-9)
    state=copy.deepcopy(sampler.state_dict());expected=sampler.sample(100)
    other=MultiSourceSampler(rows,**kwargs);other.load_state_dict(state);assert other.sample(100)==expected

def test_checkpoint_exact_stochastic_scheduler_resume_and_init(tmp_path,model):
    rows=sample_rows();sampler=MultiSourceSampler(rows);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9);cfg={'test':True}
    def update():
        opt.zero_grad(set_to_none=True);ids=sampler.sample(2)
        # Exercise all three saved RNG families and optimizer/scheduler state.
        b=fixture();v,_=loss(model,b,'P1',len(ids),alignment_weight=0)
        v=v*(random.random()+np.random.random()+1);v.backward();opt.step();sched.step()
        return float(v.detach()),copy.deepcopy(model.state_dict()),sched.get_last_lr(),ids
    update();path=tmp_path/'model.pt';save(path,model,opt,sched,sampler,1,cfg,'manifest')
    a=update();restore(path,model,opt,sched,sampler,cfg,'manifest');b=update()
    assert a[0]==b[0] and a[2:]==b[2:] and all(torch.equal(a[1][k],b[1][k]) for k in a[1])
    fresh=WorldModel(model.c);info=initialize(path,fresh);assert info['step']==1
    saved=torch.load(path,weights_only=False)
    assert all(torch.equal(v,saved['model'][k]) for k,v in fresh.state_dict().items())
    with pytest.raises(ValueError):restore(path,model,opt,sched,sampler,{'changed':True},'manifest')
    with pytest.raises(ValueError):restore(path,model,opt,sched,sampler,cfg,'changed')
    with pytest.raises(FileExistsError):save(path,model,opt,sched,sampler,2,cfg,'manifest')

def test_real_P1_images_only_and_frozen_dev():
    _,_,train,dev,coverage=load_data();b=train.batch([0,1])
    assert b['world'].shape==(2,75,768) and 'action' not in b and 'q_normalized' not in b
    assert coverage['existing_instruction_windows']+coverage['fallback_windows']==107057
    d=dev.batch([0]);assert not dev.base.text_cache.keys()-{'__human_default__'}
    assert set(r['parent_component'] for r in train.rows).isdisjoint(r['parent_component'] for r in dev.rows)
    shifted=dev.shifted(0,25,'short');assert shifted is not None
    assert torch.allclose(shifted['world_raw'],dev.get(0,(50,),'short')['world_raw'])
