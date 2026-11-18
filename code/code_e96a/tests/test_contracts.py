import copy,json,sys,random
from pathlib import Path
import numpy as np,pytest,torch
from e86.model import WAM,Config,mask_oa
from e86.fixtures import batch
from e86.objective import loss,mode_for
from e86.data import *
from e86.checkpoint import save,load
from e86.vision import preprocess,spatial_grid,split_sentences

torch.set_num_threads(2)
@pytest.fixture
def model():return WAM().eval()
@pytest.fixture
def b():return batch()
def test_mask_blocks_only_O():
    m=mask_oa(59,51,72);assert not m[:59,:59].any();assert m[:59,59:].all();assert not m[59:].any()
def test_future_cannot_change_O(model,b):
    a=torch.randn(2,50,14);w=torch.randn(2,72,768);t=torch.ones(2)*.5
    o=model(b,a,w,t,t);p=model(b,a,w+10,t,t);assert torch.equal(o['observation'],p['observation']);assert not torch.equal(o['action'],p['action'])
def test_O_zero_future_gradient(model,b):
    w=torch.randn(2,72,768,requires_grad=True);o=model(b,torch.zeros(2,50,14),w,torch.zeros(2),torch.zeros(2));o['observation'].sum().backward();assert w.grad is None or not w.grad.any()
def test_deleted_W_no_projection(model,b):
    def fail(*a):raise AssertionError('W projection called')
    model.world_proj.register_forward_hook(fail);o=model(b,torch.zeros(2,50,14),None,torch.zeros(2),torch.ones(2));assert o['world'] is None

def test_state_only_A(model,b):
    t=torch.ones(2)*.5;a=torch.zeros(2,50,14);w=torch.zeros(2,72,768);o=model(b,a,w,t,t);bb=dict(b,q_normalized=b['q_normalized']+10);p=model(bb,a,w,t,t);assert torch.equal(o['observation'],p['observation']);assert not torch.equal(o['action'],p['action'])
def test_P1_no_state_or_action(model,b):
    def fail(*a):raise AssertionError('P1 state/action projection')
    model.state_proj.register_forward_hook(fail);model.action_proj.register_forward_hook(fail);o=model(b,None,b['world'],None,torch.ones(2));assert o['action'] is None
@pytest.mark.parametrize('drop',[0.,1.])
def test_history_dropout(b,drop):
    m=WAM(Config(history_drop=drop)).train();o,_,hv=m.observation(b);assert bool(hv.all()) if drop==0 else not hv.any()
def test_history_eval_no_dropout(model,b):assert model.observation(b)[2].all()
def test_history_never_contains_actions(model,b):
    bb=dict(b,action=b['action']+100);assert torch.equal(model.observation(b)[0],model.observation(bb)[0])
def test_action_residual_and_signed_gripper(model,b):
    for p in model.parameters():p.data.zero_()
    b['q_anchor'][:,[6,13]]=0;b['action_mean'][:,:,6]=-.003
    a,_=model.infer(b,steps=1,seed=9)
    # Unit-noise output with zero vector field; deterministic seed and signed absolute offset.
    eps=torch.randn(2,50,14,generator=torch.Generator().manual_seed(9));expect=b['q_anchor'][:,None]+b['action_mean']+eps*b['action_std'];assert torch.equal(a,expect)
@pytest.mark.parametrize('mode',['sync','independent','action_only','clean'])
def test_all_P3_modes_backward(b,mode):
    m=WAM().train();v,d=loss(m,b,'P3',2,mode=mode);assert torch.isfinite(v);v.backward();assert m.blocks[0].ff[0].weight.grad.norm()>0
    assert (m.world_proj.weight.grad is None) if mode=='action_only' else m.world_proj.weight.grad.norm()>0

def test_P1_action_unchanged(b):
    m=WAM().train();v,_=loss(m,b,'P1',1);v.backward();assert m.action_proj.weight.grad is None;assert m.state_proj.weight.grad is None

def test_target_stop_gradient(b):
    b['world'].requires_grad_();m=WAM();v,_=loss(m,b,'P1',1);v.backward();assert b['world'].grad is None

def test_quality_off_no_effect(model,b):
    o=model.observation(b)[0];assert torch.equal(o,model.observation(dict(b,quality_flag=torch.ones(2,dtype=torch.long)))[0]);assert model.quality is None

def test_quality_on_adds_token(b):
    m=WAM(Config(quality_condition=True)).eval();a=m.observation(b)[0];z=m.observation(dict(b,quality_flag=torch.ones(2,dtype=torch.long)))[0];assert a.shape[1]==60;assert not torch.equal(a[:,-1],z[:,-1])

def test_quality_weights_optional(b):
    m=WAM().eval();v,_=loss(m,b,'P3',1,mode='action_only');v2,_=loss(m,dict(b,quality_weight=torch.zeros(2)),'P3',1,mode='action_only');assert torch.isfinite(v2) and v2<v

def test_demo3_cap():
    rs=[dict(parent_component=str(i),demo3=i<90) for i in range(100)];s=CappedParentSampler(rs);drawn=[]
    for _ in range(30):
        drawn+=s.sample(32);assert sum(rs[i]['demo3'] for i in drawn)<=int(len(drawn)*.45)
    assert .35<sum(rs[i]['demo3'] for i in drawn)/len(drawn)<=.45

def test_sampler_resume():
    rs=[dict(parent_component=str(i),demo3=i<5) for i in range(10)];a=CappedParentSampler(rs);a.sample(23);s=a.state_dict();x=a.sample(32);b=CappedParentSampler(rs);b.load_state_dict(s);assert b.sample(32)==x

def test_parent_isolation():
    with pytest.raises(ValueError):assert_parents([dict(parent_component='a',split='train'),dict(parent_component='a',split='dev')],{'a':'train'})

def test_ready_tamper(tmp_path):
    p=tmp_path/'source';p.write_text('old');r=tmp_path/'READY.json';r.write_text(json.dumps(dict(files=[dict(path=str(p),sha256=sha(p))])));verify_receipt(r);p.write_text('new')
    with pytest.raises(ValueError):verify_receipt(r)

def test_receipt_itself_tamper(tmp_path):
    p=tmp_path/'receipt';p.write_text('{}');h=sha(p);p.write_text('{"files": []}')
    with pytest.raises(ValueError):verify_receipt(p,h)

def test_no_existing_checkpoint_overwrite(model,tmp_path):
    opt=torch.optim.AdamW(model.parameters());s=CappedParentSampler([dict(parent_component='p',demo3=False)]);p=tmp_path/'checkpoint.pt';save(p,model,opt,0,s,{})
    with pytest.raises(FileExistsError):save(p,model,opt,0,s,{})

def test_exact_resume(b,tmp_path):
    torch.manual_seed(111);m=WAM().train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4);s=CappedParentSampler([dict(parent_component='p',demo3=False)])
    def update():
        opt.zero_grad();v,_=loss(m,b,'P3',5,mode='independent');v.backward();opt.step();s.sample(2);return float(v.detach())
    update();p=tmp_path/'c.pt';save(p,m,opt,1,s,{'data':'ok'});v=update();expected=copy.deepcopy(m.state_dict());load(p,m,opt,s,{'data':'ok'});v2=update();assert v==v2;assert all(torch.equal(expected[k],v) for k,v in m.state_dict().items())
    with pytest.raises(ValueError):load(p,m,opt,s,{'data':'tamper'})

def test_CPU_checkpoint_never_cuda(model,tmp_path,monkeypatch):
    def fail(*a,**k):raise AssertionError('CPU touched CUDA RNG')
    monkeypatch.setattr(torch.cuda,'get_rng_state',fail);monkeypatch.setattr(torch.cuda,'get_rng_state_all',fail)
    opt=torch.optim.AdamW(model.parameters());s=CappedParentSampler([dict(parent_component='p',demo3=False)]);save(tmp_path/'c',model,opt,0,s,{})

def test_fixed_grid_and_padding():
    rgb=np.ones((480,640,3),np.uint8)*127;x,v,af=preprocess(rgb);assert x.shape==(3,224,224);assert v.sum()==224*168;z=spatial_grid(torch.ones(1,196,768),v[None]);assert torch.allclose(z,torch.ones_like(z))

def test_cache_roundtrip_fixture(tmp_path):
    x=torch.randn(3,196,768);v=torch.ones(3,224,224);online=spatial_grid(x,v).half().float();p=tmp_path/'feature.npy';np.save(p,online.half().numpy());cached=torch.from_numpy(np.load(p).astype('f4'));assert torch.equal(online,cached)

def test_text_sentence_split_no_truncation():
    class Tok:
        def num_special_tokens_to_add(self,pair=False):return 2
        def encode(self,s,add_special_tokens=False):return s.split()
    text=' '.join(['A short sentence.']*35);chunks=split_sentences(text,Tok());assert len(chunks)>1;assert all(len(Tok().encode(s))<=62 for s in chunks);assert ' '.join(chunks)==text

def test_both_inference_modes(model,b):
    for mode in ('action_only','joint'):
        a,w=model.infer(b,mode,steps=2);assert a.shape==(2,50,14);assert torch.isfinite(a).all();assert (w is None)==(mode=='action_only')

def test_default_action_only(model,b):
    def fail(*a):raise AssertionError('default deployment reads W')
    model.world_proj.register_forward_hook(fail);model.infer(b,steps=1)

def test_nan_safe_missing_conditions(model,b):
    b['current_valid'][:]=False;b['history_valid'][:]=False;b['text_valid'][:]=False;b['state_valid'][:]=False
    a,_=model.infer(b,steps=1);assert torch.isfinite(a).all()

def test_core_spec(model):
    assert len(model.blocks)==6;assert model.blocks[0].attn.num_heads==8;assert model.blocks[0].ff[0].out_features==1024

def test_mode_frequencies():
    xs=[mode_for(i) for i in range(10000)]
    for k,p in [('sync',.3),('independent',.3),('action_only',.3),('clean',.1)]:assert abs(xs.count(k)/len(xs)-p)<.02


def test_deterministic_world_cos_does_not_read_noisy_future(model,b):
    a=torch.randn(2,50,14);w=torch.randn(2,72,768);t=torch.ones(2)*.5
    x=model(b,a,w,t,t);y=model(b,a,w+20,t,t)
    assert torch.equal(x['world_cos'],y['world_cos'])
    assert x['world_cos'].shape==(2,72,768)


def test_history_camera_identity_preserved(model,b):
    bb=dict(b,obs=b['obs'].clone());bb['obs'][:,:4]=b['obs'][:,:4,:,].flip(2)
    x=model.observation(b)[0][:,24:56];y=model.observation(bb)[0][:,24:56]
    assert (x-y).abs().max()>1e-5


def test_P1_absent_action_alignment_gets_no_gradient(b):
    m=WAM().train();v,_=loss(m,b,'P1',1);v.backward();assert m.align_heads[2].weight.grad is None and m.align_heads[2].bias.grad is None

def test_dropped_history_alignment_gets_no_gradient(b):
    m=WAM(Config(history_drop=1.)).train();v,_=loss(m,b,'P3',1,mode='action_only');v.backward();assert m.align_heads[1].weight.grad is None and m.align_heads[1].bias.grad is None


def test_instruction_chunk_order_is_not_permutation_invariant(model,b):
    a=torch.zeros(2,50,14);t=torch.ones(2)*.5
    x=model(b,a,None,t,t)['action'];y=model(dict(b,text=b['text'].flip(1)),a,None,t,t)['action']
    assert (x-y).abs().max()>1e-6


def test_alignment_projection_cannot_relabel_all_pairs_positive(model,b):
    from e86.objective import contrastive
    model.align_text.weight.data.zero_();model.align_text.bias.data.fill_(.2)
    t=torch.ones(2)*.5;out=model(b,torch.zeros(2,50,14),None,t,t);v=contrastive(model,out,b)
    assert torch.isfinite(v) and v>.5


def test_parent_equal_when_demo3_cap_does_not_bind():
    from collections import Counter
    rs=[]
    for p in range(10):
        for _ in range(25 if p==0 else 1):rs.append(dict(parent_component=str(p),demo3=p<2))
    s=CappedParentSampler(rs,seed=129);drawn=s.sample(10000);counts=Counter(rs[i]['parent_component'] for i in drawn)
    assert all(abs(counts[str(p)]/10000-.1)<.018 for p in range(10));assert s.demo3_count<=.45*s.count


def test_clean_aux_random_allocation_device_is_explicit(model,b,monkeypatch):
    original=torch.rand;calls=[]
    def checked(*args,**kwargs):
        if kwargs.get('generator') is not None:
            assert 'device' in kwargs;assert torch.device(kwargs['device']).type==kwargs['generator'].device.type;calls.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'rand',checked);v,d=loss(model,b,'P3',2,mode='sync',clean_aux_probability=1.)
    assert calls and d['clean_aux'] and torch.isfinite(v)
