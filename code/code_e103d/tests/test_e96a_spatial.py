import copy
import pytest
import torch
from e86.model import Config
from e94a.model import WorldModel as M1
from e96a.model import WorldModel
from test_e88 import fixture

def small():return Config(width=32,heads=4,ffn=64,layers=2)

def paired():
    torch.manual_seed(90);base=M1(small()).eval();rng=torch.get_rng_state().clone()
    torch.manual_seed(90);new=WorldModel(small()).eval()
    assert torch.equal(torch.get_rng_state(),rng)
    return base,new

def test_zero_bypass_exact_M1_forward_and_initialization():
    base,new=paired();s=new.state_dict()
    assert all(torch.equal(v,s[k]) for k,v in base.state_dict().items())
    assert torch.count_nonzero(new.current.bypass.weight)==0
    b=fixture()
    # Nonzero readout avoids a vacuous all-zero mean comparison.
    with torch.no_grad():
        base.mean_head[-1].weight.normal_(std=.02)
        new.mean_head.load_state_dict(base.mean_head.state_dict())
        assert torch.equal(base.current(b['obs'][:,4]),new.current(b['obs'][:,4]))
        assert torch.equal(base.forward_mean(b)['mean'],new.forward_mean(b)['mean'])
        for x,y in zip(base.observation(b),new.observation(b)):assert torch.equal(x,y)
        a,c=base.adapter_loss(b['obs'][:,4]);d,e=new.adapter_loss(b['obs'][:,4])
        assert torch.equal(a,d) and torch.equal(c,e)

@pytest.mark.parametrize('camera',range(3))
@pytest.mark.parametrize('cell',range(8))
def test_row_major_local_bypass_shape_and_exact_cell(camera,cell):
    base,m=paired();raw=torch.zeros(2,3,8,768)
    raw[0,camera,cell,0]=2
    with torch.no_grad():
        m.current.bypass.weight.zero_();m.current.bypass.weight[0,0]=3
        delta=m.current(raw)-base.current(raw)
    expected=torch.zeros(2,3,8,32);expected[0,camera,cell,0]=6
    assert delta.shape==(2,3,8,32)
    torch.testing.assert_close(delta,expected,rtol=0,atol=1e-6)

def test_permuting_raw_cells_changes_current_with_exact_bypass_delta():
    base,m=paired();raw=torch.randn(2,3,8,768);perm=torch.arange(7,-1,-1)
    with torch.no_grad():
        m.current.bypass.weight.normal_(std=.02)
        a=m.current(raw);b=m.current(raw[:,:,perm]);pa=base.current(raw);pb=base.current(raw[:,:,perm])
        assert torch.allclose(pa,pb,atol=2e-6)
        expected=m.current.bypass(raw[:,:,perm]-raw)
        torch.testing.assert_close(b-a,expected,rtol=2e-5,atol=3e-6)
        assert (b-a).abs().max()>.1

def test_current_only_and_adapter_target_stopgrad():
    base,m=paired();b=fixture()
    with torch.no_grad():m.current.bypass.weight.normal_(std=.02)
    a,_,_=base.observation(b);o,_,_=m.observation(b)
    assert torch.equal(a[:,24:],o[:,24:]) and not torch.equal(a[:,:24],o[:,:24])
    target=m.current(b['obs'][:,4]).detach();pred=m.reentry(b['obs'][:,4])
    val,cos=m.adapter_loss(b['obs'][:,4]);expected=(pred-target).square().mean()/target.square().mean().clamp_min(1e-6)
    torch.testing.assert_close(val,expected)
    val.backward();assert m.current.bypass.weight.grad is None and m.reentry.proj.weight.grad is not None

def test_bypass_receives_mean_supervision():
    _,m=paired();b=fixture()
    with torch.no_grad():m.mean_head[-1].weight.normal_(std=.02)
    m.forward_mean(b)['mean'].square().mean().backward()
    assert m.current.bypass.weight.grad.norm()>0
