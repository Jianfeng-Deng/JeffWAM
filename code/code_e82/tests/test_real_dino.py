import os
import pytest
import torch
from e82.config import Config
from e82.vision import VisionPair,LoRAQKV

@pytest.mark.skipif(os.environ.get('E82_TEST_DINO')!='1',reason='explicit local-weight CPU integration opt-in')
def test_local_original_DINO_last2_QV_gradients():
    torch.set_num_threads(2); torch.manual_seed(8101)
    v=VisionPair(Config().dino_weights); v.set_stage('P3_joint')
    assert len([m for m in v.modules() if isinstance(m,LoRAQKV)])==2
    x=torch.rand(1,3,28,28)
    z=v.online(x); target=v.target(x)
    assert z.shape==(1,768,2,2) and not target.requires_grad
    torch.testing.assert_close(z,target,rtol=0,atol=0) # zero-increment S0
    z.square().mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in v.online.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in v.target.parameters())
    assert sum(p.numel() for p in v.online.parameters() if p.requires_grad)==49152

@pytest.mark.skipif(os.environ.get('E82_TEST_DINO')!='1',reason='explicit local-weight CPU integration opt-in')
def test_real_DINO_LA_LW_core_LoRA_Estar():
    from e82.model import PiperWAM
    from e82.data import synthetic_batch
    from e82.objective import action_loss,world_loss
    m=PiperWAM(); m.train(); b=synthetic_batch(); b['obs_images']=[x[:,:,:,:28,:28] for x in b['obs_images']]; b['obs_pixel_valid']=[x[:,:,:,:28,:28] for x in b['obs_pixel_valid']]
    b['future_images']=[x[:,:,:,:28,:28] for x in b['future_images']]; b['future_pixel_valid']=[x[:,:,:,:28,:28] for x in b['future_pixel_valid']]
    for which in ('LA','LW'):
        m.zero_grad(set_to_none=True); o,ov=m.observation(b); w,v=m.world_target(b)
        a=torch.zeros(1,50,14); t=torch.ones(1)*.5
        av,wv,_=m.predict(o,ov,a,w,t,t)
        loss=action_loss(av,a).mean() if which=='LA' else world_loss(wv,w,v).mean(); loss.backward()
        assert sum(float(p.grad.abs().sum()) for p in m.core.parameters() if p.grad is not None)>0
        assert sum(float(p.grad.abs().sum()) for p in m.vision.online.parameters() if p.grad is not None)>0
        assert all(p.grad is None for p in m.vision.target.parameters())
