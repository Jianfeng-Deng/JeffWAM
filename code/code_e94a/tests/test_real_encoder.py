import torch,numpy as np,pytest
from e86.vision import FrozenEncoder,preprocess

def test_real_frozen_encoder_zero_gradient_cache(tmp_path):
    torch.set_num_threads(2);e=FrozenEncoder('cpu');e.train(True);assert not e.model.training;assert all(not p.requires_grad for p in e.parameters())
    rgb=np.random.default_rng(86).integers(0,256,(480,640,3),dtype=np.uint8);x,v,_=preprocess(rgb);z=e.images(x[None],v[None]);np.save(tmp_path/'cache.npy',z.half().numpy());cached=torch.from_numpy(np.load(tmp_path/'cache.npy').astype('f4'));assert torch.equal(z,cached)
    from e86.model import WAM
    from e86.fixtures import batch
    m=WAM().eval();b=batch(1);online=dict(b,obs=b['obs'].clone());offline=dict(b,obs=b['obs'].clone());online['obs'][:,4]=z.reshape(1,1,8,768).expand(1,3,8,768);offline['obs'][:,4]=cached.reshape(1,1,8,768).expand(1,3,8,768)
    with torch.no_grad():
        a,_=m.infer(online,steps=1);a2,_=m.infer(offline,steps=1)
    assert torch.equal(a,a2)
    projection=torch.nn.Linear(768,1);projection(z).sum().backward();assert all(p.grad is None for p in e.parameters());assert not torch.cuda.is_initialized()

def test_real_text_chunks_above64():
    e=FrozenEncoder('cpu');text=' '.join(['Hold the locking head with the left gripper.']*14);z=e.text(text);assert z.shape[0]>1 and z.shape[1]==768;assert not torch.cuda.is_initialized()
