import os,sys,hashlib,torch,numpy as np
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
sys.path.insert(0,str(T/"code_e102a"));sys.path.insert(0,str(T/"code_e110f"))
os.environ["E102_GROUP"]="A2"
from e99w2_data import load
from e99w2_model import WorldModel
from e90.sampling import MultiSourceSampler
from e90.data import sample_horizons
from e86.common import seed_for
from e110f_core import set_trainables,tensor_sha,deterministic_loss,infer_d,infer_f,TRAINABLE_PREFIXES

CKPT=T/"results/e106c_f2_s90_20261115_02/ckpt_20000.pt"

def run():
    torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
    _,_,tr,dev=load();cp=torch.load(CKPT,map_location="cpu",weights_only=False)
    model=WorldModel(tr.norm);model.load_state_dict(cp["model"],strict=True);model.eval()
    # Full 4000-update parent/horizon sampling hashes, independently instantiated.
    stream_hash=[]
    for _ in range(2):
        sm=MultiSourceSampler(tr.rows,seed=90,weights={"piper":1.},caps={},demo3_cap=.45)
        h=hashlib.sha256()
        for step in range(1,4001):
            torch.manual_seed(seed_for(90,"E110F",step));ids=sm.sample(32);hs=sample_horizons(22000+step,90)
            h.update(np.asarray(ids,dtype="<i8").tobytes());h.update(np.asarray(hs,dtype="<i8").tobytes())
        stream_hash.append(h.hexdigest())
    assert stream_hash[0]==stream_hash[1]
    assert "piper_4dc107335389f5711a0f65e0" not in {r["canonical_id"] for r in tr.rows}
    b=tr.batch([0],(10,),"instruction","P3")
    # A2 D call must reproduce its locked t=0 inference exactly.
    model.eval();d0=infer_d(model,b,4,90);d1=model.infer(b,"action_only",4,90)[0]
    assert torch.equal(d0,d1) and torch.isfinite(d0).all()
    # Flow integration remains finite at all pre-registered step counts.
    for s in (1,2,4,8):
        p=infer_f(model,b,s,90);assert p.shape==(1,50,14) and torch.isfinite(p).all()
    zero=infer_f(model,b,s,90,torch.zeros(1,50,12));assert torch.isfinite(zero).all()
    assert torch.equal(infer_f(model,b,s,90),infer_f(model,b,s,90))
    # One optimizer update touches only action-path parameters; world/trunk hashes are exact.
    names=set_trainables(model);frozen=tuple(n for n,_ in model.named_parameters() if n not in names);before=tensor_sha(model,frozen)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    model.train();opt.zero_grad(set_to_none=True);loss,_=deterministic_loss(model,b);assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if n in frozen)
    opt.step();assert tensor_sha(model,frozen)==before
    print({"strict_load":"passed","sample_stream_sha256":stream_hash[0],"A2_D_bitwise":"passed","Euler_steps_finite":[1,2,4,8],"freeze_one_step":"passed","trainable_count":len(names),"frozen_count":len(frozen)})
if __name__=="__main__":run()
