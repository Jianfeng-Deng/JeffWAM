"""Read train-only real windows and original local DINO on CPU; no optimizer/checkpoint."""
import json,time
from pathlib import Path
import torch
from e82.config import Config
from e82.data import DataPackageDataset
from e82.model import PiperWAM
from e82.runtime import atomic_json,brt

def main():
    torch.set_num_threads(2);torch.manual_seed(8101);torch.set_grad_enabled(False)
    root=Path(__file__).resolve().parents[1];out=root/'artifacts'/'real_cpu_forward_v1.json'
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();ds=DataPackageDataset(Config().data_root,'P3_joint',profile=True,limit=1);b=ds[0];m=PiperWAM().eval()
    o,ov=m.observation(b);w,v=m.world_target(b);t=torch.ones(1)*.5;a=torch.zeros(1,50,14)
    av,wv,oh=m.predict(o,ov,a,w,t,t,w_valid=b['world_structure_valid']);cache=m.core.cache(o,ov);ac,wc,oc=m.predict(o,ov,a,w,t,t,w_valid=b['world_structure_valid'],cache=cache)
    error=max(float((av-ac).abs().max()),float((wv-wc).abs().max()))
    assert error<=1e-6 and torch.isfinite(av).all() and torch.isfinite(wv).all()
    m.set_stage('P1');o1,v1=m.observation(b);_,w1,_=m.predict(o1,v1,w=w,tw=t,w_valid=b['world_structure_valid']);assert torch.isfinite(w1).all()
    result=dict(status='passed',brt=brt(),source_window=b['window_id'][0],parent=b['parent_component'][0],refs=ds.refs,formal_training=False,device='cpu',cuda_initialized=torch.cuda.is_initialized(),wall_seconds=time.monotonic()-start,O_shape=list(o.shape),O_valid_tokens=int(ov.sum()),A_shape=list(av.shape),W_shape=list(wv.shape),W_valid_target_tokens=int(v.sum()),W_structural_valid_tokens=int(b['world_structure_valid'].sum()),cache_max_abs=error,P1_A_absent=True,original_local_dino=True,ROI_pending_masked=True,provisional_unnormalized_features_for_numerical_contract_only=True)
    assert not result['cuda_initialized'];atomic_json(out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
