import sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e86.common import *
from e86.resource import Guard

def main():
    dest=C/'artifacts/gpu_regression_v1';dest.mkdir(exist_ok=False);guard=Guard('GPU_device_regression',120)
    try:
        import torch
        from e86.model import WAM
        from e86.objective import loss
        from e86.data import Package,CachedDataset,collate,move
        from e86.language import load as load_language
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(2);p=Package();contract=json.loads((C/'artifacts/smoke_v1/contract.json').read_text());ws=contract['P3_subset'][:4];lc=load_language(ws,C/'artifacts/language_v1');ds=CachedDataset(ws,p.captures,C/'artifacts/features_v3',json.loads((C/'artifacts/stats_v1.json').read_text()),lc);b=move(collate([ds[i] for i in range(len(ds))]),'cuda:0');m=WAM().train().to('cuda:0');out=[]
        for mode in ('sync','independent','action_only','clean'):
            guard.check();m.zero_grad(set_to_none=True);v,d=loss(m,b,'P3',2,mode=mode,clean_aux_probability=1.);v.backward();assert torch.isfinite(v);out.append(dict(mode=mode,loss=float(v.detach()),clean_aux=d['clean_aux']))
        atomic(dest/'status.json',dict(status='passed',modes=out,elapsed_s=guard.check(),formal_training_started=False));print(json.dumps(out),flush=True)
    finally:guard.finish()
if __name__=='__main__':main()
