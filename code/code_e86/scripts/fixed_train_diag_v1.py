import sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e86.common import *
from e86.resource import Guard

def main():
    dest=C/'artifacts/fixed_train_diag_v1.json';assert not dest.exists();guard=Guard('fixed_training_loss_diagnostic',180)
    try:
        import torch
        from e86.model import WAM
        from e86.data import Package,CachedDataset,collate,move
        from e86.language import load as load_language
        from e86.objective import loss
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(2)
        run=C/'artifacts/smoke_v2';contract=json.loads((run/'contract.json').read_text());p=Package();assert p.refs==contract['package'];stats=json.loads((C/'artifacts/stats_v1.json').read_text());result={}
        for stage,before,after,nolang in [('P1',None,'P1_end.pt',False),('P3','P1_end.pt','P3_end.pt',False),('P3_no_language','P1_end.pt','P3_no_language_end.pt',True)]:
            ws=contract['P1_subset' if stage=='P1' else 'P3_subset'][:8];lc=load_language(ws,C/'artifacts/language_v1');ds=CachedDataset(ws,p.captures,C/'artifacts/features_v3',stats,lc);b=move(collate([ds[i] for i in range(len(ds))]),'cuda:0')
            if nolang:b=dict(b,text_valid=torch.zeros_like(b['text_valid']),align_valid=torch.zeros_like(b['align_valid']))
            values=[]
            for checkpoint in (before,after):
                guard.check();torch.manual_seed(86);m=WAM().eval().to('cuda:0')
                if checkpoint:m.load_state_dict(torch.load(run/checkpoint,map_location='cpu',weights_only=False)['model'])
                with torch.no_grad():v,d=loss(m,b,'P1' if stage=='P1' else 'P3',9999,seed=86,mode='world_only' if stage=='P1' else 'independent',clean_aux_probability=0.)
                assert torch.isfinite(v);values.append(dict(checkpoint=checkpoint or 'reconstructed_S0_seed86',total=float(v),**d));del m;torch.cuda.empty_cache()
            result[stage]=dict(windows=len(ws),before=values[0],after=values[1],decreased=values[1]['total']<values[0]['total'])
        atomic(dest,dict(status='ready',fixed_training_diagnostic=True,history_and_state_dropout_disabled_for_matched_comparison=True,noise_seed=86,noise_step=9999,results=result,elapsed_s=guard.check(),checkpoint_selection=False));print(json.dumps(result),flush=True)
    finally:guard.finish()
if __name__=='__main__':main()
