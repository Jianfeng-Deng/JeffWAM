"""Locked independent evaluation executor. Never fit stats or use model labels as truth."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .data import ManifestDataset,StatContract,read_ready,sha256,move
from .model import PiperWAM
from .train import load
from .inference import infer
from .runtime import LatencyRecorder,Budget,atomic_json,brt

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--stats',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--split',choices=['dev','test','exploratory_dev','exploratory_test'],required=True);p.add_argument('--mode',choices=['action_only','joint'],default='action_only');p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);p.add_argument('--ledger');a=p.parse_args()
    if Path(a.output).exists() or Path(a.output+'.json').exists():raise FileExistsError(a.output)
    c=Config.load(a.config)
    if a.device.startswith('cuda') and not a.ledger:raise ValueError('GPU evaluation requires the paired shared budget ledger')
    budget=Budget(a.ledger or a.output+'.budget.jsonl',int(a.device.startswith('cuda')),c.budget_gpuh-4.,a.output)
    read_ready(c.data_root);stats=StatContract.load(a.stats)
    if stats.d.get('fixture'):raise ValueError('Real evaluation cannot use fixture scales')
    ds=ManifestDataset(a.manifest,'P3_joint',a.split);model=PiperWAM(c,inference_only=True).to(a.device);load(a.checkpoint,model,device=a.device);model.eval()
    prediction=[];truth=[];parents=[];sessions=[];windows=[];sources=[];rec=LatencyRecorder();unknown=0
    for i,r in enumerate(ds.rows):
        budget.check()
        if a.device.startswith('cuda'):torch.cuda.synchronize()
        start=time.monotonic();b=move(ds[i],a.device);info={}
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda')):pred=infer(model,b,stats,a.mode,diagnostics=info)
        if a.device.startswith('cuda'):torch.cuda.synchronize()
        rec.add(time.monotonic()-start);budget.tick();unknown+=info['UNK_count']
        prediction.append(pred.cpu().float().numpy()[0]);truth.append(b['action'].cpu().numpy()[0]);parents.extend(b['parent_component']);sessions.extend(b['session_id']);windows.extend(b['window_id']);sources.append(r['truth_source'])
    with open(a.output,'xb') as f:np.savez_compressed(f,prediction=np.asarray(prediction),truth=np.asarray(truth),window_id=np.asarray(windows),parent_component=np.asarray(parents),session_id=np.asarray(sessions),truth_source=np.asarray(sources),split=np.asarray([a.split]*len(windows)))
    atomic_json(a.output+'.json',dict(gpuh=budget.tick(),brt=brt(),mode=a.mode,manifest_sha256=sha256(a.manifest),stats_sha256=sha256(a.stats),checkpoint_sha256=sha256(a.checkpoint),latency=rec.summary(),UNK_count=unknown,n=len(windows),note='full raw decode + visual + four noises + 10 Euler; scheduling queue outside process not measured'))
if __name__=='__main__':main()
