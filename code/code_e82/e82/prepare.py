from __future__ import annotations
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .config import Config,GROUPS,TOKEN_TABLE
from .data import DataPackageDataset,JOINTS,GRIPPERS,collate,move,sha256
from .randomness import seed_all
from .model import PiperWAM
from .train import save
from .runtime import atomic_json,brt

def fit_numeric(ds):
    from .source import numeric
    import h5py
    observed={}
    # Every admitted window, parent equal weights. No dev/test source is opened.
    groups={}
    for r in ds.rows:groups.setdefault(r['parent_component'],[]).append(r)
    sums={k:None for k in ('a','q','dq','h')}; seconds={k:None for k in sums}
    for parent,rows in groups.items():
        local={k:[] for k in sums}
        for r in rows:
            z=numeric(r['numeric_cache']); i=r['anchor']; 
            if r['source_path'] not in observed:
                with h5py.File(r['source_path'],'r') as f: observed[r['source_path']]=f['observations/qpos'][:].astype(np.float64)
            oq=observed[r['source_path']];q=oq[i].copy(); a=z['action'][i:i+50].copy(); a[:,JOINTS]-=q[None,JOINTS]
            local['a'].append(a);local['q'].append(q);local['dq'].append((oq[i]-oq[i-1])/(z['timestamp'][i]-z['timestamp'][i-1]));local['h'].extend(oq[max(0,i-7):i+1])
        for k,xs in local.items():
            x=np.asarray(xs,dtype=np.float64); mean=x.mean(0); mean2=(x*x).mean(0)
            sums[k]=mean if sums[k] is None else sums[k]+mean
            seconds[k]=mean2 if seconds[k] is None else seconds[k]+mean2
    out={}; n=len(groups)
    if not n:raise ValueError('No allowed P3 train parents for stat fit')
    for k in sums:
        mu=sums[k]/n;std=np.sqrt(np.maximum(0,seconds[k]/n-mu*mu));floor=np.full(14,.01);floor[GRIPPERS]=.001
        out[k]=(mu,np.maximum(std,floor))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['s0','stats','counts']);p.add_argument('--config');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--fixture',action='store_true');p.add_argument('--world-parents',type=int,default=32);args=p.parse_args()
    c=Config.load(args.config) if args.config else Config();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);seed_all(c.seed)
    if args.operation=='stats':
        # Refuses pending P3 quality rather than silently fit final scales on a candidate pool.
        ds3=DataPackageDataset(c.data_root,'P3_joint');ds1=DataPackageDataset(c.data_root,'P1')
        numeric_stats=fit_numeric(ds3);m=PiperWAM(c).to(args.device).eval();groups={}
        for i,r in enumerate(ds1.rows):groups.setdefault(r['parent_component'],[]).append(i)
        rng=np.random.default_rng(c.seed); parents=sorted(groups); selected=rng.choice(len(parents),min(len(parents),args.world_parents),replace=False)
        ids=[groups[parents[k]][int(rng.integers(len(groups[parents[k]])))] for k in selected]
        if not ids:raise ValueError('No train P1 parents for W stats')
        sums=np.zeros((91,768),np.float64);sums2=sums.copy();counts=np.zeros(91,np.int64)
        with torch.no_grad():
            for i in ids:
                b=move(ds1[i],args.device);z,v=m.world_target(b);z=z[0].cpu().numpy();v=v[0].cpu().numpy()
                sums+=(z*v[...,None]).sum(0);sums2+=(z*z*v[...,None]).sum(0);counts+=v.sum(0)
        mean=sums/np.maximum(counts[:,None],1);std=np.sqrt(np.maximum(0,sums2/np.maximum(counts[:,None],1)-mean*mean))
        for a,b,w in GROUPS:
            count=counts[a:b].sum(); mu=sums[a:b].sum(0)/max(1,count); var=sums2[a:b].sum(0)/max(1,count)-mu*mu
            mean[a:b]=mu; std[a:b]=np.sqrt(np.maximum(0,var))
            pos=std[a:b][std[a:b]>0];floor=max(float(np.median(pos))*.1 if pos.size else 0,1e-6);std[a:b]=np.maximum(std[a:b],floor)
        d=dict(fitted_split='train',world_fitted_stage='P1',brt=brt(),data_refs=ds3.refs,world_parent_ids=[ds1.rows[i]['parent_component'] for i in ids],action_parent_ids=sorted({r['parent_component'] for r in ds3.rows}),encoder_sha256=sha256(c.dino_weights),world_mean=mean.tolist(),world_std=std.tolist())
        for k,label in [('a','action'),('q','q'),('dq','dq'),('h','history')]:d[label+'_mean']=numeric_stats[k][0].tolist();d[label+'_std']=numeric_stats[k][1].tolist()
        atomic_json(out,d);return
    m=PiperWAM(c,fixture=args.fixture)
    if args.operation=='counts':atomic_json(out,dict(parameters=m.parameter_counts(),tokens=TOKEN_TABLE,brt=brt(),fixture=args.fixture));return
    save(out,m,None,0,dict(seed=c.seed,stage='S0',config=asdict(c),config_hash=c.digest(),fixture=args.fixture,encoder_sha256=None if args.fixture else sha256(c.dino_weights),brt=brt()))
if __name__=='__main__':main()
