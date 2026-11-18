import collections
import numpy as np
import torch
import torch.nn.functional as F
from e86.data import move,collate
from .common import atomic
from .objective import chain_loss
from .data import HORIZONS

def aggregate(records, keys, resamples=2000):
    if not records:return dict(windows=0,parents=0,values=None,ci95=None)
    parents=sorted({r['parent'] for r in records})
    a=np.array([[np.mean([r[k] for r in records if r['parent']==p]) for k in keys] for p in parents])
    rng=np.random.default_rng(90)
    draws=a[rng.integers(len(a),size=(resamples,len(a)))].mean(1)
    values=dict(zip(keys,a.mean(0).tolist()))
    ci={k:np.quantile(draws[:,j],[.025,.975]).tolist() for j,k in enumerate(keys)}
    if set(('model','copy','linear'))<=set(keys):
        for baseline in ('copy','linear'):
            values['skill_'+baseline]=1-values['model']/max(values[baseline],1e-12)
            x=1-draws[:,keys.index('model')]/np.maximum(draws[:,keys.index(baseline)],1e-12)
            ci['skill_'+baseline]=np.quantile(x,[.025,.975]).tolist()
    return dict(windows=len(records),parents=len(parents),values=values,ci95=ci)

def groups(records,keys):
    strata=sorted({r['stratum'] for r in records})
    demos=sorted({r['demo'] for r in records})
    return dict(overall=aggregate(records,keys),
                strata={s:aggregate([r for r in records if r['stratum']==s],keys) for s in strata},
                demos={s:aggregate([r for r in records if r['demo']==s],keys) for s in demos})

def record_identity(row):
    return dict(parent=row['parent_component'],stratum=row['stratum'],demo='demo6' if row['stratum'].startswith('demo6') else row['stratum'])

def chain_batches(data,i,k,device):
    first=data.batch([i],(k,),'short','P1'); direct=data.batch([i],(2*k,),'short','P1')
    if not first['world_valid'].all() or not direct['world_valid'].all():return None
    shifted=data.shifted(i,k,'short','P1')
    if shifted is None or not shifted['world_valid'].all():return None
    shifted=collate([shifted]);shifted['world_positions']=shifted['world_positions'][0]
    return tuple(move(b,device) for b in (first,shifted,direct))

@torch.no_grad()
def evaluate(model,data,guard,destination,steps=4):
    model.eval(); records=[]; missing=collections.Counter(); chains=[]; additions=[]
    for i,row in enumerate(data.rows):
        ident=record_identity(row)
        for h in HORIZONS:
            ids,reason=data.index.resolve(row,h)
            if ids is None:missing[(h,reason)]+=1;continue
            guard.check(); b=move(data.batch([i],(h,),'short','P1'),'cuda:0')
            pred=model.rollout(b,steps=steps,seed=9001+i)
            cur=b['world_current']; truth=b['world_raw']; scale=b['world_scale']
            js=b['history_valid'][0].nonzero().flatten()
            j=int(js[-1]) if len(js) else 4; previous=b['obs'][0,j]
            z,_=data.index.get(row['canonical_id']);ct=z['camera_exposure_timestamp'];ratio=[]
            for cam in range(3):
                now=row['obs_rows'][4][cam];past=row['obs_rows'][j][cam]
                dt=ct[now,cam]-ct[past,cam]
                ratio.append((ct[ids[cam],cam]-ct[now,cam])/dt if dt>1e-7 else 0.)
            n=8 if h<=50 else 1
            if h>50:previous=previous.mean(1,keepdim=True)
            ratio=torch.tensor(ratio,dtype=cur.dtype,device=cur.device)[:,None].expand(3,n).reshape(1,-1,1)
            linear=cur+(cur-previous.reshape(1,-1,768))*ratio
            mse=lambda x:float(((x-truth)/scale).square().mean())
            records.append(dict(**ident,h=h,model=mse(pred),copy=mse(cur),linear=mse(linear)))
        for k in (25,50):
            guard.check(); bs=chain_batches(data,i,k,'cuda:0')
            if bs is None:continue
            _,metrics=chain_loss(model,*bs,k,use_predicted=True,steps=steps,seed=9025+i)
            chains.append(dict(**ident,k=k,chain_model=metrics['chain_supervised'],chain_gap=metrics['chain_gap']))
        a=data.get(i,(25,50),'short','P1')
        if a['world_valid'].all():
            cur=a['world_current'][:24];middle=a['world_raw'][:24];end=a['world_raw'][24:]
            direct=end-cur; summed=(middle-cur)+(end-middle)
            additions.append(dict(**ident,relative_error=float((direct-summed).norm()/direct.norm().clamp_min(1e-12)),cosine=float(F.cosine_similarity(direct.flatten(),summed.flatten(),dim=0))))
        atomic(destination.parent/'eval_status.json',dict(status='evaluating',done=i+1,total=len(data.rows)))
    result=dict(schema='e90-ckpt1-dev-v1',checkpoint_selection=False,language='fixed human short instruction only',
                input='W-only; no action/state',flow_steps=steps,noise_seed=9001,bootstrap_resamples=2000,
                estimator='equal parents, mean windows within parent; paired parent bootstrap, ratio-of-means skills',
                cached_dev_windows=len(data.rows),cached_dev_parents=len({r['parent_component'] for r in data.rows}),
                horizons={str(h):groups([r for r in records if r['h']==h],['model','copy','linear']) for h in HORIZONS},
                missing={str(h):{why:n for (hh,why),n in missing.items() if hh==h} for h in HORIZONS},
                chains={str(k):groups([r for r in chains if r['k']==k],['chain_model','chain_gap']) for k in (25,50)},
                additivity=groups(additions,['relative_error','cosine']),
                limitation='Existing frozen cached dev subset only; sparse +100; one-parent bootstrap intervals are degenerate. Additivity telescopes exact fixed encoded endpoints, not a rollout accuracy test.')
    atomic(destination,result);atomic(destination.parent/'dev_records.json',dict(world=records,chain=chains,additivity=additions))
    return result
