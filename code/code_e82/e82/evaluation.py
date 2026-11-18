from __future__ import annotations
from collections import defaultdict
import numpy as np

GROUPS={'left_joint':list(range(6)),'left_gripper':[6],'right_joint':list(range(7,13)),'right_gripper':[13]}
HORIZONS=(8,10,25,32,50)

def window_metrics(pred,true,scale):
    pred=np.asarray(pred); true=np.asarray(true); scale=np.asarray(scale)
    assert pred.shape==true.shape and pred.shape[-2:]==(50,14)
    if not np.isfinite(pred).all() or not np.isfinite(true).all(): raise ValueError('Nonfinite evaluation')
    err=pred-true; result={}
    for k in HORIZONS:
        norms=[]
        for j in range(14):
            e=err[:,:k,j]; units=1000 if j in (6,13) else 1; suffix='mm' if j in (6,13) else 'rad'
            result[f'dim{j}/K{k}/MAE_{suffix}']=np.abs(e).mean(1)*units
            result[f'dim{j}/K{k}/RMSE_{suffix}']=np.sqrt((e*e).mean(1))*units
        for group,ids in GROUPS.items():
            e=err[:,:k,ids]; units=1000 if 'gripper' in group else 1
            suffix='mm' if 'gripper' in group else 'rad'
            result[f'{group}/K{k}/MAE_{suffix}']=np.abs(e).mean((1,2))*units
            result[f'{group}/K{k}/RMSE_{suffix}']=np.sqrt((e*e).mean((1,2)))*units
            norm=np.sqrt(((e/scale[None,:k,ids])**2).mean((1,2)))
            result[f'{group}/K{k}/normalized_RMSE']=norm; norms.append(norm)
        result[f'K{k}/group_equal_RMSE']=np.mean(norms,0)
    return result

def parent_equal(values,parents,sessions):
    sums=defaultdict(list); p_sessions={}
    for x,p,s in zip(values,parents,sessions):
        s=str(s) if s else 'unknown'
        if p in p_sessions and p_sessions[p]!=s: raise ValueError('Parent assigned inconsistent session')
        sums[p].append(float(x)); p_sessions[p]=s
    per={p:float(np.mean(xs)) for p,xs in sums.items()}
    return dict(mean=float(np.mean(list(per.values()))),parents=per,sessions=p_sessions,n_parents=len(per))

def paired_bootstrap(r,p,draws=10000,seed=8101,hierarchical=True,strata=None):
    """Pair R/P by exact parent IDs; session->parent cluster bootstrap, within strata."""
    if set(r['parents'])!=set(p['parents']) or r['sessions']!=p['sessions']: raise ValueError('Unpaired parents/sessions')
    parents=sorted(r['parents']); strata=strata or {x:'all' for x in parents}
    strata_groups=defaultdict(list)
    for x in parents: strata_groups[strata[x]].append(x)
    rng=np.random.default_rng(seed); samples=[]; diffs=[]
    for _ in range(draws):
        ids=[]
        for _,ps in sorted(strata_groups.items()):
            if hierarchical:
                by_session=defaultdict(list)
                for x in ps: by_session[r['sessions'][x]].append(x)
                ss=sorted(by_session)
                for s in rng.choice(ss,len(ss),replace=True):
                    pool=by_session[s]; ids.extend(rng.choice(pool,len(pool),replace=True))
            else: ids.extend(rng.choice(ps,len(ps),replace=True))
        rv=np.mean([r['parents'][i] for i in ids]); pv=np.mean([p['parents'][i] for i in ids])
        diffs.append(pv-rv)
        if abs(rv)>1e-12: samples.append(pv/rv-1)
    n_sessions=len(set(r['sessions'].values()))
    return dict(relative_change=p['mean']/r['mean']-1 if abs(r['mean'])>1e-12 else None,
                relative_ci95=np.quantile(samples,[.025,.975]).tolist() if samples else None,
                absolute_difference=p['mean']-r['mean'],absolute_ci95=np.quantile(diffs,[.025,.975]).tolist(),
                n_parents=len(parents),n_sessions=n_sessions,session_generalization_unstable=n_sessions<10,
                hierarchical=hierarchical,draws=draws)

def choose_checkpoint(candidates):
    # Inputs are human/recorded-feedback dev action metrics, never world loss/model labels.
    eligible=[c for c in candidates if c['split']=='dev' and c['truth_source'] in ('human','recorded_feedback') and c['guardrails_pass']]
    if not eligible: raise ValueError('No eligible dev checkpoint')
    return min(eligible,key=lambda c:(c['action_only_K8'],c['joint_step']))

def choose_deployment(record):
    if record.get('split')!='dev' or record.get('truth_source') not in ('human','recorded_feedback'):
        raise ValueError('Deployment selection requires independent dev action truth')
    return 'joint' if record['joint_over_action_only_K8']<=.98 and record['joint_over_action_only_K32']<=1.02 and record['worst_group_ratio']<=1.02 and record['latency_qualified'] else 'action_only'
