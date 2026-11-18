import json,collections,numpy as np,glob
from pathlib import Path
from scipy.stats import rankdata
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");R=T/"results/e113v_vlm_multiframe_20261116_01";B3=T/"results/e99b3_20261115_01"
src=json.load(open(B3/"sources.json"));cid2p={x["cid"]:x["parent"] for x in src}
spec=json.load(open(B3/"spec.json"));PI={p["id"]:i for i,p in enumerate(spec["predicates"])}
lab={}
for l in open(T/"data/e98_predicate_labels_train_causal_v1/predicates_causal.jsonl"):
    r=json.loads(l);p=cid2p.get(r["canonical_id"])
    if p and r["causal_observable_answer"] in("yes","no"):lab[p,r["source_frame"],PI[r["predicate"]]]=int(r["causal_observable_answer"]=="yes")
rows=[];P={v:[] for v in["V0","V1","V2","V3"]};par=[];pid=[];y=[]
for f in sorted(R.glob("pc_*_train.npz")):
    p=f.name[:-10];z=np.load(f)
    for i,fr in enumerate(z["frames"]):
        for j in range(21):
            k=(p,int(fr),j)
            if k in lab:
                y.append(lab[k]);pid.append(j);par.append(p)
                for v in P:P[v].append(z[v][i,j])
y=np.array(y);pid=np.array(pid);par=np.array(par);P={v:np.array(a) for v,a in P.items()}
def auc(yy,pp):
    n=yy.sum();m=len(yy)-n
    return np.nan if n*m==0 else (rankdata(pp)[yy==1].sum()-n*(n+1)/2)/(n*m)
def macro(v,idx=None):
    idx=np.arange(len(y)) if idx is None else idx;vals=[]
    for j in range(21):
        mm=idx[pid[idx]==j];a=auc(y[mm],P[v][mm])
        if np.isfinite(a):vals.append(a)
    return float(np.mean(vals)),len(vals)
ps=sorted(set(par));rng=np.random.default_rng(113);out={"n_labels":len(y),"n_yes":int(y.sum()),"parents":len(ps)}
byp={p:np.where(par==p)[0] for p in ps}
for v in P:
    pt,k=macro(v);bs=[]
    for _ in range(500):
        d=rng.integers(len(ps),size=len(ps));idx=np.concatenate([byp[ps[i]] for i in d]);bs.append(macro(v,idx)[0])
    out[v]=dict(macro_auroc=pt,predicates_estimable=k,ci95=[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))])
# paired vs V0
base=[]
for v in["V1","V2","V3"]:
    ds=[]
    rng=np.random.default_rng(113)
    for _ in range(500):
        d=rng.integers(len(ps),size=len(ps));idx=np.concatenate([byp[ps[i]] for i in d]);ds.append(macro(v,idx)[0]-macro("V0",idx)[0])
    out[v]["minus_V0"]=dict(value=out[v]["macro_auroc"]-out["V0"]["macro_auroc"],ci95=[float(np.quantile(ds,.025)),float(np.quantile(ds,.975))])
order=["V0","V1","V2","V3"];best=max(order,key=lambda v:(round(out[v]["macro_auroc"],4),-order.index(v)))
out["selected"]=best
(R/"TRAIN_SELECTION.json").write_text(json.dumps(out,indent=1))
(R/"VARIANT_LOCK.json").write_text(json.dumps(dict(variant=best,rule="highest train-subset macro AUROC, tie to simpler variant, no dev used",train_selection=str(R/"TRAIN_SELECTION.json")),indent=1))
print(json.dumps(out,indent=1))
