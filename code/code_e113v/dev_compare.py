import os,sys,json
os.environ["ARM"]="CA";sys.path[:0]=["/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e102s2","/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e98w"]
import numpy as np,predicate_metrics as pm
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");B3=T/"results/e99b3_20261115_01";R=T/"results/e113v_vlm_multiframe_20261116_01"
src=json.load(open(B3/"sources.json"));dev=sorted([s for s in src if s["pool"]=="dev"],key=lambda s:s["parent"]);names=[s["parent"] for s in dev];pari={p:i for i,p in enumerate(names)}
PI={p["id"]:i for i,p in enumerate(json.load(open(B3/"spec.json"))["predicates"])}
labels=[r for r in json.load(open(B3/"dev_labels.json")) if r["causal_valid"]]
yy=np.array([{"no":0,"yes":1,"unknown":2,"not_applicable":3}[r["causal_answer"]] for r in labels]);pid=np.array([PI[r["predicate"]] for r in labels]);par=np.array([pari[r["parent"]] for r in labels]);sel=yy!=3
draws=np.stack([np.bincount(v,minlength=20) for v in np.random.default_rng(10201).integers(20,size=(2000,20))]).astype("f8")
V3={p:np.load(R/(p+"_dev_V3.npz")) for p in names}
A={}
for p in names:
    z=np.load(B3/(p+"_vlm.npz"));A[p]=dict(zip(z["frames"].tolist(),z["prob"]))
Bsc={p:np.load(B3/(p+"_scores.npz")) for p in names}
def get(kind,r):
    p=r["parent"];f=r["frame"];j=PI[r["predicate"]]
    if kind=="V3":
        fr=V3[p]["frames"].tolist();return V3[p]["prob"][fr.index(f),j]
    if kind=="a":return A[p][f][j]
    return Bsc[p]["b_prob"][f,j]
res={};arrs={}
for kind in ["a","V3","b"]:
    p=np.array([get(kind,r) for r in labels]);vals=[]
    for j in range(21):
        m=sel&(pid==j);vals.append(pm.stats_rows(yy[m],p[m],np.zeros(m.sum(),bool),.5,par[m],draws)[:,0])
    arrs[kind]=pm.mean(np.stack(vals),0)
    res[kind]=dict(value=pm.finite(arrs[kind][0]),ci95=pm.interval(arrs[kind][1:]))
for a,b in [("V3","a"),("b","V3"),("b","a")]:
    d=arrs[a]-arrs[b];res[a+"_minus_"+b]=dict(value=pm.finite(d[0]),ci95=pm.interval(d[1:]))
res["n_records"]=int(sel.sum());res["n_yes_no"]=int((yy<2).sum())
(R/"DEV_COMPARE.json").write_text(json.dumps(res,indent=1));print(json.dumps(res,indent=1))
