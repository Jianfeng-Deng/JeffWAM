import numpy as np,json
from pathlib import Path
R=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e118_pilot_20261117_02");z=np.load(R/"oof.npz",allow_pickle=True)
par=z["parent"];ps=sorted(set(par));pi={p:i for i,p in enumerate(ps)};P=np.array([pi[p] for p in par]);draws=np.random.default_rng(90).integers(len(ps),size=(2000,len(ps)))
W=np.stack([np.bincount(d,minlength=len(ps)) for d in draws]).astype("f8")
def prep(gran,metric):
    lab=z[gran];cls=sorted(set(lab));ci={c:i for i,c in enumerate(cls)};C=np.array([ci[c] for c in lab])
    tot=np.zeros((len(cls),len(ps)));np.add.at(tot,(C,P),1);moving=np.array([("hold|hold" not in c) for c in cls])
    return lab,C,tot,cls,moving
def corr(gran,v,metric):
    key=f"{gran}|{v}|{metric}";x=z[key];return x
def bal(gran,v,metric,which="all",minn=5):
    lab,C,tot,cls,moving=prep(gran,metric);x=corr(gran,v,metric);cor=np.zeros_like(tot);np.add.at(cor,(C,P),x)
    sel=np.ones(len(cls),bool) if which=="all" else (moving if which=="moving" else ~moving)
    sel&=tot.sum(1)>=minn
    def f(w):
        t=w@tot[sel].T;c=w@cor[sel].T;return np.where(t>0,c/np.maximum(t,1),np.nan).mean(1) if w.ndim>1 else None
    return f
def stat(gran,v,metric,which):
    f=bal(gran,v,metric,which);return f(np.vstack([np.ones(len(ps)),W]))
def ci(a):return a[0],np.nanquantile(a[1:],[.025,.975])
out={}
for gran in ("coarse","fine"):
  for metric in ("top1","top5"):
    for which in ("all","moving"):
        S={v:stat(gran,v,metric,which) for v in ["dI","dA","dIdA","cur","pos","cur+dI","prior","shuffled"]}
        print(f"== {gran} {metric} 类别平衡召回({which}类) classes:",int((prep(gran,metric)[2].sum(1)>=5).sum()))
        print("  ".join("%s %.3f"%(v,S[v][0]) for v in S))
        for a,b in (("dI","cur"),("dI","pos"),("dI","shuffled"),("dI","prior"),("cur+dI","cur"),("dA","pos"),("dA","shuffled"),("dA","prior"),("dIdA","dA")):
            d=S[a]-S[b];m,c=ci(d);out[f"{gran}|{metric}|{which}|{a}-{b}"]=[float(m),[float(c[0]),float(c[1])]]
            print("   %-12s %+.3f [%+.3f,%+.3f]"%(a+"−"+b,m,c[0],c[1]))
json.dump(out,open(R/"summary.json","w"),indent=1)
