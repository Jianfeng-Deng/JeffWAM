import os,sys,json
os.environ["ARM"]="CF";sys.path[:0]=["code_e102s2","code_e86"]
from s2 import *
rows=json.loads((T/"results/e115p1_eval_20261117_01/rows.json").read_text())
ent={e["index"]:e for e in json.loads((T/"results/e106c_control_20261115_02/PRETRAIN_LOCK.json").read_text())["entries"]}
def has_boundary(parent,anchor,span=50):
    y=truths[parent][anchor:anchor+span+1];y=y[y>=0]
    return len(y)>1 and (y!=y[0]).any()
def near_boundary(parent,anchor,span=50,pad=30):
    lo=max(0,anchor-pad);y=truths[parent][lo:anchor+span+1+pad];y=y[y>=0];return len(y)>1 and (y!=y[0]).any()
for r in rows:
    e=ent[r["index"]];r["bd"]=has_boundary(r["parent"],e["anchor"]);r["near"]=near_boundary(r["parent"],e["anchor"])
arms=["D","S0","S1","S2"];ps=sorted({r["parent"] for r in rows});draws=np.random.default_rng(90).integers(len(ps),size=(2000,len(ps)))
def sel(arm,kind):return [r for r in rows if r["arm"]==arm and (kind=="all" or (kind=="boundary" and r["bd"]) or (kind=="interior" and not r["bd"]) or (kind=="near" and r["near"]))]
def metric(rs,name):
    if name=="l1_8":return np.mean([r["l1_8"] for r in rs]);
    if name=="l1_50":return np.mean([r["l1_50"] for r in rs])
    if name=="acc":return np.mean([r["acc"] for r in rs])
    if name=="f1":
        tp=sum(r["ev"]["10"][0] for r in rs);pr=sum(r["ev"]["10"][1] for r in rs);tt=sum(r["ev"]["10"][2] for r in rs);return 2*tp/max(pr+tt,1)
def diff(a,b,kind,name):
    A=sel(a,kind);B=sel(b,kind)
    pa={p:[r for r in A if r["parent"]==p] for p in ps};pb={p:[r for r in B if r["parent"]==p] for p in ps}
    def f(idx):
        x=[r for i in idx for r in pa[ps[i]]];y=[r for i in idx for r in pb[ps[i]]]
        return metric(x,name)-metric(y,name) if x and y else np.nan
    pt=f(np.arange(len(ps)));bs=np.array([f(d) for d in draws]);return pt,np.nanquantile(bs,[.025,.975])
for kind in ("all","boundary","interior","near"):
    n=len({r["index"] for r in sel("S0",kind)});print("== 窗口类型",kind,"窗口数",n,"parent数",len({r["parent"] for r in sel("S0",kind)}))
    for a,b in (("S1","S0"),("S2","S0")):
        for name in ("l1_8","l1_50","acc","f1"):
            pt,c=diff(a,b,kind,name);print("  %s−%s %-5s %+.5f [%+.5f,%+.5f]"%(a,b,name,pt,c[0],c[1]))
    print("  S0 基线 l1_8 %.4f l1_50 %.4f copy50 %.4f"%(metric(sel("S0",kind),"l1_8"),metric(sel("S0",kind),"l1_50"),np.mean([r["copy50"] for r in sel("S0",kind)])))
