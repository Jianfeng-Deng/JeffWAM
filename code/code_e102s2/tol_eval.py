import os,sys,json
os.environ["ARM"]="CA";sys.path[:0]=["code_e102s2","code_e86"]
from tune import *
TOL=[30,60,90,150]
dsorted=sorted(dev,key=lambda s:s["parent"]);names=[s["parent"] for s in dsorted]
draws=np.random.default_rng(10201).integers(20,size=(2000,20))
roots={"R":T/"results/e102s_20261115_01","CA":T/"results/e102s2_ca_20261116_01","CF":T/"results/e102s2_cf_20261116_01","CS":T/"results/e102s2_cs_20261116_01"}
stat={}  # (arm,variant,tol)->array [20,3] matched,npred,ntrue
def counts(seq,true,tol):
    pb=boundaries(seq);tb=boundaries(true);return len(match(pb,tb,tol)),len(pb),len(tb)
for arm,r in roots.items():
    seqs={p:np.load(r/(p+"_dev_frozen_sequences.npz")) for p in names}
    keys=[k for k in seqs[names[0]].files if k!="truth"]
    for v in keys:
        if arm!="R" and v=="E101_S2":continue
        for t in TOL:
            stat[arm,v,t]=np.array([counts(seqs[p][v],truths[p],t) for p in names],float)
def agg(a,idx=None):
    a=a if idx is None else a[idx];m,pp,tt=a.sum(0);return m/tt,m/pp,2*m/(tt+pp)  # recall,precision,f1
def ci(f):
    pt=f(None);bs=np.array([f(d) for d in draws]);return pt,np.quantile(bs,[.025,.975])
def show(label,fn):
    out=[]
    for t in TOL:
        pt,c=ci(lambda idx:fn(t,idx)[0])
        out.append("±%d: %.3f [%.3f,%.3f]"%(t,pt,c[0],c[1]))
    print("%-34s recall  "%label+" | ".join(out))
def diff(label,a,b):
    out=[]
    for t in TOL:
        pt,c=ci(lambda idx:agg(stat[a[0],a[1],t],idx)[0]-agg(stat[b[0],b[1],t],idx)[0]);out.append("±%d: %+.3f [%+.3f,%+.3f]"%(t,pt,c[0],c[1]))
    print("%-34s Δrecall  "%label+" | ".join(out))
print("== 20 个 dev 片段，对原模型真值，边界召回（fixed_S01）")
for arm in roots:
    for h in ["H0","H1","H2","H3"]:
        show(arm+"_"+h,lambda t,idx,arm=arm,h=h:agg(stat[arm,h+"__fixed_S01",t],idx))
show("E101_S2",lambda t,idx:agg(stat["R","E101_S2",t],idx))
print("== 配对差（历史是否有用：H1-H3 相对 H0）")
for arm in ["R","CF","CS"]:
    for h in ["H1","H2","H3"]:diff(arm+"_"+h+" − "+arm+"_H0",(arm,h+"__fixed_S01"),(arm,"H0__fixed_S01"))
print("== 配对差（标签来源：因果标签 vs 原版）")
for arm in ["CA","CF","CS"]:diff(arm+"_H0 − R_H0",(arm,"H0__fixed_S01"),("R","H0__fixed_S01"))
diff("CS_H0 − CF_H0",("CS","H0__fixed_S01"),("CF","H0__fixed_S01"))
print("== 精确率/假切换（±90，fixed_S01，H0）")
for arm in roots:
    r,p,f=agg(stat[arm,"H0__fixed_S01",90]);print(arm,"recall %.3f precision %.3f f1 %.3f"%(r,p,f))
