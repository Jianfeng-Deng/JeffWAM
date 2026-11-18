import os,sys,json,pickle
os.environ["ARM"]="CF";sys.path[:0]=["code_e102s2","code_e86"]
from s2 import *
PR=pickle.load(open(T/"results/e115p1_jerk_20261118_01/preds.pkl","rb"))
ent={e["index"]:e for e in json.loads((T/"results/e106c_control_20261115_02/PRETRAIN_LOCK.json").read_text())["entries"]}
JOINTS=[0,1,2,3,4,5,7,8,9,10,11,12]
def bd(parent,a):
    y=truths[parent][a:a+51];y=y[y>=0];return len(y)>1 and (y!=y[0]).any()
data={}
for arm,idx,pred,truth in PR:
    p=pred[:,JOINTS];tt=truth[:,JOINTS]
    d2=np.diff(p,2,axis=0);d3=np.diff(p,3,axis=0);t2=np.diff(tt,2,axis=0);t3=np.diff(tt,3,axis=0)
    data[(arm,idx)]=dict(acc=float((d2**2).mean()),jerk=float((d3**2).mean()),tacc=float((t2**2).mean()),tjerk=float((t3**2).mean()),parent=ent[idx]["parent"],bd=bd(ent[idx]["parent"],ent[idx]["anchor"]))
idxs=sorted({i for _,i in data});ps=sorted({v["parent"] for v in data.values()});draws=np.random.default_rng(90).integers(len(ps),size=(2000,len(ps)))
def stat(a,key,kind,ref=None):
    def f(sel):
        vals=[data[(a,i)][key]-(data[(ref,i)][key] if ref else 0) for i in idxs if (kind=="all" or (kind=="bd" and data[(a,i)]["bd"]) or (kind=="in" and not data[(a,i)]["bd"])) and data[(a,i)]["parent"] in sel]
        return np.mean(vals) if vals else np.nan
    return f
def boot(f):
    pt=f(set(ps));bs=[f({ps[i] for i in d}) for d in draws];return pt,np.nanquantile(bs,[.025,.975])
print("窗口:全部",len(idxs),"含切换",sum(data[("S0",i)]["bd"] for i in idxs))
for key,label in (("acc","二阶差分能量(加速度)"),("jerk","三阶差分能量(jerk)")):
    print("==",label,"（均值；真值作参照）")
    for kind in ("all","bd","in"):
        row=[]
        for a in ("D","S0","S1","S2"):row.append("%s %.3e"%(a,boot(stat(a,key,kind))[0]))
        tr=np.mean([data[("S0",i)]["t"+key] for i in idxs if (kind=="all" or (kind=="bd")==data[("S0",i)]["bd"])])
        print("  %-3s"%kind,"  ".join(row),"| 真值 %.3e"%tr)
    for kind in ("all","bd","in"):
        for a in ("S1","S2"):
            pt,c=boot(stat(a,key,kind,"S0"));print("  %s S%s−S0 %+.3e [%+.3e,%+.3e]"%(kind,a[1],pt,c[0],c[1]))
