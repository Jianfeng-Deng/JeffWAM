# E115-P2 step B (CPU): predicate probe + stage emission applied to pooled real / copy / predicted future tokens.
import os,sys,json
os.environ["ARM"]="CF";sys.path[:0]=["/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e102s2","/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e86"]
from s2 import *
D=T/"results/e115p2_pooled_20261117_02";CF=T/"results/e102s2_cf_20261116_01"
z=np.load(D/"pooled.npz");rows=json.loads((D/"rows.json").read_text())
with (CF/"H0_full.pkl").open("rb") as f:head=pickle.load(f)
with (CF/"final_aux.pkl").open("rb") as f:aux=pickle.load(f)["H0"]
COND=["real","cur","gt","n1","no","sh"];prob={};stg={}
for c in COND:
    p,_=predict(head,z[c]);prob[c]=p;stg[c]=emission(p,aux).argmax(1)
par=np.array([r["parent"] for r in rows]);H=np.array([r["h"] for r in rows]);an=np.array([r["anchor"] for r in rows])
tf=np.array([truths[r["parent"]][r["anchor"]+r["h"]] if r["anchor"]+r["h"]<len(truths[r["parent"]]) else -1 for r in rows]);ta=np.array([truths[r["parent"]][r["anchor"]] for r in rows])
ps=sorted(set(par));draws=np.random.default_rng(90).integers(len(ps),size=(2000,len(ps)));pidx={p:np.where(par==p)[0] for p in ps}
def acc(c,sel):
    m=sel&(tf>=0);return (stg[c][m]==tf[m]).mean() if m.any() else np.nan
def fid(c,sel):return (stg[c][sel]==stg["real"][sel]).mean()
def pcorr(c,sel):
    a=prob[c][sel];b=prob["real"][sel];out=[]
    for j in range(21):
        if j==PI["P073"] or a[:,j].std()<1e-6 or b[:,j].std()<1e-6:continue
        out.append(np.corrcoef(a[:,j],b[:,j])[0,1])
    return np.mean(out) if out else np.nan
def boot(fn):
    pt=fn(np.arange(len(rows)));bs=[]
    for d in draws:
        idx=np.concatenate([pidx[ps[i]] for i in d]);bs.append(fn(idx))
    bs=np.array(bs);return pt,np.nanquantile(bs,[.025,.975])
def mk(h,fn):
    def f(idx):
        sel=np.zeros(len(rows),bool);sel[idx]=True;sel&=(H==h)   # note: duplicates collapse; use weights below
        return fn(sel)
    return f
# parent bootstrap with multiplicities: build index list directly (duplicates preserved)
def stat(h,kind,c,ref=None):
    def f(idx):
        idx=idx[H[idx]==h]
        if kind=="acc":
            m=tf[idx]>=0;return (stg[c][idx][m]==tf[idx][m]).mean() if m.any() else np.nan
        if kind=="chg":
            m=(tf[idx]>=0)&(ta[idx]>=0)&(tf[idx]!=ta[idx]);return (stg[c][idx][m]==tf[idx][m]).mean() if m.sum()>3 else np.nan
        if kind=="fid":return (stg[c][idx]==stg["real"][idx]).mean()
        if kind=="pcorr":
            a=prob[c][idx];b=prob["real"][idx];o=[]
            for j in range(21):
                if j==PI["P073"] or a[:,j].std()<1e-6 or b[:,j].std()<1e-6:continue
                o.append(np.corrcoef(a[:,j],b[:,j])[0,1])
            return np.mean(o) if o else np.nan
    return f
def run(h,kind,c,base=None):
    f=stat(h,kind,c);g=stat(h,kind,base) if base else None
    h_=lambda idx:(f(idx)-g(idx)) if g else f(idx)
    pt=h_(np.arange(len(rows)));bs=[]
    for d in draws:bs.append(h_(np.concatenate([pidx[ps[i]] for i in d])))
    bs=np.array(bs);return pt,np.nanquantile(bs,[.025,.975])
out={};fmt=lambda t:"%.3f [%.3f,%.3f]"%(t[0],*t[1])
print("windows",len(rows)//4,"parents",len(ps))
for h in (10,25,50,100):
    n=int((H==h).sum());print("=== horizon +%d (windows %d, change-windows %d)"%(h,n,int(((H==h)&(tf>=0)&(ta>=0)&(tf!=ta)).sum())))
    for kind,label in (("acc","阶段准确率(对原真值)"),("chg","阶段准确率·阶段发生变化的窗口"),("fid","与真实未来token的阶段一致率"),("pcorr","谓词概率相关(对真实未来token)")):
        line=[]
        for c in COND:
            if kind in("fid","pcorr") and c=="real":continue
            r=run(h,kind,c);out[f"{h}|{kind}|{c}"]=[float(r[0]),[float(x) for x in r[1]]];line.append("%s %s"%(c,fmt(r)))
        print(label+": "+" | ".join(line))
    for kind in("acc","chg","fid","pcorr"):
        for c in ("gt","n1","no","sh"):
            r=run(h,kind,c,"cur");out[f"{h}|{kind}|{c}-cur"]=[float(r[0]),[float(x) for x in r[1]]]
        print("  Δ(预测−复制当前) %-5s"%kind," | ".join("%s %s"%(c,fmt(out[f"{h}|{kind}|{c}-cur"][0:1]+[out[f"{h}|{kind}|{c}-cur"][1]])) for c in ("gt","n1","no","sh")))
(D/"summary.json").write_text(json.dumps(out,indent=1))
