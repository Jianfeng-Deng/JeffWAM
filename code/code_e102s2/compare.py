# E102-S2 comparison: causal-label arms vs retrospective E102-S reference. Reads frozen dev outputs only.
import os
os.environ.setdefault("ARM","CA")
from tune import *
import predicate_metrics as pm
REF=T/"results/e102s_20261115_01"
ARMS={"CF":T/"results/e102s2_cf_20261116_01","CS":T/"results/e102s2_cs_20261116_01"}
dsorted=sorted(dev,key=lambda s:s["parent"]);names=[s["parent"] for s in dsorted];pari={p:i for i,p in enumerate(names)}
draws_idx=np.random.default_rng(10201).integers(20,size=(2000,20))
draws=np.stack([np.bincount(v,minlength=20) for v in draws_idx]).astype("f8")
labels=[r for r in oldread("dev_labels.json") if r["causal_valid"]]
yy=np.array([{"no":0,"yes":1,"unknown":2,"not_applicable":3}[r["causal_answer"]] for r in labels]);pid=np.array([PI[r["predicate"]] for r in labels]);par=np.array([pari[r["parent"]] for r in labels])
sel=yy!=3
def macro_auc(root,method):
 sc={p:np.load(root/(p+"_dev_frozen_scores.npz")) for p in names}
 p=np.array([sc[r["parent"]][method+"_prob"][r["frame"],PI[r["predicate"]]] for r in labels]);ab=np.array([sc[r["parent"]][method+"_unknown"][r["frame"],PI[r["predicate"]]]>=.5 for r in labels])
 vals=[]
 for j in range(21):
  m=sel&(pid==j);vals.append(pm.stats_rows(yy[m],p[m],ab[m],.5,par[m],draws)[:,0])
 return pm.mean(np.stack(vals),0)
def ci(a):return dict(value=pm.finite(a[0]),ci95=pm.interval(a[1:]))
out={"predicate_macro_auroc":{},"stage":{}}
auc={}
for name,root in [("R",REF)]+list(ARMS.items()):
 if not (root/"DEV_FROZEN_COMPLETE.json").exists():continue
 for m in ["H0","H1","H2","H3"]:auc[name,m]=macro_auc(root,m)
for (name,m),a in auc.items():
 out["predicate_macro_auroc"][name+"_"+m]=ci(a)
 if name!="R":out["predicate_macro_auroc"][name+"_"+m+"_minus_R_same_head"]=ci(a-auc["R",m])
 if m!="H0":out["predicate_macro_auroc"][name+"_"+m+"_minus_"+name+"_H0"]=ci(a-auc[name,"H0"])
def boots(root):
 rr=json.loads((root/"dev_frozen_parent_statistics.json").read_text());res={}
 for name,v in rr.items():
  if name=="E101_S2":continue
  v=sorted(v,key=lambda r:r["parent"]);pt=aggregate(v);bb=[aggregate([v[i] for i in ix]) for ix in draws_idx];ks=list(pt)
  res[name]=np.array([[np.nan if row[k] is None else row[k] for k in ks] for row in [pt]+bb]),ks
 return res
bs={}
for name,root in [("R",REF)]+list(ARMS.items()):
 if (root/"DEV_FROZEN_COMPLETE.json").exists():bs[name]=boots(root)
KEYS=["recall30","precision30","f1_30","false_per_min30","macro_f1","error_median30"]
for name,b in bs.items():
 for v,(a,ks) in b.items():
  out["stage"][name+"_"+v]={k:ci(a[:,ks.index(k)]) for k in KEYS}
  if name=="CS" and v in bs["CF"]:out["stage"]["CS_"+v+"_minus_CF"]={k:ci(a[:,ks.index(k)]-bs["CF"][v][0][:,ks.index(k)]) for k in KEYS}
  if name!="R" and v in bs["R"]:out["stage"][name+"_"+v+"_minus_R"]={k:ci(a[:,ks.index(k)]-bs["R"][v][0][:,ks.index(k)]) for k in KEYS}
  h0=v.split("__")[0];ini=v.split("__")[1]
  if h0!="H0":out["stage"][name+"_"+v+"_minus_"+name+"_H0"]={k:ci(a[:,ks.index(k)]-b[("H0__"+ini)][0][:,ks.index(k)]) for k in KEYS}
(T/"results/e102s2_comparison_cs_20261116.json").write_text(json.dumps(out,ensure_ascii=False,indent=1))
def f(d):return "NA" if d["value"] is None else "%.4f [%s,%s]"%(d["value"],*("NA" if x is None else "%.4f"%x for x in d["ci95"]))
for k,v in out["predicate_macro_auroc"].items():print(k,f(v))
for k,v in out["stage"].items():print(k,"recall30",f(v["recall30"]),"f1_30",f(v["f1_30"]),"false/min",f(v["false_per_min30"]))
