import json,sys
from pathlib import Path
import numpy as np
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");OUT=T/"results/e110f_eval_20261116_01"
raw=json.loads((OUT/"raw_eval.json").read_text())
def boot_idx(n):
 return np.random.default_rng(90).integers(0,n,size=(2000,n))
def ci(x):return [float(v) for v in np.quantile(np.asarray(x),[.025,.975])]
def parent_action(rows,subset):
 rr=[r for r in rows if subset=="main" or r["original48"]]
 g={}
 for r in rr:g.setdefault(r["parent"],[]).append(r)
 return {p:{k:float(np.mean([x[k] for x in v])) for k in v[0] if k not in ("group","index","parent","original48","collapsed")} | {"collapse_rate":float(np.mean([x.get("collapsed",False) for x in v]))} for p,v in g.items()}
def summarize_action(rows,subset):
 p=parent_action(rows,subset);parents=sorted(p);idx=boot_idx(len(parents));metrics=list(p[parents[0]])
 out={"parents":len(parents)}
 for m in metrics:
  a=np.asarray([p[x][m] for x in parents]);out[m]={"mean":float(a.mean()),"ci95":ci(a[idx].mean(1))}
 for h in (8,50):
  a=np.asarray([p[x][f"l1_{h}"] for x in parents]);cp=np.asarray([p[x][f"copy{h}"] for x in parents])
  out[f"relative_copy_improvement_{h}_pct"]=float(100*(cp.mean()-a.mean())/cp.mean())
 return out,p
def parent_grip(rows,subset,tol):
 rr=[r for r in rows if (subset=="main" or r["original48"])]
 g={}
 for r in rr:
  p=g.setdefault(r["parent"],dict(tp=0,pred=0,true=0,minutes=0.,state=[]))
  e=r["events"][str(tol)];p["tp"]+=e["tp"];p["pred"]+=e["pred"];p["true"]+=e["true"];p["minutes"]+=e["minutes"];p["state"].append(r["state_accuracy"])
 out={}
 for p,x in g.items():
  out[p]=dict(f1=2*x["tp"]/max(1,x["pred"]+x["true"]),false=(x["pred"]-x["tp"])/max(x["minutes"],1e-9),state=float(np.mean(x["state"])),tp=x["tp"],pred=x["pred"],true=x["true"])
 return out
def summarize_grip(rows,subset):
 p10=parent_grip(rows,subset,10);parents=sorted(p10);idx=boot_idx(len(parents));ret={"parents":len(parents)}
 for tol in (5,10,20):
  pp=parent_grip(rows,subset,tol);f=np.asarray([pp[x]["f1"] for x in parents]);fs=np.asarray([pp[x]["false"] for x in parents])
  ret[str(tol)]=dict(f1=float(f.mean()),f1_ci95=ci(f[idx].mean(1)),false_switches_per_minute=float(fs.mean()),false_ci95=ci(fs[idx].mean(1)))
 s=np.asarray([p10[x]["state"] for x in parents]);ret["state_accuracy"]=dict(mean=float(s.mean()),ci95=ci(s[idx].mean(1)))
 return ret,p10
def parent_world(rows,subset):
 rr=[r for r in rows if subset=="main" or r["original48"]]
 g={}
 for r in rr:
  p=g.setdefault(r["parent"],dict(mse=[],copy=[],no=[]));p["mse"].append(r["mse"]);p["copy"].append(r["copy"]);p["no"].append(r["no_traj_mse"])
 return {p:dict(skill=100*(1-np.mean(x["mse"])/max(np.mean(x["copy"]),1e-12)),mse=np.mean(x["mse"]),copy=np.mean(x["copy"]),no_traj=100*(1-np.mean(x["no"])/max(np.mean(x["copy"]),1e-12))) for p,x in g.items()}
def summarize_world(rows,subset):
 out={}
 for tag in sorted({r["group"] for r in rows}):
  for h in (25,50,100):
   for method in ("N1","N8"):
    r=[x for x in rows if x["group"]==tag and x["h"]==h and x["method"]==method]
    if not r:continue
    p=parent_world(r,subset);parents=sorted(p);idx=boot_idx(len(parents));a=np.asarray([p[x]["skill"] for x in parents])
    out[f"{tag}|h{h}|{method}"]=dict(parents=len(parents),skill_pct=float(a.mean()),ci95_pct=ci(a[idx].mean(1)),mean_mse=float(np.mean([p[x]["mse"] for x in parents])),copy_mse=float(np.mean([p[x]["copy"] for x in parents])),no_traj_skill_pct=float(np.mean([p[x]["no_traj"] for x in parents])))
 return out
summary={"status":"complete","sample_stream_sha256":raw["sample_stream_sha256"],"checkpoint_sha256":{"D":raw["d_sha"],"F":raw["f_sha"]},"n_windows":211,"n_parents":27,"original48_windows":48,"subsets":{}}
parent_by_subset={}
for subset in ("main","original48"):
 actions={};grips={};worldrows={}
 for tag in ("D","S1","S2","S4","S8"):
  ar=[r for r in raw["actions"] if r["group"]==tag]
  er=[r for r in raw["events"] if r["group"]==tag]
  actions[tag],p=summarize_action(ar,subset)
  grips[tag],gp=summarize_grip(er,subset)
  parent_by_subset[(subset,tag)]={"action":p,"grip":gp}
 wr=[r for r in raw["world"] if r["group"] in ("D","S1","S2","S4","S8")]
 summary["subsets"][subset]=dict(action=actions,gripper=grips,world=summarize_world(wr,subset))
 comparisons={}
 for tag in ("S1","S2","S4","S8"):
  fd=parent_by_subset[(subset,tag)]["action"];dd=parent_by_subset[(subset,"D")]["action"];parents=sorted(set(fd)&set(dd));idx=boot_idx(len(parents));c={}
  for h in (8,50):
   for name,fmetric in (("single",f"l1_{h}"),("k8_mean",f"k8mean{h}")):
    a=np.asarray([fd[p][fmetric] for p in parents]);b=np.asarray([dd[p][f"l1_{h}"] for p in parents]);x=(a-b)/b
    c[f"{name}_l1_{h}_relative_diff_pct"]=dict(mean_pct=float(100*x.mean()),ci95_pct=[100*z for z in ci(x[idx].mean(1))])
   best=np.asarray([fd[p][f"best{h}"] for p in parents]);base=np.asarray([dd[p][f"l1_{h}"] for p in parents]);delta=best-base
   c[f"bestK_l1_{h}_difference"]=dict(mean=float(delta.mean()),ci95=ci(delta[idx].mean(1)))
  fg=parent_by_subset[(subset,tag)]["grip"];dg=parent_by_subset[(subset,"D")]["grip"]
  e=np.asarray([fg[p]["f1"]-dg[p]["f1"] for p in parents]);c["f1_10_difference"]=dict(mean=float(e.mean()),ci95=ci(e[idx].mean(1)))
  false=np.asarray([fg[p]["false"]-dg[p]["false"] for p in parents]);c["false_switch_10_difference"]=dict(mean=float(false.mean()),ci95=ci(false[idx].mean(1)))
  c["same_parent_count"]=len(parents);comparisons[tag]=c
 world_comp={}
 w_summary=summary["subsets"][subset]["world"]
 for tag in ("S1","S2","S4","S8"):
  for h in (25,50,100):
   for method in ("N1","N8"):
    fkey=f"{tag}|h{h}|{method}";dkey=f"D|h{h}|{method}"
    if fkey not in w_summary or dkey not in w_summary:continue
    def wp(key):
     rec=[r for r in wr if r["group"]==key.split("|")[0] and r["h"]==h and r["method"]==method]
     return parent_world(rec,subset)
    fp,dp=wp(fkey),wp(dkey);ps=sorted(set(fp)&set(dp));ii=boot_idx(len(ps));delta=np.asarray([fp[p]["skill"]-dp[p]["skill"] for p in ps])
    world_comp[f"{tag}|h{h}|{method}"]={"delta_skill_pp":float(delta.mean()),"ci95_pp":ci(delta[ii].mean(1)),"parents":len(ps)}
 summary["subsets"][subset]["flow_vs_D"]=comparisons
 summary["subsets"][subset]["world_flow_vs_D"]=world_comp
 summary["latency"]=raw["latency"];summary["peak_mib"]=raw["peak_mib"];summary["gpu_seconds"]=raw["gpu_seconds"];summary["brt"]=raw["brt"]
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({"status":"complete","subsets":{k:{"D_l1_8":v["action"]["D"]["l1_8"]["mean"],"S4_l1_8":v["action"]["S4"]["l1_8"]["mean"],"S4_diff":v["flow_vs_D"]["S4"]["single_l1_8_relative_diff_pct"],"S4_F1diff":v["flow_vs_D"]["S4"]["f1_10_difference"]} for k,v in summary["subsets"].items()},"latency":summary["latency"],"peak_mib":summary["peak_mib"],"gpu_seconds":summary["gpu_seconds"]},ensure_ascii=False))
