import os,sys,json,glob
os.environ["ARM"]="CA";os.environ["PYTHONPATH"]="code_e102s2:code_e86"
sys.path[:0]=["code_e102s2","code_e86"]
from s2 import *
H=Path("results/e114h_human_stage_20261116/boundaries")
hum={}
for f in sorted(H.glob("*.json")):
    a=json.loads(f.read_text());n=a["n_frames"];y=np.full(n,-1,int);b=sorted(a["boundaries"],key=lambda x:x["frame"])
    for i,m in enumerate(b):
        e=b[i+1]["frame"] if i+1<len(b) else n;y[m["frame"]:e]=SI.get(m["stage"],-1)
    hum[a["parent"]]=y
ps=sorted(hum);print("human parents",len(ps),"frames",sum(len(hum[p]) for p in ps))
# 1) human vs model retrospective truth (dev)
def agree(p):
    t=truths[p];h=hum[p];m=(t>=0)&(h>=0);return (t[m]==h[m]).mean(),(t>=0).mean(),(h>=0).mean()
for p in ps:print(p[3:11],"frame agreement %.3f"%agree(p)[0])
rec={"human_vs_model":[],"recall":{}}
for tol in(15,30):
    pp=[];mt=0;nh=0;nm=0;errs=[]
    for p in ps:
        hb=boundaries(hum[p]);mb=boundaries(truths[p]);pr=match(mb,hb,tol);mt+=len(pr);nh+=len(hb);nm+=len(mb);errs+=[mb[i][0]-hb[j][0] for i,j in pr]
    print("tol",tol,"human boundaries",nh,"model-retro boundaries",nm,"matched",mt,"recall of human boundaries by model %.3f"%(mt/nh),"precision %.3f"%(mt/nm),"median signed err",float(np.median(errs)) if errs else None)
# 2) frozen predictions vs human truth (same parents)
roots={"R(E102-S)":Path("results/e102s_20261115_01"),"CF":Path("results/e102s2_cf_20261116_01"),"CS":Path("results/e102s2_cs_20261116_01")}
for name,r in roots.items():
    for v in ["H0__fixed_S01","H1__fixed_S01","E101_S2"]:
        recs_h=[];recs_m=[]
        for p in ps:
            z=np.load(r/(p+"_dev_frozen_sequences.npz"))
            if v not in z.files:continue
            recs_h.append(record(z[v],hum[p]));recs_m.append(record(z[v],truths[p]))
        if not recs_h:continue
        ah=aggregate(recs_h);am=aggregate(recs_m)
        print("%-10s %-14s vs HUMAN recall30 %.3f f1_30 %.3f prec30 %.3f acc %.3f | vs model-retro recall30 %.3f f1_30 %.3f acc %.3f"%(name,v,ah["recall30"],ah["f1_30"],ah["precision30"],ah["accuracy"],am["recall30"],am["f1_30"],am["accuracy"]))
