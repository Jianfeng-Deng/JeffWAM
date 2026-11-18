import os,sys,json
os.environ["ARM"]="CA";sys.path[:0]=["code_e102s2","code_e86"]
from s2 import *
H=Path("results/e114h_human_stage_20261116/boundaries");hum={}
for f in sorted(H.glob("*.json")):
    a=json.loads(f.read_text());n=a["n_frames"];y=np.full(n,-1,int);b=sorted(a["boundaries"],key=lambda x:x["frame"])
    for i,m in enumerate(b):
        e=b[i+1]["frame"] if i+1<len(b) else n;y[m["frame"]:e]=SI.get(m["stage"],-1)
    hum[a["parent"]]=y
ps=sorted(hum);TOL=[15,30,60,90,150,300]
def rate(pred,true,tol):
    mt=0;nt=0;npd=0
    for p in ps:
        pb=boundaries(pred[p]);tb=boundaries(true[p]);mt+=len(match(pb,tb,tol));nt+=len(tb);npd+=len(pb)
    return mt/nt,mt/npd,2*mt/(nt+npd)
print("== 模型回顾式边界 vs 人工边界（带类型的一一匹配）：召回=人工边界被模型找到的比例")
for t in TOL:
    r,p,f=rate(truths,hum,t);print("tol %3d帧(%.1fs) recall %.3f precision %.3f f1 %.3f"%(t,t/30,r,p,f))
# offsets of matched at 300
errs=[]
for p in ps:
    pb=boundaries(truths[p]);tb=boundaries(hum[p]);errs+=[pb[i][0]-tb[j][0] for i,j in match(pb,tb,300)]
e=np.array(errs);print("matched@300",len(e),"abs offset median %.0f p75 %.0f p90 %.0f ; signed median %.0f (正=模型更晚)"%(np.median(abs(e)),np.quantile(abs(e),.75),np.quantile(abs(e),.9),np.median(e)))
# stage sequence agreement (order of stages, ignoring timing)
import difflib
def seq(y):
    return [int(s) for _,a,s in boundaries(y)]
for p in ps:
    a=seq(truths[p]);b=seq(hum[p]);r=difflib.SequenceMatcher(None,a,b).ratio();print(p[3:11],"stage-order similarity %.2f"%r,"model %d human %d boundaries"%(len(a),len(b)))
roots={"R(E102-S)":Path("results/e102s_20261115_01"),"CF":Path("results/e102s2_cf_20261116_01"),"CS":Path("results/e102s2_cs_20261116_01")}
print("== 已冻结预测 vs 人工边界（H0 固定S01）：召回/precision@各容差")
for name,r in roots.items():
    pred={p:np.load(r/(p+"_dev_frozen_sequences.npz"))["H0__fixed_S01"] for p in ps}
    print(name," ".join("%d帧:R%.2f/P%.2f"%(t,*rate(pred,hum,t)[:2]) for t in TOL))
pred={p:np.load(Path("results/e102s_20261115_01")/(p+"_dev_frozen_sequences.npz"))["E101_S2"] for p in ps}
print("E101_S2"," ".join("%d帧:R%.2f/P%.2f"%(t,*rate(pred,hum,t)[:2]) for t in TOL))
