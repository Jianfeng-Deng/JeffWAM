# E120 (Claude): (A) joint-space Delta direction test on gripper open/close; (B) MaskCLIP-style per-patch localization.
import json,sys,os,numpy as np,torch,cv2,collections
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");R=T/"results/e120_jointspace_20261117_01";R.mkdir(exist_ok=True)
sys.path.insert(0,str(T/"code_e86"))
from e86.vision import preprocess
from transformers import AutoTokenizer,SiglipModel
M="/mnt/workspace/dengjianfeng/models/siglip2";tok=AutoTokenizer.from_pretrained(M,local_files_only=True);m=SiglipModel.from_pretrained(M,local_files_only=True).cuda().eval()
F=T/"results/e102s_20261115_01/features";pf=json.loads((T/"results/e102s_20261115_01/folds.json").read_text())["parent_to_fold"]
src={s["parent"]:s for s in json.loads((T/"results/e99b3_20261115_01/sources.json").read_text())}
def pooled(x):o=m.get_image_features(pixel_values=x);return getattr(o,"pooler_output",o)
def text(ts):
    with torch.inference_mode():
        i=tok([t.lower() for t in ts],padding="max_length",max_length=64,truncation=False,return_tensors="pt").to("cuda");e=m.get_text_features(**i);e=getattr(e,"pooler_output",e);return torch.nn.functional.normalize(e.float(),dim=-1)
def frames(parent,cam,idx):
    cap=cv2.VideoCapture(src[parent]["videos"][cam]);need=sorted(set(idx));out={};cur=-1;j=0
    while j<len(need):
        ok=cap.grab();cur+=1
        if not ok:break
        if cur==need[j]:
            ok,bgr=cap.retrieve();out[cur]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);j+=1
    cap.release();return out
def embed(rgbs):
    xs=torch.stack([preprocess(r)[0] for r in rgbs]).cuda();out=[]
    with torch.inference_mode():
        for i in range(0,len(xs),64):out.append(torch.nn.functional.normalize(pooled(xs[i:i+64]).float(),dim=-1))
    return torch.cat(out)
# ---------------- A: gripper Delta test ----------------
rng=np.random.default_rng(11);ev=[]
for p in sorted(pf):
    s=np.load(F/p/"state.npy",mmap_mode="r");n=len(s)
    for side,col,cam in (("left",6,"cam_left_wrist"),("right",13,"cam_right_wrist")):
        g=np.asarray(s[:,col]);ch=np.where(np.diff(g)!=0)[0]+1
        for i in ch:
            if i<35 or i+35>=n:continue
            pre,post=g[i-1],g[i]
            if (g[i-30:i]==pre).all() and (g[i:i+30]==post).all():ev.append(dict(parent=p,side=side,cam=cam,i=int(i),kind="close" if pre==1 else "open"))
        quiet=[t for t in range(45,n-45) if (g[t-45:t+46]==g[t]).all()]
        for t in rng.choice(quiet,min(6,len(quiet)),replace=False) if quiet else []:ev.append(dict(parent=p,side=side,cam=cam,i=int(t),kind="ctrl"))
print("events",collections.Counter(e["kind"] for e in ev),flush=True)
bycam={}
for e in ev:bycam.setdefault((e["parent"],e["cam"]),[]).append(e)
rows=[]
for (p,cam),es in bycam.items():
    idx=[e["i"]-30 for e in es]+[e["i"]+30 for e in es];fr=frames(p,cam,idx)
    keep=[e for e in es if e["i"]-30 in fr and e["i"]+30 in fr]
    if not keep:continue
    eb=embed([fr[e["i"]-30] for e in keep]);ea=embed([fr[e["i"]+30] for e in keep])
    for e,b,a in zip(keep,eb,ea):rows.append(dict(e,dv=(a-b).cpu()))
CL=["a closed gripper","the gripper is closed","a gripper closed around an object","a robot gripper with its fingers closed"];OP=["an open gripper","the gripper is open","a gripper opened wide","a robot gripper with its fingers apart"]
dT=text(CL).mean(0)-text(OP).mean(0);dT=dT.cpu();dTc=(text(["a red cube"])[0]-text(["a blue cube"])[0]).cpu()
cos=lambda a,b:float((a*b).sum()/(a.norm()*b.norm()+1e-9))
for r in rows:r["cos"]=cos(r["dv"],dT);r["cos_ctrl_text"]=cos(r["dv"],dTc)
ps=sorted({r["parent"] for r in rows});pi={p:i for i,p in enumerate(ps)};draws=np.random.default_rng(90).integers(len(ps),size=(2000,len(ps)))
def stat(kind,key="cos"):
    pp=np.zeros(len(ps));cn=np.zeros(len(ps))
    for r in rows:
        if r["kind"]==kind:pp[pi[r["parent"]]]+=r[key];cn[pi[r["parent"]]]+=1
    return pp,cn
def boot(f):
    pt=f(np.ones(len(ps)));bs=[f(np.bincount(d,minlength=len(ps)).astype(float)) for d in draws];return float(pt),[float(np.nanquantile(bs,.025)),float(np.nanquantile(bs,.975))]
res={"counts":dict(collections.Counter(r["kind"] for r in rows))}
for key in ("cos","cos_ctrl_text"):
    for kind in ("close","open","ctrl"):
        pp,cn=stat(kind,key);res[f"{key}|{kind}"]=boot(lambda w:(w@pp)/max(w@cn,1e-9))
        # pooled (event-level) mean as well
        res[f"{key}|{kind}|event_mean"]=float(np.mean([r[key] for r in rows if r["kind"]==kind]))
ppc,cnc=stat("close");ppo,cno=stat("open")
res["cos|close-open"]=boot(lambda w:(w@ppc)/max(w@cnc,1e-9)-(w@ppo)/max(w@cno,1e-9))
# AUROC close vs open by cos (event-level, parent-bootstrap by resampling events of sampled parents)
from scipy.stats import rankdata
def auroc(sel):
    y=np.array([r["kind"]=="close" for r in sel]);s=np.array([r["cos"] for r in sel]);n1=y.sum();n0=len(y)-n1
    return float((rankdata(s)[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1*n0 else float("nan")
sel=[r for r in rows if r["kind"] in("close","open")];res["auroc_close_vs_open"]=auroc(sel)
bs=[]
for d in draws[:500]:
    w=np.bincount(d,minlength=len(ps));ss=[r for r in sel for _ in range(w[pi[r["parent"]]])];bs.append(auroc(ss))
res["auroc_close_vs_open_ci"]=[float(np.nanquantile(bs,.025)),float(np.nanquantile(bs,.975))]
for cam in ("cam_left_wrist","cam_right_wrist"):
    s2=[r for r in sel if r["cam"]==cam];res[f"auroc|{cam}"]=auroc(s2)
res["cos_text_norms"]=dict(delta_T_norm=float(dT.norm()),delta_T_vs_ctrl_text=cos(dT,dTc))
res["dv_norm_mean"]={k:float(np.mean([r["dv"].norm() for r in rows if r["kind"]==k])) for k in ("close","open","ctrl")}
(R/"A_gripper_delta.json").write_text(json.dumps(res,indent=1));print(json.dumps(res,indent=1),flush=True)
# ---------------- B: MaskCLIP-style per-patch localization ----------------
head=m.vision_model.head;d=768
def patch_feats(x):
    with torch.inference_mode():
        out=m.vision_model(pixel_values=x);hs=out.last_hidden_state
        a=head.attention;W=a.in_proj_weight;b=a.in_proj_bias;v=hs@W[2*d:].T+b[2*d:];o=a.out_proj(v);y=o+head.mlp(head.layernorm(o))
    return hs,y
queries={"orange zip tie":"an orange zip tie","white table":"a white table","black gripper":"a black robot gripper","transparent container":"a transparent plastic container","blue sky (control)":"a blue sky"}
Q=text(list(queries.values()));rng=np.random.default_rng(3);samples=[]
pl=sorted(pf)
for cam in ("cam_high","cam_left_wrist","cam_right_wrist"):
    pick=rng.choice(pl,25,replace=False)
    for p in pick:
        n=len(np.load(F/p/"state.npy",mmap_mode="r"));f=int(rng.integers(30,n-30));fr=frames(p,cam,[f]);samples.append(dict(parent=p,cam=cam,f=f,rgb=fr[f]))
def hsv_patch_labels(x):
    rgb=((x*0.5+0.5).clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy();hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    orange=((hsv[...,0]>=5)&(hsv[...,0]<=25)&(hsv[...,1]>=130)&(hsv[...,2]>=110)).astype("f4")
    white=((hsv[...,1]<=40)&(hsv[...,2]>=170)).astype("f4")
    dark=(hsv[...,2]<=60).astype("f4")
    pool=lambda a:a.reshape(14,16,14,16).mean((1,3)).reshape(-1)
    return pool(orange),pool(white),pool(dark)
from scipy.stats import rankdata as rd
def auc1(score,lab):
    y=lab;n1=y.sum();n0=len(y)-n1
    if n1<3 or n0<3:return None
    return float((rd(score)[y].sum()-n1*(n1+1)/2)/(n1*n0))
resB={};acc=collections.defaultdict(list);viz=[]
for s in samples:
    x=preprocess(s["rgb"])[0][None].cuda();hs,y=patch_feats(x);yn=torch.nn.functional.normalize(y[0].float(),dim=-1);hn=torch.nn.functional.normalize(hs[0].float(),dim=-1)
    sim_mask=(yn@Q.T).cpu().numpy();sim_raw=(hn@Q.T).cpu().numpy();o,w,dk=hsv_patch_labels(x[0])
    for qi,(qname,_) in enumerate(queries.items()):
        lab={"orange zip tie":o>=.25,"white table":w>=.6,"black gripper":dk>=.5}.get(qname)
        if lab is None:continue
        for tag,sim in (("maskclip",sim_mask),("raw_hidden",sim_raw)):
            a=auc1(sim[:,qi],lab)
            if a is not None:acc[(qname,s["cam"],tag)].append(a)
    if len(viz)<6 and s["cam"]=="cam_high":viz.append((s,sim_mask,x[0].cpu()))
for (q,cam,tag),v in sorted(acc.items()):
    resB[f"{q}|{cam}|{tag}"]=dict(mean_auroc=float(np.mean(v)),n=len(v),ci=[float(np.quantile([np.mean(np.random.default_rng(i).choice(v,len(v))) for i in range(500)],.025)),float(np.quantile([np.mean(np.random.default_rng(i).choice(v,len(v))) for i in range(500)],.975))])
(R/"B_maskclip.json").write_text(json.dumps(resB,indent=1))
for k,v in resB.items():print("%-52s AUROC %.3f [%.3f,%.3f] n=%d"%(k,v["mean_auroc"],v["ci"][0],v["ci"][1],v["n"]))
# heatmap figure
import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
fig,ax=plt.subplots(len(viz),4,figsize=(12,3*len(viz)))
for r,(s,sim,x) in enumerate(viz):
    img=((x*0.5+0.5).clamp(0,1)).permute(1,2,0).numpy();ax[r,0].imshow(img);ax[r,0].set_title("cam_high");ax[r,0].axis("off")
    for c,qn in enumerate(["orange zip tie","black gripper","white table"]):
        qi=list(queries).index(qn);h=sim[:,qi].reshape(14,14);h=(h-h.min())/(h.max()-h.min()+1e-9);ax[r,c+1].imshow(img);ax[r,c+1].imshow(cv2.resize(h,(224,224)),alpha=.55,cmap="jet");ax[r,c+1].set_title(qn);ax[r,c+1].axis("off")
plt.tight_layout();plt.savefig(R/"B_heatmaps.png",dpi=70);print("saved fig")
