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
from scipy.stats import rankdata
# ---------------- B: MaskCLIP-style per-patch localization ----------------
head=m.vision_model.head;d=768
def patch_feats(x):
    with torch.inference_mode():
        out=m.vision_model(pixel_values=x);hs=out.last_hidden_state
        a=head.attention;W=a.in_proj_weight;b=a.in_proj_bias;v=hs@W[2*d:].T+b[2*d:];o=a.out_proj(v);y=o+head.mlp(head.layernorm(o))
    return hs,y
queries={"orange zip tie":"an orange zip tie","white table":"a white table","black gripper":"a black robot gripper","transparent container":"a transparent plastic container","blue sky (control)":"a blue sky"}
Q=text(list(queries.values()));rng=np.random.default_rng(3);samples=[]
pl=[p for p in sorted(pf) if len(np.load(F/p/'state.npy',mmap_mode='r'))>200]
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
