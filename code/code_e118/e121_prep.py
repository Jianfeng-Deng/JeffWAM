# E121 prep: verify that cached pooled features (b.npy) are mean-pooled last_hidden_state, then build the MaskCLIP-style bridge on them.
# The value/out projection is linear, so projecting the pooled token equals pooling the projected tokens; the residual MLP is applied to the pooled token (approximation).
import json,sys,numpy as np,torch,cv2
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");R=T/"results/e121_bridge_20261117_01";R.mkdir(exist_ok=True)
sys.path.insert(0,str(T/"code_e86"))
from e86.vision import preprocess
from transformers import SiglipModel
m=SiglipModel.from_pretrained("/mnt/workspace/dengjianfeng/models/siglip2",local_files_only=True).cuda().eval();head=m.vision_model.head;d=768
F=T/"results/e102s_20261115_01/features";pf=json.loads((T/"results/e102s_20261115_01/folds.json").read_text())["parent_to_fold"]
src={s["parent"]:s for s in json.loads((T/"results/e99b3_20261115_01/sources.json").read_text())}
def bridge(x):  # x: (n,768) pooled hidden -> bridged, L2-normalised
    a=head.attention;W=a.in_proj_weight;b=a.in_proj_bias
    with torch.inference_mode():
        v=x@W[2*d:].T+b[2*d:];o=a.out_proj(v);y=o+head.mlp(head.layernorm(o))
    return torch.nn.functional.normalize(y.float(),dim=-1)
# 1) verification against cached b.npy on a few frames
cos=[];ratio=[]
for p in sorted(pf)[:4]:
    n=len(np.load(F/p/"state.npy",mmap_mode="r"));fr=[200,400]
    for ci,cam in enumerate(("cam_high","cam_left_wrist","cam_right_wrist")):
        cap=cv2.VideoCapture(src[p]["videos"][cam]);cur=-1;got={}
        while len(got)<len(fr):
            ok=cap.grab();cur+=1
            if not ok:break
            if cur in fr:ok,bgr=cap.retrieve();got[cur]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        cap.release()
        for f,rgb in got.items():
            x,valid,_=preprocess(rgb);x=x[None].cuda()
            with torch.inference_mode():hs=m.vision_model(pixel_values=x).last_hidden_state[0]
            vm=torch.nn.functional.avg_pool2d(valid[None,None].cuda().float(),16,16).reshape(-1)
            mean=(hs*vm[:,None]).sum(0)/vm.sum().clamp_min(1e-6);ref=torch.from_numpy(np.asarray(np.load(F/p/"b.npy",mmap_mode="r")[f],"f4")[ci*768:(ci+1)*768]).cuda()
            cos.append(float(torch.nn.functional.cosine_similarity(mean,ref,dim=0)));ratio.append(float(mean.norm()/ref.norm()))
print("pooled-hidden vs cached b: cos mean %.4f min %.4f ; norm ratio mean %.3f"%(np.mean(cos),np.min(cos),np.mean(ratio)),flush=True)
# 2) bridged features for the E118 sample set
z=np.load(T/"results/e118_pilot_20261117_02/data.npz",allow_pickle=True);cur=torch.from_numpy(z["cur"]).cuda()
J=torch.cat([bridge(cur[:,i*768:(i+1)*768]) for i in range(3)],1).cpu().numpy()
np.save(R/"J_cur.npy",J);print("J_cur",J.shape,"mean cos between two random samples same cam %.3f"%float((J[:2000,:768]@J[2000:4000,:768].T).mean()),flush=True)
json.dump(dict(cos_mean=float(np.mean(cos)),cos_min=float(np.min(cos)),norm_ratio=float(np.mean(ratio))),open(R/"verify.json","w"))
