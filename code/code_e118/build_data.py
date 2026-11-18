# E118 pilot data (Claude): FK motion text (arm-specific, h=50) with image-diff, action-diff, current features. train parents only.
import json,numpy as np,pyarrow.parquet as pq,pandas as pd,torch,os
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");OUT=T/"results/e118_pilot_20261117_02";F=T/"results/e102s_20261115_01"
EXCL="piper_4dc107335389f5711a0f65e0"
pf=json.loads((F/"folds.json").read_text())["parent_to_fold"]
df=pq.read_table(T/"data/e84_piper_align_v1/fk/fk_motion_labels.parquet").to_pandas()
d=df[(df.horizon_steps==50)&(df.eligible_for_e84)&(df.parent_component.isin(pf))&(df.canonical_id!=EXCL)].reset_index(drop=True)
print("rows",len(d),"parents",d.parent_component.nunique(),flush=True)
cur=[];dI=[];dA=[];keep=[];cache={}
def load(p):
    if p not in cache:cache.clear();cache[p]=(np.load(F/"features"/p/"b.npy",mmap_mode="r"),np.load(F/"features"/p/"state.npy",mmap_mode="r"))
    return cache[p]
for i,r in d.sort_values(["parent_component","anchor_source_row"]).iterrows():
    b,s=load(r.parent_component);a=int(r.anchor_source_row);n=min(len(b),len(s))
    if a+50>=n or s[a,32]<.5 or s[a+50,32]<.5:continue
    cur.append(np.asarray(b[a],"f4"));dI.append(np.asarray(b[a+50],"f4")-np.asarray(b[a],"f4"))
    dA.append(np.concatenate([np.asarray(s[a+25,:32],"f4")-np.asarray(s[a,:32],"f4"),np.asarray(s[a+50,:32],"f4")-np.asarray(s[a,:32],"f4")]));keep.append(i)
d=d.loc[keep].reset_index(drop=True)
print("kept",len(d),flush=True)
cls=(d.arm+"|"+d.horizontal_direction.astype(str)+"|"+d.vertical_direction.astype(str)+"|"+d.amplitude.astype(str)+"|"+d.gripper.astype(str)).values
uniq=sorted(set(d.text_en));ti={t:i for i,t in enumerate(uniq)}
from transformers import AutoTokenizer,SiglipModel
M="/mnt/workspace/dengjianfeng/models/siglip2";tok=AutoTokenizer.from_pretrained(M,local_files_only=True);m=SiglipModel.from_pretrained(M,local_files_only=True).cuda().eval()
emb=[]
with torch.inference_mode():
    for i in range(0,len(uniq),256):
        x=tok([t.lower() for t in uniq[i:i+256]],padding="max_length",max_length=64,truncation=False,return_tensors="pt").to("cuda");z=m.get_text_features(**x);z=getattr(z,"pooler_output",z);emb.append(torch.nn.functional.normalize(z.float(),dim=-1).cpu().numpy())
emb=np.concatenate(emb);print("text emb",emb.shape,flush=True)
np.savez_compressed(OUT/"data.npz",cur=np.stack(cur),dI=np.stack(dI),dA=np.stack(dA),parent=d.parent_component.values,anchor=d.anchor_source_row.values,arm=d.arm.values,cls=cls,
  text_idx=np.array([ti[t] for t in d.text_en]),text_emb=emb,uniq=np.array(uniq),n_frames=np.array([len(load(p)[0]) for p in d.parent_component.values]),
  hdir=d.horizontal_direction.astype(str).values,vdir=d.vertical_direction.astype(str).values,amp=d.amplitude.astype(str).values,grip=d.gripper.astype(str).values,
  fold=np.array([pf[p] for p in d.parent_component.values]))
print("classes",len(set(cls)));import collections;print(collections.Counter(cls).most_common(6))
