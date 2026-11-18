import os,sys,json,hashlib,datetime
from pathlib import Path
import numpy as np,torch
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");FEATS=T/"results/e115p1_stage_feats"
for p in ("code_e110f","code_e106c","code_e102a"):sys.path.insert(0,str(T/p))
INIT=T/"results/e106c_f2_s90_20261115_02/ckpt_20000.pt"
def brt():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
_cache={}
def stage_feat(parent,frame,kind):
    if kind=="zero":return np.zeros(20,"f4")
    k=(parent,kind)
    if k not in _cache:_cache[k]=np.load(FEATS/(parent+".npz"))["ctrl" if kind=="ctrl" else "oracle"].astype("f4")
    a=_cache[k];return a[min(max(int(frame),0),len(a)-1)]
def widen(model):
    old=model.proprio_proj[0];new=torch.nn.Linear(184,old.out_features)
    with torch.no_grad():
        new.weight[:,:164]=old.weight;new.weight[:,164:]=0;new.bias.copy_(old.bias)
    model.proprio_proj[0]=new;return model
def add_stage(b,rows,kind):
    S=np.stack([stage_feat(r["parent_component"],r["anchor"],kind) for r in rows]);s=torch.from_numpy(S).to(b["proprio"].device,b["proprio"].dtype)
    b=dict(b);b["proprio"]=torch.cat([b["proprio"],s],1);return b
