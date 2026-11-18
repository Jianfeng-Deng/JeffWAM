# E115-P1 stage features: per-frame soft posterior (10) + dwell-constrained hard one-hot (10) from the E102-S H0 controller.
# train parents: five-fold OOF (no leakage); dev parents: frozen full controller. Oracle: model-retrospective one-hot.
import os,sys,json
sys.path[:0]=["/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e102s","/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e86"]
from tune import *
from scipy.special import logsumexp as lse
OUT=T/"results/e115p1_stage_feats";OUT.mkdir(exist_ok=True)
lock=json.loads((R/"CONTROLLER_LOCK.json").read_text());cfg=lock["selected"]["H0__fixed_S01"]["config"];temp,m=cfg["temp"],cfg["m"]
def soft_posterior(e,aux,temp):
    n=len(e);la=np.full((10,10),-np.inf)
    for i in range(10):
        for j in range(10):
            if aux["transition"][i,j]>0:la[i,j]=np.log(aux["transition"][i,j])
    f=np.full(10,-np.inf);f[0]=0.;out=np.zeros((n,10),"f4")
    for t in range(n):
        ff=np.array([lse(f+la[:,j]) for j in range(10)])+e[t]*temp
        f=ff-lse(ff);out[t]=np.exp(f)
    return out
def onehot(y):
    z=np.zeros((len(y),10),"f4");ok=y>=0;z[np.flatnonzero(ok),y[ok]]=1;return z
aux_fold={k:build_aux("H0",k) for k in range(5)}
with (R/"final_aux.pkl").open("rb") as f:aux_final=pickle.load(f)["H0"]
for s in train+dev:
    p=s["parent"]
    if s["pool"]=="train":
        a=aux_fold[fold[p]];prob=np.load(R/(p+"_H0_oof.npz"))["prob"]
    else:
        a=aux_final;prob=np.load(R/(p+"_dev_frozen_scores.npz"))["H0_prob"]
    e=emission(prob,a);soft=soft_posterior(e,a,temp);hard=onehot(hmm(e,a,temp,m,"fixed_S01"));orc=onehot(truths[p])
    np.savez_compressed(OUT/(p+".npz"),ctrl=np.concatenate([soft,hard],1).astype("f2"),oracle=np.concatenate([orc,orc],1).astype("f2"))
    print(p,len(e),flush=True)
