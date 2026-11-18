import json,numpy as np,torch,collections,sys
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");R=T/"results/e118_pilot_20261117_02"
z=np.load(R/"data.npz",allow_pickle=True)
par=z["parent"];fold=z["fold"];arm=z["arm"];N=len(par);emb=z["text_emb"];tix=z["text_idx"]
coarse=np.array([f"{a}|{h}|{v}|{g}" for a,h,v,g in zip(arm,z["hdir"],z["vdir"],z["grip"])]);fine=z["cls"]
pos=np.stack([z["anchor"]/z["n_frames"],z["anchor"]/2000.,z["n_frames"]/2000.],1).astype("f4")
armoh=(arm=="left").astype("f4")[:,None];armoh=np.concatenate([armoh,1-armoh],1)
FEAT={"dI":z["dI"],"dA":z["dA"],"dIdA":np.concatenate([z["dI"],z["dA"]],1),"cur":z["cur"],"pos":pos,"cur+dI":np.concatenate([z["cur"],z["dI"]],1)}
dev="cuda";VARS=list(FEAT);SEEDS=[90,91,92]
def run(variant,seed,labels):
    out=np.zeros((N,),bool);top5=np.zeros((N,),bool);rr=np.zeros(N,"f4")
    X=FEAT[variant]
    for k in range(5):
        tr=np.where(fold!=k)[0];te=np.where(fold==k)[0]
        mu=X[tr].mean(0);sd=X[tr].std(0)+1e-6;Xs=np.concatenate([(X-mu)/sd,armoh],1).astype("f4")
        torch.manual_seed(seed+k)
        enc=torch.nn.Sequential(torch.nn.Linear(Xs.shape[1],512),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(512,128)).to(dev)
        tp=torch.nn.Linear(768,128).to(dev);lt=torch.nn.Parameter(torch.tensor(2.3,device=dev))
        opt=torch.optim.AdamW(list(enc.parameters())+list(tp.parameters())+[lt],lr=1e-3,weight_decay=1e-4)
        cl=labels[tr];cnt=collections.Counter(cl);w=np.array([1/np.sqrt(cnt[c]) for c in cl]);w/=w.sum();rng=np.random.default_rng(seed*7+k)
        Xt=torch.from_numpy(Xs).to(dev);E=torch.from_numpy(emb).to(dev);cid={c:i for i,c in enumerate(sorted(set(labels)))};lab=torch.tensor([cid[c] for c in labels],device=dev)
        for step in range(2000):
            idx=torch.from_numpy(rng.choice(tr,512,p=w)).to(dev)
            q=torch.nn.functional.normalize(enc(Xt[idx]),dim=-1);t=torch.nn.functional.normalize(tp(E[torch.from_numpy(tix).to(dev)[idx]]),dim=-1)
            logits=q@t.T*lt.exp().clamp(max=100);same=(lab[idx][:,None]==lab[idx][None])&~torch.eye(512,dtype=torch.bool,device=dev)
            logits=logits.masked_fill(same,-1e4);y=torch.arange(512,device=dev)
            loss=.5*(torch.nn.functional.cross_entropy(logits,y)+torch.nn.functional.cross_entropy(logits.T,y))
            opt.zero_grad();loss.backward();opt.step()
        # candidates: one canonical sentence per class (most frequent in training folds)
        rep={};
        for c in sorted(set(cl)):
            ii=tr[cl==c];rep[c]=collections.Counter(tix[ii]).most_common(1)[0][0]
        cls=sorted(rep);C=torch.nn.functional.normalize(tp(E[torch.tensor([rep[c] for c in cls],device=dev)]),dim=-1)
        with torch.no_grad():
            Q=torch.nn.functional.normalize(enc.eval()(Xt[torch.from_numpy(te).to(dev)]),dim=-1);S=(Q@C.T).cpu().numpy()
        truth=np.array([cls.index(c) if c in rep else -1 for c in labels[te]])
        order=np.argsort(-S,1)
        for j,i in enumerate(te):
            if truth[j]<0:rr[i]=np.nan;continue
            rank=int(np.where(order[j]==truth[j])[0][0]);out[i]=rank==0;top5[i]=rank<5;rr[i]=1/(rank+1)
    return out,top5,rr
res={}
for gran,labels in (("coarse",coarse),("fine",fine)):
    for v in VARS:
        o=np.zeros(N);t5=np.zeros(N);r=np.zeros(N)
        for s in SEEDS:
            a,b,c=run(v,s,labels);o+=a;t5+=b;r+=np.nan_to_num(c)
        res[gran,v]=(o/len(SEEDS),t5/len(SEEDS),r/len(SEEDS));print(gran,v,"top1 %.3f top5 %.3f mrr %.3f"%(o.mean()/len(SEEDS),t5.mean()/len(SEEDS),r.mean()/len(SEEDS)),flush=True)
    # shuffled (dI queries paired with wrong text within fold) and frequency prior
    rng=np.random.default_rng(5);o=np.zeros(N);t5=np.zeros(N)
    for k in range(5):
        tr=labels[fold!=k];te=np.where(fold==k)[0];cnt=collections.Counter(tr);rank=[c for c,_ in cnt.most_common()]
        for i in te:
            c=labels[i];r_=rank.index(c) if c in rank else 10**6;o[i]=r_==0;t5[i]=r_<5
        # shuffled: rank of the true class under random ordering
        ncls=len(rank)
        for i in te:o[i]=o[i]  # keep prior for prior baseline
    res[gran,"prior"]=(o,t5,np.zeros(N))
    o2=np.zeros(N);t52=np.zeros(N)
    for k in range(5):
        te=np.where(fold==k)[0];ncls=len(set(labels[fold!=k]))
        for i in te:
            if labels[i] in set(labels[fold!=k]):r_=rng.integers(ncls);o2[i]=r_==0;t52[i]=r_<5
    res[gran,"shuffled"]=(o2,t52,np.zeros(N))
np.savez_compressed(R/"oof.npz",**{f"{g}|{v}|{m}":x[i] for (g,v),x in res.items() for i,m in enumerate(("top1","top5","mrr"))},coarse=coarse,fine=fine,parent=par,fold=fold)
print("saved",flush=True)
