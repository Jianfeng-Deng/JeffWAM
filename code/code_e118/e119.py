import json,numpy as np,torch,collections
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");R=T/"results/e119_cond_20261117_01";z=np.load(T/"results/e118_pilot_20261117_02/data.npz",allow_pickle=True)
arm=z["arm"];hd=z["hdir"];vd=z["vdir"];am=z["amp"];gr=z["grip"];fold=z["fold"];par=z["parent"];N=len(arm)
HD={"hold":"","forward":"forward","backward":"backward","left":"left","right":"right","forward-left":"forward and left","forward-right":"forward and right","backward-left":"backward and left","backward-right":"backward and right"}
def tA(a,h,v,m,g):
    if h=="hold" and v=="hold":mv="holds its position"
    else:
        d=" and ".join(x for x in (HD[h],{"up":"up","down":"down","hold":""}[v]) if x);mv=f"moves {d} with {m} displacement"
    gp={"close":"closes its gripper","open":"opens its gripper","hold":"keeps its gripper unchanged"}[g]
    return f"The {a} arm {mv} and {gp}."
def tB(a,h,v,m,g):
    if h=="hold" and v=="hold":mv=f"Keep the {a} arm still"
    else:
        d=" and ".join(x for x in (HD[h],{"up":"upward","down":"downward","hold":""}[v]) if x);mv=f"Shift the {a} gripper {d} by a {m} amount"
    gp={"close":", then close the gripper.","open":", then open the gripper.","hold":", leaving the gripper as it is."}[g]
    return mv+gp
keys=sorted(set(zip(arm,hd,vd,am,gr)));ki={k:i for i,k in enumerate(keys)};kid=np.array([ki[k] for k in zip(arm,hd,vd,am,gr)])
from transformers import AutoTokenizer,SiglipModel
M="/mnt/workspace/dengjianfeng/models/siglip2";tok=AutoTokenizer.from_pretrained(M,local_files_only=True);sm=SiglipModel.from_pretrained(M,local_files_only=True).cuda().eval()
def enc(texts):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(texts),256):
            x=tok([t.lower() for t in texts[i:i+256]],padding="max_length",max_length=64,truncation=False,return_tensors="pt").to("cuda");e=sm.get_text_features(**x);e=getattr(e,"pooler_output",e);out.append(torch.nn.functional.normalize(e.float(),dim=-1).cpu())
    return torch.cat(out)
EA=enc([tA(*k) for k in keys]);EB=enc([tB(*k) for k in keys]);print("example A:",tA(*keys[100]),"| B:",tB(*keys[100]),"| cos(A,B) mean over keys",float((EA*EB).sum(1).mean()),flush=True)
del sm;torch.cuda.empty_cache()
coarse=np.array([f"{a}|{h}|{v}|{g}" for a,h,v,g in zip(arm,hd,vd,gr)]);moving=~((hd=="hold")&(vd=="hold")&(gr=="hold"))
dev="cuda";cur=z["cur"];dA=z["dA"];dI=z["dI"];armoh=np.stack([(arm=="left"),(arm=="right")],1).astype("f4")
ps=sorted(set(par));pi={p:i for i,p in enumerate(ps)};P=np.array([pi[p] for p in par])
EAg=EA.to(dev);EBg=EB.to(dev);kidg=torch.from_numpy(kid).to(dev)
def build(cond_text):
    class Net(torch.nn.Module):
        def __init__(s):
            super().__init__();s.tp=torch.nn.Linear(768,128) if cond_text else None;d=2306+(128 if cond_text else 0)
            s.f=torch.nn.Sequential(torch.nn.Linear(d,512),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(512,512),torch.nn.GELU(),torch.nn.Linear(512,64+2304))
        def forward(s,x,t=None):
            if s.tp is not None:x=torch.cat([x,s.tp(t)],1)
            return s.f(x)
    return Net().to(dev)
sse={c:{"dA":np.zeros(len(ps)),"dI":np.zeros(len(ps)),"dAm":np.zeros(len(ps)),"dIm":np.zeros(len(ps))} for c in ("NONE","A","B","SHUF")};sst={"dA":np.zeros(len(ps)),"dI":np.zeros(len(ps)),"dAm":np.zeros(len(ps)),"dIm":np.zeros(len(ps))}
ctl={"A":{"dA":[],"dI":[]},"B":{"dA":[],"dI":[]}};ctlp=[]
rng=np.random.default_rng(5);SEEDS=[90,91]
for k in range(5):
    tr=np.where(fold!=k)[0];te=np.where(fold==k)[0]
    mu=cur[tr].mean(0);sd=cur[tr].std(0)+1e-6;mA=dA[tr].mean(0);sA=dA[tr].std(0)+1e-6;mI=dI[tr].mean(0);sI=dI[tr].std(0)+1e-6
    X=torch.from_numpy(np.concatenate([(cur-mu)/sd,armoh],1).astype("f4")).to(dev);Y=torch.from_numpy(np.concatenate([(dA-mA)/sA,(dI-mI)/sI],1).astype("f4")).to(dev)
    preds={c:0 for c in ("NONE","A","B","SHUF")};swap={"A":0,"B":0};swj=None
    # swap partner: held-out sample with different coarse class
    ted=te.copy();partner=np.empty(len(te),int)
    for j,i in enumerate(te):
        while True:
            c=rng.choice(te)
            if coarse[c]!=coarse[i]:break
        partner[j]=c
    for seed in SEEDS:
        for cond in (False,True):
            torch.manual_seed(seed+k);net=build(cond);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4);r=np.random.default_rng(seed*11+k)
            for step in range(3000):
                idx=torch.from_numpy(r.choice(tr,512)).to(dev);o=net(X[idx],EAg[kidg[idx]] if cond else None)
                loss=torch.nn.functional.mse_loss(o[:,:64],Y[idx,:64])+torch.nn.functional.mse_loss(o[:,64:],Y[idx,64:]);opt.zero_grad();loss.backward();opt.step()
            net.eval()
            with torch.no_grad():
                tei=torch.from_numpy(te).to(dev);pj=torch.from_numpy(partner).to(dev)
                if not cond:preds["NONE"]=preds["NONE"]+net(X[tei]).cpu().numpy()/len(SEEDS)
                else:
                    preds["A"]=preds["A"]+net(X[tei],EAg[kidg[tei]]).cpu().numpy()/len(SEEDS);preds["B"]=preds["B"]+net(X[tei],EBg[kidg[tei]]).cpu().numpy()/len(SEEDS)
                    preds["SHUF"]=preds["SHUF"]+net(X[tei],EAg[kidg[pj]]).cpu().numpy()/len(SEEDS);swap["A"]=swap["A"]+preds["SHUF"]*0  # placeholder
                    swap.setdefault("Bs",0);swap["Bs"]=swap["Bs"]+net(X[tei],EBg[kidg[pj]]).cpu().numpy()/len(SEEDS)
    Yt=Y[torch.from_numpy(te).to(dev)].cpu().numpy();mv=moving[te]
    for c in ("NONE","A","B","SHUF"):
        e=(preds[c]-Yt)**2;np.add.at(sse[c]["dA"],P[te],e[:,:64].sum(1));np.add.at(sse[c]["dI"],P[te],e[:,64:].sum(1));np.add.at(sse[c]["dAm"],P[te],(e[:,:64].sum(1))*mv);np.add.at(sse[c]["dIm"],P[te],(e[:,64:].sum(1))*mv)
    np.add.at(sst["dA"],P[te],(Yt[:,:64]**2).sum(1));np.add.at(sst["dI"],P[te],(Yt[:,64:]**2).sum(1));np.add.at(sst["dAm"],P[te],(Yt[:,:64]**2).sum(1)*mv);np.add.at(sst["dIm"],P[te],(Yt[:,64:]**2).sum(1)*mv)
    # controllability: prototypes from training folds, coarse class mean target
    proto={};cl=coarse[tr]
    for c in set(cl):
        ii=tr[cl==c]
        if len(ii)>=10:proto[c]=Y[torch.from_numpy(ii).to(dev)].mean(0).cpu().numpy()
    def cos(a,b):return float(((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b)+1e-9))[0])
    for j,i in enumerate(te):
        c2=coarse[partner[j]]
        if not moving[i] or c2 not in proto:continue
        pr=proto[c2]
        for name,sw in (("A",preds["SHUF"][j:j+1]),("B",swap["Bs"][j:j+1])):
            base=preds["A"][j:j+1] if name=="A" else preds["B"][j:j+1]
            ctl[name]["dA"].append((P[i],float(cos(sw[:,:64],pr[:64])-cos(base[:,:64],pr[:64]))));ctl[name]["dI"].append((P[i],float(cos(sw[:,64:],pr[64:])-cos(base[:,64:],pr[64:]))))
    print("fold",k,"done",flush=True)
draws=np.random.default_rng(90).integers(len(ps),size=(2000,len(ps)));W=np.stack([np.bincount(d,minlength=len(ps)) for d in draws]).astype("f8");W=np.vstack([np.ones(len(ps)),W])
def r2(c,t):return 1-(W@sse[c][t])/(W@sst[t])
def q(a):return float(a[0]),[float(np.quantile(a[1:],.025)),float(np.quantile(a[1:],.975))]
out={}
for t in ("dA","dI","dAm","dIm"):
    base=r2("NONE",t)
    for c in ("NONE","A","B","SHUF"):out[f"R2|{t}|{c}"]=q(r2(c,t))
    for c in ("A","B","SHUF"):out[f"dR2|{t}|{c}-NONE"]=q(r2(c,t)-base)
    out[f"gainratio|{t}|B/A"]=q((r2("B",t)-base)/(r2("A",t)-base))
for name in ("A","B"):
    for t in ("dA","dI"):
        arr=np.array(ctl[name][t]);pp=np.zeros(len(ps));cn=np.zeros(len(ps));np.add.at(pp,arr[:,0].astype(int),arr[:,1]);np.add.at(cn,arr[:,0].astype(int),1)
        out[f"control|{name}|{t}"]=q((W@pp)/(W@cn))
json.dump(out,open(R/"summary.json","w"),indent=1)
for k,v in out.items():print("%-28s %.4f [%.4f,%.4f]"%(k,v[0],v[1][0],v[1][1]))
