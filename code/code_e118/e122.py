# E122: history-only dropout probe. Does dropping history (keeping the full current frame) force use of the language condition?
import json,numpy as np,torch,collections,os
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");R=T/"results/e122_history_20261118_01";R.mkdir(exist_ok=True)
z=np.load(T/"results/e118_pilot_20261117_02/data.npz",allow_pickle=True);F=T/"results/e102s_20261115_01/features"
arm=z["arm"];hd=z["hdir"];vd=z["vdir"];am=z["amp"];gr=z["grip"];fold=z["fold"];par=z["parent"];anc=z["anchor"].astype(int);N=len(arm)
HD={"hold":"","forward":"forward","backward":"backward","left":"left","right":"right","forward-left":"forward and left","forward-right":"forward and right","backward-left":"backward and left","backward-right":"backward and right"}
def tA(a,h,v,m,g):
    if h=="hold" and v=="hold":mv="holds its position"
    else:
        d=" and ".join(x for x in (HD[h],{"up":"up","down":"down","hold":""}[v]) if x);mv=f"moves {d} with {m} displacement"
    gp={"close":"closes its gripper","open":"opens its gripper","hold":"keeps its gripper unchanged"}[g]
    return f"The {a} arm {mv} and {gp}."
keys=sorted(set(zip(arm,hd,vd,am,gr)));ki={k:i for i,k in enumerate(keys)};kid=np.array([ki[k] for k in zip(arm,hd,vd,am,gr)])
from transformers import AutoTokenizer,SiglipModel
M="/mnt/workspace/dengjianfeng/models/siglip2";tok=AutoTokenizer.from_pretrained(M,local_files_only=True);sm=SiglipModel.from_pretrained(M,local_files_only=True).cuda().eval()
out=[]
with torch.inference_mode():
    texts=[tA(*k).lower() for k in keys]
    for i in range(0,len(texts),256):
        x=tok(texts[i:i+256],padding="max_length",max_length=64,truncation=False,return_tensors="pt").to("cuda");e=sm.get_text_features(**x);e=getattr(e,"pooler_output",e);out.append(torch.nn.functional.normalize(e.float(),dim=-1).cpu())
EA=torch.cat(out);EAs=(EA-EA.mean(0,keepdim=True))/(EA.std(0,keepdim=True)+1e-6);del sm;torch.cuda.empty_cache()
dev="cuda";EAg=EAs.to(dev);kidg=torch.from_numpy(kid).to(dev)
# ---- history features (visual diff to 12 frames earlier, proprio diffs to 6/12/24 frames earlier)
cur=z["cur"];dA=z["dA"];hv=np.zeros_like(cur);hp=np.zeros((N,96),"f4")
for p in sorted(set(par)):
    ii=np.where(par==p)[0];b=np.load(F/p/"b.npy",mmap_mode="r");s=np.load(F/p/"state.npy",mmap_mode="r");a=anc[ii]
    hv[ii]=cur[ii]-np.asarray(b[np.maximum(a-12,0)],"f4")
    hp[ii]=np.concatenate([np.asarray(s[a,:32],"f4")-np.asarray(s[np.maximum(a-k,0),:32],"f4") for k in (6,12,24)],1)
print("history built",flush=True)
armoh=np.stack([(arm=="left"),(arm=="right")],1).astype("f4")
def amask(a):
    m=np.zeros(64,bool);pos=[14,15,16] if a=="left" else [23,24,25];g=6 if a=="left" else 13
    for off in (0,32):m[[off+p for p in pos]+[off+g]]=True
    return m
AM=np.stack([amask(a) for a in arm]);POSM=np.zeros_like(AM)
for i,a in enumerate(arm):POSM[i,[32+p for p in ([14,15,16] if a=="left" else [23,24,25])]]=True
coarse=np.array([f"{a}|{h}|{v}|{g}" for a,h,v,g in zip(arm,hd,vd,gr)]);moving=~((hd=="hold")&(vd=="hold")&(gr=="hold"))
ps=sorted(set(par));pi={p:i for i,p in enumerate(ps)};P=np.array([pi[p] for p in par])
class Net(torch.nn.Module):
    def __init__(s,dtext):
        super().__init__();s.tp=torch.nn.Linear(dtext,128) if dtext else None;d=2304+2+2400+1+(128 if dtext else 0)
        s.f=torch.nn.Sequential(torch.nn.Linear(d,512),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(512,512),torch.nn.GELU(),torch.nn.Linear(512,64))
    def forward(s,x,t=None):
        if s.tp is not None:x=torch.cat([x,s.tp(t)],1)
        return s.f(x)
SEEDS=[90,91];MODES=("none","keep","drop")
def accum():return {"sse":np.zeros(len(ps)),"sst":np.zeros(len(ps)),"sg":np.zeros(len(ps)),"sgn":np.zeros(len(ps)),"ctl":[]}
acc={(m,ev,c):accum() for m in MODES for ev in ("with","without") for c in ("NONE","AS","AS_SHUF")}
rng=np.random.default_rng(5)
for k in range(5):
    tr=np.where(fold!=k)[0];te=np.where(fold==k)[0]
    mu=cur[tr].mean(0);sd=cur[tr].std(0)+1e-6;mh=hv[tr].mean(0);sh=hv[tr].std(0)+1e-6;mp=hp[tr].mean(0);sp=hp[tr].std(0)+1e-6;mA=dA[tr].mean(0);sA=dA[tr].std(0)+1e-6
    Xc=torch.from_numpy(((cur-mu)/sd).astype("f4")).to(dev);Xa=torch.from_numpy(armoh).to(dev)
    Xh=torch.from_numpy(np.concatenate([(hv-mh)/sh,(hp-mp)/sp],1).astype("f4")).to(dev);Y=torch.from_numpy(((dA-mA)/sA).astype("f4")).to(dev)
    def make(idx,hist_on):
        h=Xh[idx]*hist_on[:,None];return torch.cat([Xc[idx],Xa[idx],h,hist_on[:,None]],1)
    partner=np.empty(len(te),int)
    for j,i in enumerate(te):
        while True:
            c=rng.choice(te)
            if coarse[c]!=coarse[i]:break
        partner[j]=c
    tei=torch.from_numpy(te).to(dev);pj=torch.from_numpy(partner).to(dev);ones=torch.ones(len(te),device=dev);zeros=torch.zeros(len(te),device=dev)
    for mode in MODES:
        preds={(ev,c):0 for ev in ("with","without") for c in ("NONE","AS","AS_SHUF")}
        for seed in SEEDS:
            for cond in (False,True):
                torch.manual_seed(seed+k);net=Net(768 if cond else None).to(dev);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4);r=np.random.default_rng(seed*11+k)
                for step in range(3000):
                    ix=torch.from_numpy(r.choice(tr,512)).to(dev)
                    if mode=="none":on=torch.zeros(512,device=dev)
                    elif mode=="keep":on=torch.ones(512,device=dev)
                    else:on=(torch.rand(512,device=dev)>=.7).float()
                    o=net(make(ix,on),EAg[kidg[ix]] if cond else None);loss=torch.nn.functional.mse_loss(o,Y[ix]);opt.zero_grad();loss.backward();opt.step()
                net.eval();f=1/len(SEEDS)
                with torch.no_grad():
                    for ev,on in (("with",ones if mode!="none" else zeros),("without",zeros)):
                        xin=make(tei,on)
                        if not cond:preds[(ev,"NONE")]=preds[(ev,"NONE")]+net(xin).cpu().numpy()*f
                        else:
                            preds[(ev,"AS")]=preds[(ev,"AS")]+net(xin,EAg[kidg[tei]]).cpu().numpy()*f;preds[(ev,"AS_SHUF")]=preds[(ev,"AS_SHUF")]+net(xin,EAg[kidg[pj]]).cpu().numpy()*f
        Yt=Y[tei].cpu().numpy();amk=AM[te];pmk=POSM[te];mv=moving[te]
        proto={};cl=coarse[tr];Ytr=Y.cpu().numpy()
        for c in set(cl):
            ii=tr[cl==c]
            if len(ii)>=10:proto[c]=Ytr[ii].mean(0)
        for (ev,c),pr in preds.items():
            a=acc[(mode,ev,c)];e=((pr-Yt)**2*amk).sum(1);np.add.at(a["sse"],P[te],e);np.add.at(a["sst"],P[te],(Yt**2*amk).sum(1))
            sg=((np.sign(pr)==np.sign(Yt))&pmk);np.add.at(a["sg"],P[te],sg.sum(1)*mv);np.add.at(a["sgn"],P[te],pmk.sum(1)*mv)
        for ev in ("with","without"):
            for j,i in enumerate(te):
                c2=coarse[partner[j]]
                if not moving[i] or c2 not in proto:continue
                m_=AM[i];pr_=proto[c2][m_];sw=preds[(ev,"AS_SHUF")][j][m_];bs=preds[(ev,"AS")][j][m_]
                cs=lambda a_,b_:float((a_*b_).sum()/(np.linalg.norm(a_)*np.linalg.norm(b_)+1e-9))
                acc[(mode,ev,"AS")]["ctl"].append((P[i],cs(sw,pr_)-cs(bs,pr_)))
    print("fold",k,"done",flush=True)
draws=np.random.default_rng(90).integers(len(ps),size=(2000,len(ps)));W=np.stack([np.bincount(d,minlength=len(ps)) for d in draws]).astype("f8");W=np.vstack([np.ones(len(ps)),W])
def q(a):return float(a[0]),[float(np.quantile(a[1:],.025)),float(np.quantile(a[1:],.975))]
def r2(m,ev,c):a=acc[(m,ev,c)];return 1-(W@a["sse"])/(W@a["sst"])
def sgn(m,ev,c):a=acc[(m,ev,c)];return (W@a["sg"])/(W@a["sgn"])
res={}
for m in MODES:
    for ev in ("with","without"):
        if m=="none" and ev=="with":continue
        base=r2(m,ev,"NONE")
        res[f"{m}|{ev}|R2_NONE"]=q(base);res[f"{m}|{ev}|R2_AS"]=q(r2(m,ev,"AS"));res[f"{m}|{ev}|gain"]=q(r2(m,ev,"AS")-base)
        res[f"{m}|{ev}|shuf_gain"]=q(r2(m,ev,"AS_SHUF")-base);res[f"{m}|{ev}|sign_NONE"]=q(sgn(m,ev,"NONE"));res[f"{m}|{ev}|sign_AS"]=q(sgn(m,ev,"AS"))
        arr=np.array(acc[(m,ev,"AS")]["ctl"]);pp=np.zeros(len(ps));cn=np.zeros(len(ps));np.add.at(pp,arr[:,0].astype(int),arr[:,1]);np.add.at(cn,arr[:,0].astype(int),1);res[f"{m}|{ev}|control"]=q((W@pp)/(W@cn))
json.dump(res,open(R/"summary.json","w"),indent=1)
for k_,v in res.items():print("%-28s %.4f [%.4f,%.4f]"%(k_,v[0],v[1][0],v[1][1]),flush=True)
