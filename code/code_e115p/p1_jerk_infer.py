import os,sys,json,time
from common import *
from e99w2_model import WorldModel,JOINTS
from e102x_setup import datasets
from e102a_contract import contract,onset_labels
from e110f_core import infer_d
from e110f_eval import float_batch,transitions,match
from e86.data import move
OUT=T/"results/e115p1_jerk_20261118_01";OUT.mkdir()
torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
_,_,tr,dev=datasets();ent=json.loads((T/"results/e106c_control_20261115_02/PRETRAIN_LOCK.json").read_text())["entries"]
avail={p.stem for p in FEATS.glob("pc_*.npz")};E=[e for e in ent if e["parent"] in avail]
print("eval windows with stage features",len(E),"parents",len({e["parent"] for e in E}),flush=True)
ck={"D":T/"results/e110f_d_20261116_05/ckpt_04000.pt","S0":T/"results/e115p1_s0_20261117_01/ckpt_04000.pt","S1":T/"results/e115p1_s1_20261117_01/ckpt_04000.pt","S2":T/"results/e115p1_s2_20261117_01/ckpt_04000.pt"}
KIND={"D":None,"S0":"zero","S1":"ctrl","S2":"oracle"};c=contract();rows=[];onset={};PR=[]
for arm,path in ck.items():
    if not path.exists():print("missing",arm);continue
    cp=torch.load(path,map_location="cpu",weights_only=False);m=WorldModel(tr.norm)
    if KIND[arm]:m=widen(m)
    m.load_state_dict(cp["model"],strict=True);m=m.cuda().eval()
    with torch.inference_mode():
        for s in range(0,len(E),4):
            es=E[s:s+4];ids=[e["index"] for e in es];b=move(float_batch(dev.batch(ids,(10,),"short")),"cuda")
            if KIND[arm]:b=add_stage(b,[dict(parent_component=e["parent"],anchor=e["anchor"]) for e in es],KIND[arm])
            truth=b["action"];q=b["q_current"];pred=infer_d(m,b,4,90);ji=torch.as_tensor(JOINTS,device="cuda")
            for k,e in enumerate(es):
                l1=lambda x,h:float((x[k,:h,ji]-truth[k,:h,ji]).abs().mean())
                cp8=float((q[k,None,ji]-truth[k,:8,ji]).abs().mean());cp50=float((q[k,None,ji]-truth[k,:50,ji]).abs().mean())
                ps=(pred[k,:,torch.as_tensor([6,13],device="cuda")].cpu().numpy()>np.asarray(c["threshold_m"])).astype("i1");ts=b["grip_label"][k].cpu().numpy().astype("i1");cur=(b["grip_current"][k].cpu().numpy()>.5).astype("i1")
                z,qq,pose,good=dev.states.get(dev,e["cid"])
                if e["cid"] not in onset:onset[e["cid"]]=onset_labels(qq,z["continuous_segment"],good,c)[0]
                init=onset[e["cid"]][e["anchor"]];ev={str(t):[0,0,0] for t in (5,10,20)}
                for side in range(2):
                    pe=transitions(ps[:,side],cur[side]);te=transitions(ts[:,side],init[side])
                    for t in (5,10,20):ev[str(t)][0]+=len(match(pe,te,t));ev[str(t)][1]+=len(pe);ev[str(t)][2]+=len(te)
                PR.append((arm,e["index"],pred[k].float().cpu().numpy(),truth[k].float().cpu().numpy()));rows.append(dict(arm=arm,parent=e["parent"],index=e["index"],l1_8=l1(pred,8),l1_50=l1(pred,50),copy8=cp8,copy50=cp50,acc=float((ps==ts).mean()),ev=ev))
    print("done",arm,flush=True)

import pickle;pickle.dump(PR,open(OUT/"preds.pkl","wb"));print("saved",len(PR),flush=True)
