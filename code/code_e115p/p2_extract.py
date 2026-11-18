# E115-P2 step A (GPU): pooled (cam-major 3x8 mean) real / copy / predicted future tokens at each window and horizon.
from common import *
from e99w2_model import WorldModel,trajectory
from e102x_setup import datasets
from e110f_core import infer_d
from e110f_eval import float_batch
from e86.data import move
OUT=T/"results/e115p2_pooled_20261117_02";OUT.mkdir()
torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
_,_,tr,dev=datasets();ent=json.loads((T/"results/e106c_control_20261115_02/PRETRAIN_LOCK.json").read_text())["entries"]
avail={p.stem for p in FEATS.glob("pc_*.npz")};E=[e for e in ent if e["parent"] in avail]
cp=torch.load(INIT,map_location="cpu",weights_only=False);m=WorldModel(tr.norm);m.load_state_dict(cp["model"],strict=True);m=m.cuda().eval()
def pool(x):return x.reshape(len(x),3,x.shape[1]//3,768).mean(2).reshape(len(x),2304).cpu().numpy().astype("f4")
rows=[];arr={k:[] for k in ("real","cur","gt","no","n1","sh")}
with torch.inference_mode():
    for h in (10,25,50,100):
        Eh=[e for e in E if h in e["horizons"]];Eh=[Eh[i] for i in np.random.default_rng(5).permutation(len(Eh))]
        for s in range(0,len(Eh),4):
            es=Eh[s:s+4];b=move(float_batch(dev.batch([e["index"] for e in es],(h,),"short")),"cuda");assert b["world_valid"].all()
            scale=b["world_scale"];cur=b["world_current"]
            def to_raw(P):return cur+P*scale
            gt=m.forward_mean(b)["mean"];no=m.forward_mean(dict(b,robot=torch.zeros_like(b["robot"]),robot_valid=torch.zeros_like(b["robot_valid"])))["mean"]
            sh=m.forward_mean(dict(b,robot=b["robot"].roll(1,0),robot_valid=b["robot_valid"].roll(1,0)))["mean"];a=infer_d(m,b,4,90);n1=m.forward_mean(dict(b,robot=trajectory(b["context"],a)))["mean"]
            for k,e in enumerate(es):rows.append(dict(parent=e["parent"],index=e["index"],anchor=e["anchor"],h=h))
            for key,x in (("real",b["world_raw"]),("cur",cur),("gt",to_raw(gt)),("no",to_raw(no)),("n1",to_raw(n1)),("sh",to_raw(sh))):arr[key].append(pool(x))
np.savez_compressed(OUT/"pooled.npz",**{k:np.concatenate(v) for k,v in arr.items()});(OUT/"rows.json").write_text(json.dumps(rows))
print("saved",len(rows),"parents",len({r["parent"] for r in rows}),flush=True)
