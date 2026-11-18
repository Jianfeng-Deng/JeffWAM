from common import *
from e99w2_model import WorldModel
from e102x_setup import datasets
from e110f_eval import float_batch
from e86.data import move
import json
_,_,tr,dev=datasets();ent=json.loads((T/"results/e106c_control_20261115_02/PRETRAIN_LOCK.json").read_text())["entries"]
avail={p.stem for p in FEATS.glob("pc_*.npz")};E=[e for e in ent if e["parent"] in avail][:6]
b=dev.batch([e["index"] for e in E],(25,),"short")
print({k:(tuple(v.shape),str(v.dtype)) for k,v in b.items() if k in("world","world_positions","world_valid","world_horizons","obs")})
Y=b["world"].float();pos=b["world_positions"];print("positions",pos.tolist())
print({k:tuple(v.shape) for k,v in b.items() if k.startswith("world_")})
Wr=b["world_raw"].float();Wc=b["world_current"].float()
for k,e in enumerate(E[:4]):
    f=np.load(T/"results/e102s_20261115_01/features"/e["parent"]/"b.npy",mmap_mode="r");ref25=np.asarray(f[e["anchor"]+25],dtype="f4");ref0=np.asarray(f[e["anchor"]],dtype="f4")
    # raw future tokens 24x768 : which pooled arrangement matches the per-frame 2304-D b feature?
    y=Wr[k].numpy();c=Wc[k].numpy()
    for name,arr,ref in (("future@+25",y,ref25),("current",c,ref0)):
        for nm,pooled in (("cam-major 3x8",np.concatenate([arr[g*8:(g+1)*8].mean(0) for g in range(3)])),("pos-major 8x3",np.concatenate([arr[g::3].mean(0) for g in range(3)]))):
            print(e["parent"][3:11],name,nm,"maxabs %.4f corr %.4f"%(abs(pooled-ref).max(),np.corrcoef(pooled,ref)[0,1]))
