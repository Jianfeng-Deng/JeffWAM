import os,sys,time,json,hashlib,subprocess
from common import *
from e99w2_data import load
from e99w2_model import WorldModel
from e90.sampling import MultiSourceSampler
from e90.data import sample_horizons
from e86.common import seed_for
from e86.data import move
from e110f_core import set_trainables,tensor_sha,deterministic_loss
ARM=sys.argv[1];KIND={"S0":"zero","S1":"ctrl","S2":"oracle"}[ARM];STEPS=4000
run=T/f"results/e115p1_{ARM.lower()}_20261117_01";assert not run.exists();run.mkdir()
torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(90);np.random.seed(90)
_,_,tr,_=load();cp=torch.load(INIT,map_location="cpu",weights_only=False)
model=WorldModel(tr.norm);model.load_state_dict(cp["model"],strict=True);model=widen(model).cuda()
avail={p.stem for p in FEATS.glob("pc_*.npz")}
sub=[i for i,r in enumerate(tr.rows) if r["parent_component"] in avail and r["canonical_id"]!="piper_4dc107335389f5711a0f65e0"]
print(ARM,"subset rows",len(sub),"parents",len({tr.rows[i]["parent_component"] for i in sub}),flush=True)
trainable=set_trainables(model);frozen=tuple(n for n,_ in model.named_parameters() if n not in trainable);fsha=tensor_sha(model,frozen)
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
sampler=MultiSourceSampler([tr.rows[i] for i in sub],seed=90,weights={"piper":1.},caps={},demo3_cap=.45);stream=hashlib.sha256();log=(run/"updates.jsonl").open("x");t0=time.monotonic()
for step in range(1,STEPS+1):
    seed=seed_for(90,"E115P1",step);torch.manual_seed(seed);np.random.seed(seed&0xffffffff)
    ids=sampler.sample(32);orig=[sub[i] for i in ids];hs=sample_horizons(22000+step,90);stream.update(np.asarray(orig,dtype="<i8").tobytes())
    b=move(tr.batch(orig,hs),"cuda");b=add_stage(b,[tr.rows[i] for i in orig],KIND)
    opt.zero_grad(set_to_none=True);model.train();loss,parts=deterministic_loss(model,b);assert torch.isfinite(loss);loss.backward()
    gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step()
    if step==1 or step%50==0:log.write(json.dumps(dict(step=step,loss=float(loss),delta=float(parts["delta"]),grip=float(parts["grip"]),elapsed=time.monotonic()-t0))+"\n");log.flush()
    if step%500==0:print(ARM,step,round(time.monotonic()-t0),float(parts["delta"]),float(parts["grip"]),flush=True)
assert tensor_sha(model,frozen)==fsha
torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=ARM,kind=KIND,steps=STEPS,sample_stream_sha256=stream.hexdigest(),subset_rows=len(sub),source=str(INIT)),run/"ckpt_04000.pt")
(run/"summary.json").write_text(json.dumps(dict(status="complete",arm=ARM,stream=stream.hexdigest(),elapsed_s=time.monotonic()-t0,brt=brt())))
print("done",ARM,stream.hexdigest()[:12],flush=True)
