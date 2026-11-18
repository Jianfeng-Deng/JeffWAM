import os,sys,time,json,hashlib,threading,subprocess,datetime,traceback
from pathlib import Path
import numpy as np,torch
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");ROOT=T/"code_e110f";SRC=T/"code_e102a";CONTROL=T/"results/e106c_control_20261115_02";INIT=T/"results/e106c_f2_s90_20261115_02/ckpt_20000.pt"
sys.path.insert(0,str(SRC));sys.path.insert(0,str(ROOT))
from e99w2_data import load
from e99w2_model import WorldModel
from e90.sampling import MultiSourceSampler
from e90.data import sample_horizons
from e86.common import seed_for
from e86.data import move
from e110f_core import set_trainables,tensor_sha,deterministic_loss,flow_loss
STEPS=4000;PEAK=6144;BUDGET=9000
def brt():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def atomic(p,x):
 p=Path(p);tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));os.replace(tmp,p)
def gpu_rows():
 s=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
 return [dict(index=int(a),uuid=b.strip(),free=int(c)) for a,b,c in (l.split(",") for l in s.strip().splitlines())]
def gpu_used():
 total=0
 for p in T.glob("results/e110f_*/gpu_ledger*.json"):
  try:total+=float(json.loads(p.read_text()).get("elapsed_s",0))
  except Exception:pass
 return total
def disk_used():
 total=0
 for root in [ROOT,*T.glob("results/e110f_*")]:
  if root.is_dir():
   for p in root.rglob("*"):
    if p.is_file():total+=p.stat().st_size
 return total
class Guard:
 def __init__(self,run,phase,uuid):
  self.start=time.monotonic();self.uuid=uuid;self.run=run;self.stop=threading.Event();self.path=run/"gpu_ledger_train.json";self.prior=gpu_used();self.last_check=-10.;assert not self.path.exists()
  row=next(r for r in gpu_rows() if r["uuid"]==uuid);assert row["free"]>=PEAK*1.5
  self.meta=dict(phase=phase,status="active",gpu=uuid,peak_estimate_mib=PEAK,free_at_start_mib=row["free"],started_brt=brt(),elapsed_s=0.)
  atomic(self.path,self.meta);self.th=threading.Thread(target=self.watch,daemon=True);self.th.start()
 def check(self,force=False):
  elapsed=time.monotonic()-self.start
  if not force and elapsed-self.last_check<5:return elapsed
  row=next(r for r in gpu_rows() if r["uuid"]==self.uuid);assert row["free"]>=PEAK*1.5,f"shared GPU memory pressure: {row}"
  self.last_check=elapsed;assert self.prior+elapsed<BUDGET,"E110-F 2.5 GPUh cap"
  assert disk_used()<9_900_000_000,"E110-F 10GB cap"
  if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()/2**20<=PEAK,"peak estimate exceeded"
  self.meta.update(last_gpu_row=row,elapsed_s=elapsed,gpu_seconds_total=self.prior+elapsed,disk_bytes=disk_used());atomic(self.path,self.meta)
  return elapsed
 def watch(self):
  while not self.stop.wait(2):
   try:self.check(force=True)
   except BaseException as e:self.meta.update(status="stopped_guard",error=repr(e),elapsed_s=time.monotonic()-self.start);atomic(self.path,self.meta);os._exit(88)
 def close(self):
  self.stop.set();self.th.join(5);self.meta.update(status="closed",elapsed_s=time.monotonic()-self.start,finished_brt=brt());atomic(self.path,self.meta)
def train(group):
 assert group in ("D","F")
 run=T/f"results/e110f_{group.lower()}_20261116_05";assert not run.exists();run.mkdir()
 guard=None
 try:
  torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(90);np.random.seed(90)
  _,_,tr,_=load();cp=torch.load(INIT,map_location="cpu",weights_only=False)
  source_sha=hashlib.sha256(INIT.read_bytes()).hexdigest();assert source_sha=="dd27fe05c0345b93bdd8adeb9b95d47ee64c55ee7afb1a6b28fa55a0893fdfb5"
  model=WorldModel(tr.norm);model.load_state_dict(cp["model"],strict=True)
  uuid=os.environ.get("CUDA_VISIBLE_DEVICES");assert uuid and uuid.startswith("GPU-"),"explicit GPU UUID required"
  row=next(r for r in gpu_rows() if r["uuid"]==uuid);assert row["free"]>=PEAK*1.5,f"insufficient free VRAM: {row}"
  guard=Guard(run,"train_"+group,uuid);torch.cuda.reset_peak_memory_stats();model=model.cuda()
  opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(cp["optimizer"])
  trainable=set_trainables(model);frozen=tuple(n for n,_ in model.named_parameters() if n not in trainable);frozen_sha=tensor_sha(model,frozen)
  sampler=MultiSourceSampler(tr.rows,seed=90,weights={"piper":1.},caps={},demo3_cap=.45);stream=hashlib.sha256();first_probe=None
  assert "piper_4dc107335389f5711a0f65e0" not in {r["canonical_id"] for r in tr.rows}
  flowgen=torch.Generator(device="cuda")
  last={}
  with (run/"updates.jsonl").open("x") as log:
   for step in range(1,STEPS+1):
    elapsed=guard.check();seed=seed_for(90,"E110F",step);torch.manual_seed(seed);np.random.seed(seed&0xffffffff)
    ids=sampler.sample(32);hs=sample_horizons(22000+step,90);stream.update(np.asarray(ids,dtype="<i8").tobytes());stream.update(np.asarray(hs,dtype="<i8").tobytes());b=move(tr.batch(ids,hs),"cuda")
    if step==1:
     first_probe=(list(ids),tuple(hs));model.eval()
     with torch.no_grad():
      w0=torch.zeros_like(b["world"]);t0=torch.zeros(len(b["obs"]),device="cuda");world_before=model(b,None,w0,t0,t0)["mean"].detach().cpu()
    opt.zero_grad(set_to_none=True);model.train()
    if group=="D":loss,parts=deterministic_loss(model,b)
    else:
     flowgen.manual_seed(seed);loss,parts=flow_loss(model,b,flowgen)
    assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if n in frozen)
    gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    assert torch.isfinite(gn);opt.step();torch.cuda.synchronize()
    last=dict(step=step,loss=float(loss.detach()),grad_norm=float(gn),horizons=list(hs),**{k:float(v) for k,v in parts.items()})
    if step==1 or step%50==0:
     log.write(json.dumps(last)+"\n");log.flush();atomic(run/"status.json",dict(status="training",group=group,steps=STEPS,elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,gpu=guard.uuid,brt=brt(),**last))
    if step%500==0:print(group,step,round(guard.check(),1),flush=True)
  after=tensor_sha(model,frozen);assert after==frozen_sha
  pi,ph=first_probe;pb=move(tr.batch(pi,ph),"cuda");model.eval();t0=torch.zeros(len(pb["obs"]),device="cuda")
  with torch.no_grad():world_after=model(pb,None,torch.zeros_like(pb["world"]),t0,t0)["mean"].detach().cpu()
  assert torch.equal(world_before,world_after),"frozen unconditioned world output changed"
  torch.cuda.synchronize();state={k:v.detach().cpu() for k,v in model.state_dict().items()}
  cpout=dict(model=state,step=STEPS,group=group,seed=90,source=str(INIT),source_sha=source_sha,trainable=list(trainable),frozen_sha_before=frozen_sha,frozen_sha_after=after,optimizer_step="source AdamW state continued",sample_stream_sha256=stream.hexdigest(),unconditioned_world_output_bitwise=True)
  torch.save(cpout,run/"ckpt_04000.pt")
  atomic(run/"summary.json",dict(status="complete",group=group,step=STEPS,source_sha=source_sha,trainable=list(trainable),frozen_sha_before=frozen_sha,frozen_sha_after=after,sample_stream_sha256=stream.hexdigest(),unconditioned_world_output_bitwise=True,peak_mib=torch.cuda.max_memory_allocated()/2**20,elapsed_s=guard.check(),last=last,brt=brt()))
  print("completed",group,STEPS,round(guard.check(),1),flush=True)
 finally:
  if guard:guard.close()
if __name__=="__main__":
 try:train(sys.argv[1])
 except BaseException:
  run=T/f"results/e110f_{sys.argv[1].lower()}_20261116_02"
  if run.exists():atomic(run/"error.json",dict(status="failed",error=traceback.format_exc(),brt=brt()))
  raise
