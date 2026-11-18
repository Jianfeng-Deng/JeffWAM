import os,sys,time,json,hashlib,threading,subprocess,datetime,traceback
from pathlib import Path
import numpy as np,torch
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");ROOT=T/"code_e111f";SRC=T/"code_e102a";E110=T/"code_e110f"
PREV=T/"results/e111f_flow_20261116_04";CKPT=PREV/"ckpt_add_08000.pt"
RUN=T/"results/e111f_flow_20261116_05";TOTAL=16000;PEAK=6144;BUDGET=9000
sys.path.insert(0,str(SRC));sys.path.insert(0,str(ROOT));sys.path.insert(0,str(E110))
from e99w2_data import load
from e99w2_model import WorldModel
from e90.sampling import MultiSourceSampler
from e90.data import sample_horizons
from e86.common import seed_for
from e86.data import move
from e110f_core import set_trainables,tensor_sha,flow_loss
def brt():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def atomic(p,x):
 p=Path(p);tmp=p.with_name(p.name+f".{threading.get_ident()}.tmp");tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));os.replace(tmp,p)
def gpu_rows():
 s=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
 return [dict(index=int(a),uuid=b.strip(),free=int(c)) for a,b,c in (l.split(",") for l in s.strip().splitlines())]
def disk_bytes():
 return sum(p.stat().st_size for root in [ROOT,*T.glob("results/e111f_*")] if root.exists() for p in root.rglob("*") if p.is_file())
def prior_gpu():
 total=0.
 for p in T.glob("results/e111f_*/gpu_ledger*.json"):
  try:total+=float(json.loads(p.read_text()).get("elapsed_s",0))
  except Exception:pass
 return total
class Guard:
 def __init__(self,uuid):
  self.start=time.monotonic();self.prior=prior_gpu();self.uuid=uuid;self.stop=threading.Event();self.path=RUN/"gpu_ledger.json";self.last=-10
  row=next(r for r in gpu_rows() if r["uuid"]==uuid);assert row["free"]>=PEAK*1.5,("GPU admission",row)
  self.meta=dict(status="active",gpu=uuid,peak_estimate_mib=PEAK,free_at_start_mib=row["free"],prior_gpu_seconds=self.prior,started_brt=brt(),elapsed_s=0.)
  atomic(self.path,self.meta);self.th=threading.Thread(target=self.watch,daemon=True);self.th.start()
 def check(self,force=False):
  elapsed=time.monotonic()-self.start
  if not force and elapsed-self.last<5:return elapsed
  row=next(r for r in gpu_rows() if r["uuid"]==self.uuid);assert row["free"]>=PEAK*1.5,("shared GPU memory pressure",row)
  self.last=elapsed;assert self.prior+elapsed<BUDGET,"E111-F 2.5 GPUh cap";assert disk_bytes()<9_500_000_000,"E111-F 10GB cap"
  if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()/2**20<=PEAK,"peak estimate exceeded"
  self.meta.update(last_gpu_row=row,elapsed_s=elapsed,gpu_seconds_total=self.prior+elapsed,disk_bytes=disk_bytes());atomic(self.path,self.meta);return elapsed
 def watch(self):
  while not self.stop.wait(2):
   try:self.check(True)
   except BaseException as e:self.meta.update(status="stopped_guard",error=repr(e),elapsed_s=time.monotonic()-self.start);atomic(self.path,self.meta);os._exit(88)
 def close(self):
  self.stop.set();self.th.join(5);self.meta.update(status="closed",elapsed_s=time.monotonic()-self.start,gpu_seconds_total=self.prior+time.monotonic()-self.start,finished_brt=brt());atomic(self.path,self.meta)
def digest_step(stream,ids,hs):
 stream.update(np.asarray(ids,dtype="<i8").tobytes());stream.update(np.asarray(hs,dtype="<i8").tobytes())
def train():
 assert not RUN.exists();RUN.mkdir();guard=None
 try:
  torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(90);np.random.seed(90)
  _,_,tr,_=load()
  cp_sha=hashlib.sha256(CKPT.read_bytes()).hexdigest();assert cp_sha=="ef3709299f67677bac8009410bba0583a674360a81488048b9394b96aa647175"
  cp=torch.load(CKPT,map_location="cpu",weights_only=False)
  assert cp["additional_step"]==8000 and cp["e110f_step"]==12000
  model=WorldModel(tr.norm);model.load_state_dict(cp["model"],strict=True)
  uuid=os.environ.get("CUDA_VISIBLE_DEVICES");assert uuid and uuid.startswith("GPU-"),"explicit GPU UUID required"
  guard=Guard(uuid);torch.cuda.reset_peak_memory_stats();model=model.cuda()
  opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(cp["optimizer"])
  assert all(pg["lr"]==3e-4 and pg["weight_decay"]==1e-4 for pg in opt.param_groups)
  opt_params={id(p) for pg in opt.param_groups for p in pg["params"]};assert opt_params=={id(p) for p in model.parameters()}
  trainable=set_trainables(model)
  for n,p in model.named_parameters():
   if n in trainable:
    st=opt.state.get(p);assert st and "exp_avg" in st and "exp_avg_sq" in st and int(st["step"])==34000,(n,"AdamW restore mismatch")
  frozen=tuple(n for n,_ in model.named_parameters() if n not in trainable);frozen_sha=tensor_sha(model,frozen)
  assert frozen_sha==cp["frozen_sha_after"]
  sampler=MultiSourceSampler(tr.rows,seed=90,weights={"piper":1.},caps={},demo3_cap=.45)
  assert "piper_4dc107335389f5711a0f65e0" not in {r["canonical_id"] for r in tr.rows}
  sampler.load_state_dict(cp["sampler"])
  probe=MultiSourceSampler(tr.rows,seed=90,weights={"piper":1.},caps={},demo3_cap=.45);probe.load_state_dict(cp["sampler"])
  stream=hashlib.sha256();flowgen=torch.Generator(device="cuda")
  # First post-checkpoint sample is compared to a second restore from the saved state.
  start_step=int(cp["e110f_step"]);next_step=start_step+1;seed=seed_for(90,"E110F",next_step)
  torch.manual_seed(seed);np.random.seed(seed&0xffffffff)
  ids=sampler.sample(32);probe_ids=probe.sample(32);assert ids==probe_ids,"sampler state restore mismatch"
  hs=sample_horizons(22000+next_step,90);digest_step(stream,ids,hs);b=move(tr.batch(ids,hs),"cuda")
  opt.zero_grad(set_to_none=True);model.train();flowgen.manual_seed(seed);loss,_=flow_loss(model,b,flowgen)
  assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for n,p in model.named_parameters() if n in frozen)
  gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(gn)
  opt.step();torch.cuda.synchronize()
  assert tensor_sha(model,frozen)==frozen_sha,"frozen parameters changed during restore probe"
  atomic(RUN/"restore_unit_test.json",dict(status="passed",checkpoint_sha256=cp_sha,additional_step=8000,e110f_step=start_step,optimizer_step_before=34000,optimizer_step_after=int(opt.state[model.delta_head.weight]["step"]),optimizer_parameter_coverage=len(opt_params),trainable_parameters=len(trainable),sampler_state_restored=True,first_next_batch_match=True,frozen_sha_before=frozen_sha,frozen_sha_after_first_step=tensor_sha(model,frozen),lr=opt.param_groups[0]["lr"],weight_decay=opt.param_groups[0]["weight_decay"],brt=brt()))
  del cp;last={}
  with (RUN/"updates.jsonl").open("x") as log:
   for addstep in range(8002,TOTAL+1):
    guard.check();step=12000+(addstep-8000);seed=seed_for(90,"E110F",step);torch.manual_seed(seed);np.random.seed(seed&0xffffffff)
    ids=sampler.sample(32);hs=sample_horizons(22000+step,90);digest_step(stream,ids,hs);b=move(tr.batch(ids,hs),"cuda")
    opt.zero_grad(set_to_none=True);model.train();flowgen.manual_seed(seed);loss,parts=flow_loss(model,b,flowgen)
    assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for n,p in model.named_parameters() if n in frozen)
    gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(gn)
    opt.step();torch.cuda.synchronize()
    last=dict(additional_step=addstep,e110f_step=step,loss=float(loss.detach()),grad_norm=float(gn),**{k:float(v) for k,v in parts.items()})
    if addstep%50==0:
     log.write(json.dumps(last)+"\n");log.flush()
     atomic(RUN/"status.json",dict(status="training",additional_steps_total=TOTAL,elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,gpu=uuid,brt=brt(),last=last))
    if addstep%500==0:print("F add",addstep,round(guard.check(),1),flush=True)
    if addstep in (12000,16000):
     name=f"ckpt_add_{addstep:05d}.pt";path=RUN/name;assert not path.exists()
     torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),sampler=sampler.state_dict(),additional_step=addstep,e110f_step=step,seed=90,source=str(CKPT),source_sha256=cp_sha,lr=opt.param_groups[0]["lr"],sample_stream_prefix_sha256="c81a30a3682d3b6531bf93a17a5954510073e7e733bedd865e37127f1a9a05b0",prior_suffix_sample_stream_sha256="577e819027291adefb813ecc4f3f5aaafb1fc3819dbd4e7709cf613ce5000c1a",suffix_sample_stream_sha256=stream.hexdigest(),trainable=list(trainable),frozen_sha_before=frozen_sha,frozen_sha_after=tensor_sha(model,frozen),optimizer_delta_head_step=int(opt.state[model.delta_head.weight]["step"])),path)
     atomic(RUN/f"ckpt_add_{addstep:05d}.json",dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),step=step,additional_step=addstep,optimizer_delta_head_step=int(opt.state[model.delta_head.weight]["step"]),sample_stream_prefix_sha256="c81a30a3682d3b6531bf93a17a5954510073e7e733bedd865e37127f1a9a05b0",prior_suffix_sample_stream_sha256="577e819027291adefb813ecc4f3f5aaafb1fc3819dbd4e7709cf613ce5000c1a",suffix_sample_stream_sha256=stream.hexdigest(),brt=brt()))
     atomic(RUN/"status.json",dict(status="checkpoint_saved",elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,last=last,brt=brt()))
  assert tensor_sha(model,frozen)==frozen_sha
  atomic(RUN/"summary.json",dict(status="complete",additional_steps=TOTAL,effective_flow_steps=20000,e110f_step=20000,optimizer_delta_head_step=int(opt.state[model.delta_head.weight]["step"]),lr=opt.param_groups[0]["lr"],frozen_sha_before=frozen_sha,frozen_sha_after=tensor_sha(model,frozen),elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,last=last,brt=brt()))
 except BaseException:
  atomic(RUN/"error.json",dict(status="failed",traceback=traceback.format_exc(),brt=brt()));raise
 finally:
  if guard:guard.close()
if __name__=="__main__":train()
