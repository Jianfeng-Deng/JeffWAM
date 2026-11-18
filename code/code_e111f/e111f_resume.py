import os,sys,time,json,hashlib,threading,subprocess,datetime,traceback
from pathlib import Path
import numpy as np,torch
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");ROOT=T/"code_e111f";SRC=T/"code_e102a";E110=T/"code_e110f"
INIT=T/"results/e106c_f2_s90_20261115_02/ckpt_20000.pt";F05=T/"results/e110f_f_20261116_05/ckpt_04000.pt"
RUN=T/"results/e111f_flow_20261116_02";TOTAL=16000;PEAK=6144;BUDGET=9000
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
 p=Path(p);tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));os.replace(tmp,p)
def gpu_rows():
 s=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
 return [dict(index=int(a),uuid=b.strip(),free=int(c)) for a,b,c in (l.split(",") for l in s.strip().splitlines())]
def disk_bytes():
 return sum(p.stat().st_size for root in [ROOT,RUN] if root.exists() for p in root.rglob("*") if p.is_file())
class Guard:
 def __init__(self,uuid):
  self.start=time.monotonic();self.uuid=uuid;self.stop=threading.Event();self.path=RUN/"gpu_ledger.json";self.last=-10
  row=next(r for r in gpu_rows() if r["uuid"]==uuid);assert row["free"]>=PEAK*1.5,("GPU admission",row)
  self.meta=dict(status="active",gpu=uuid,peak_estimate_mib=PEAK,free_at_start_mib=row["free"],started_brt=brt(),elapsed_s=0.)
  atomic(self.path,self.meta);self.th=threading.Thread(target=self.watch,daemon=True);self.th.start()
 def check(self,force=False):
  elapsed=time.monotonic()-self.start
  if not force and elapsed-self.last<5:return elapsed
  row=next(r for r in gpu_rows() if r["uuid"]==self.uuid);assert row["free"]>=PEAK*1.5,("shared GPU memory pressure",row)
  self.last=elapsed;assert elapsed<BUDGET,"E111-F 2.5 GPUh cap";assert disk_bytes()<9_500_000_000,"E111-F 10GB cap"
  if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()/2**20<=PEAK,"peak estimate exceeded"
  self.meta.update(last_gpu_row=row,elapsed_s=elapsed,disk_bytes=disk_bytes());atomic(self.path,self.meta);return elapsed
 def watch(self):
  while not self.stop.wait(2):
   try:self.check(True)
   except BaseException as e:self.meta.update(status="stopped_guard",error=repr(e),elapsed_s=time.monotonic()-self.start);atomic(self.path,self.meta);os._exit(88)
 def close(self):
  self.stop.set();self.th.join(5);self.meta.update(status="closed",elapsed_s=time.monotonic()-self.start,finished_brt=brt());atomic(self.path,self.meta)
def digest_step(stream,ids,hs):
 stream.update(np.asarray(ids,dtype="<i8").tobytes());stream.update(np.asarray(hs,dtype="<i8").tobytes())
def train():
 assert not RUN.exists();RUN.mkdir();guard=None
 try:
  torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(90);np.random.seed(90)
  _,_,tr,_=load();cp=torch.load(INIT,map_location="cpu",weights_only=False)
  assert hashlib.sha256(INIT.read_bytes()).hexdigest()=="dd27fe05c0345b93bdd8adeb9b95d47ee64c55ee7afb1a6b28fa55a0893fdfb5"
  model=WorldModel(tr.norm);model.load_state_dict(cp["model"],strict=True)
  uuid=os.environ.get("CUDA_VISIBLE_DEVICES");assert uuid and uuid.startswith("GPU-"),"explicit GPU UUID required"
  guard=Guard(uuid);torch.cuda.reset_peak_memory_stats();model=model.cuda()
  opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(cp["optimizer"]);del cp
  for pg in opt.param_groups:assert pg["lr"]==3e-4 and pg["weight_decay"]==1e-4
  opt_params={id(p) for pg in opt.param_groups for p in pg["params"]}
  assert opt_params=={id(p) for p in model.parameters()},"optimizer parameter coverage mismatch"
  trainable=set_trainables(model)
  for n,p in model.named_parameters():
   if n in trainable:
    st=opt.state.get(p);assert st and "exp_avg" in st and "exp_avg_sq" in st,("missing AdamW state",n)
    assert int(st["step"])==22000,("source step mismatch",n,int(st["step"]))
  frozen=tuple(n for n,_ in model.named_parameters() if n not in trainable);frozen_sha=tensor_sha(model,frozen)
  sampler=MultiSourceSampler(tr.rows,seed=90,weights={"piper":1.},caps={},demo3_cap=.45)
  assert "piper_4dc107335389f5711a0f65e0" not in {r["canonical_id"] for r in tr.rows}
  flowgen=torch.Generator(device="cuda");stream=hashlib.sha256();replay_hash=hashlib.sha256()
  start_opt_step=int(opt.state[model.delta_head.weight]["step"]);assert start_opt_step==22000,start_opt_step
  optimizer_test=dict(parameter_coverage=len(opt_params),model_parameters=sum(1 for _ in model.parameters()),trainable_parameters=len(trainable),stateful_trainables=sum(1 for n,p in model.named_parameters() if n in trainable and "exp_avg" in opt.state[p]),source_step=start_opt_step)
  # Recover omitted F05 AdamW and sampler states by exact replay from E106C.
  for step in range(1,4001):
   guard.check();seed=seed_for(90,"E110F",step);torch.manual_seed(seed);np.random.seed(seed&0xffffffff)
   ids=sampler.sample(32);hs=sample_horizons(22000+step,90);digest_step(stream,ids,hs);digest_step(replay_hash,ids,hs);b=move(tr.batch(ids,hs),"cuda")
   opt.zero_grad(set_to_none=True);model.train();flowgen.manual_seed(seed);loss,_=flow_loss(model,b,flowgen)
   assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for n,p in model.named_parameters() if n in frozen)
   gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(gn)
   opt.step();torch.cuda.synchronize()
   if step%500==0:
    atomic(RUN/"status.json",dict(status="deterministic_replay",replayed_steps=step,elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,gpu=uuid,brt=brt()))
    print("replay",step,round(guard.check(),1),flush=True)
  expected=torch.load(F05,map_location="cpu",weights_only=False)
  replay_sha=tensor_sha(model)
  expected_sha=hashlib.sha256()
  for n,_ in model.named_parameters():
   v=expected["model"][n];expected_sha.update(n.encode());expected_sha.update(v.detach().cpu().contiguous().numpy().tobytes())
  assert replay_sha==expected_sha.hexdigest(),"replayed F05 parameters differ; stopping before continuation"
  assert stream.hexdigest()=="38c5e37ede09b613a15786d0dd28f2f1df281493d585a3794b691f176092e6b3","E110-F sample stream mismatch"
  assert tensor_sha(model,frozen)==frozen_sha,"frozen parameters changed in replay"
  assert int(opt.state[model.delta_head.weight]["step"])==26000
  del expected;torch.cuda.empty_cache()
  atomic(RUN/"replay_unit_test.json",dict(status="passed",optimizer_restore=optimizer_test,source_optimizer_step=start_opt_step,replayed_updates=4000,restored_optimizer_step=int(opt.state[model.delta_head.weight]["step"]),f05_model_sha256=replay_sha,stream_sha256=stream.hexdigest(),frozen_sha_before=frozen_sha,frozen_sha_after_replay=tensor_sha(model,frozen),continuation_sampler_state="reconstructed by replay of exact deterministic E110-F draws",lr=opt.param_groups[0]["lr"],brt=brt()))
  with (RUN/"updates.jsonl").open("x") as log:
   for addstep in range(1,TOTAL+1):
    guard.check();step=4000+addstep;seed=seed_for(90,"E110F",step);torch.manual_seed(seed);np.random.seed(seed&0xffffffff)
    ids=sampler.sample(32);hs=sample_horizons(22000+step,90);digest_step(stream,ids,hs);b=move(tr.batch(ids,hs),"cuda")
    opt.zero_grad(set_to_none=True);model.train();flowgen.manual_seed(seed);loss,parts=flow_loss(model,b,flowgen)
    assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for n,p in model.named_parameters() if n in frozen)
    gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(gn)
    opt.step();torch.cuda.synchronize()
    last=dict(additional_step=addstep,e110f_step=step,loss=float(loss.detach()),grad_norm=float(gn),**{k:float(v) for k,v in parts.items()})
    if addstep==1 or addstep%50==0:
     log.write(json.dumps(last)+"\n");log.flush();atomic(RUN/"status.json",dict(status="training",additional_steps=TOTAL,elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,gpu=uuid,brt=brt(),**last))
    if addstep%500==0:print("F add",addstep,round(guard.check(),1),flush=True)
    if addstep%4000==0:
     name=f"ckpt_add_{addstep:05d}.pt";path=RUN/name;assert not path.exists()
     torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),sampler=sampler.state_dict(),additional_step=addstep,e110f_step=step,seed=90,source=str(F05),source_sha256=hashlib.sha256(F05.read_bytes()).hexdigest(),lr=opt.param_groups[0]["lr"],sample_stream_sha256=stream.hexdigest(),trainable=list(trainable),frozen_sha_before=frozen_sha,frozen_sha_after=tensor_sha(model,frozen)),path)
     atomic(RUN/f"ckpt_add_{addstep:05d}.json",dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),step=step,additional_step=addstep,optimizer_delta_head_step=int(opt.state[model.delta_head.weight]["step"]),sample_stream_sha256=stream.hexdigest(),brt=brt()))
     atomic(RUN/"status.json",dict(status="checkpoint_saved",additional_step=addstep,e110f_step=step,elapsed_s=guard.check(),peak_mib=torch.cuda.max_memory_allocated()/2**20,brt=brt(),**last))
  assert tensor_sha(model,frozen)==frozen_sha
  atomic(RUN/"summary.json",dict(status="complete",additional_steps=TOTAL,effective_flow_steps=4000+TOTAL,e110f_steps=4000+TOTAL,optimizer_delta_head_step=int(opt.state[model.delta_head.weight]["step"]),lr=opt.param_groups[0]["lr"],sample_stream_sha256=stream.hexdigest(),frozen_sha_before=frozen_sha,frozen_sha_after=tensor_sha(model,frozen),peak_mib=torch.cuda.max_memory_allocated()/2**20,elapsed_s=guard.check(),last=last,brt=brt()))
 except BaseException:
  atomic(RUN/"error.json",dict(status="failed",traceback=traceback.format_exc(),brt=brt()));raise
 finally:
  if guard:guard.close()
if __name__=="__main__":train()
