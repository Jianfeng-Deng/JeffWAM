import os,sys,json,time,hashlib,threading,subprocess,datetime,traceback
from pathlib import Path
import numpy as np,torch
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");SRC=T/"code_e102a";EXT=T/"code_e106c";ROOT=T/"code_e110f";CONTROL=T/"results/e106c_control_20261115_02";OUT=Path(sys.argv[2])
sys.path.insert(0,str(EXT));sys.path.insert(0,str(SRC));sys.path.insert(0,str(ROOT))
from e99w2_model import WorldModel,JOINTS,trajectory
from e102x_setup import datasets
from e102a_contract import contract,onset_labels
from e110f_core import infer_d,infer_f
from e86.data import move
PEAK=6144;BUDGET=9000
def brt():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def atomic(p,x):
 p=Path(p);tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(x,ensure_ascii=False));os.replace(tmp,p)
def gpu_rows():
 s=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
 return [dict(index=int(a),uuid=b.strip(),free=int(c)) for a,b,c in (x.split(",") for x in s.strip().splitlines())]
def used_gpu():
 n=0.
 for p in T.glob("results/e111f_*/gpu_ledger*.json"):
  try:n+=float(json.loads(p.read_text()).get("elapsed_s",0))
  except Exception:pass
 return n
class Guard:
 def __init__(self,run,uuid):
  self.start=time.monotonic();self.prior=used_gpu();self.last=-10.;self.stop=threading.Event();self.uuid=uuid;self.path=run/"gpu_ledger_eval.json";assert not self.path.exists()
  r=next(x for x in gpu_rows() if x["uuid"]==uuid);assert r["free"]>=PEAK*1.5
  self.meta=dict(status="active",gpu=uuid,peak_estimate_mib=PEAK,free_at_start_mib=r["free"],elapsed_s=0.,started_brt=brt());atomic(self.path,self.meta)
  self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
 def check(self,force=False):
  elapsed=time.monotonic()-self.start
  if not force and elapsed-self.last<5:return elapsed
  r=next(x for x in gpu_rows() if x["uuid"]==self.uuid);assert r["free"]>=PEAK*1.5,f"shared VRAM pressure {r}"
  assert self.prior+elapsed<BUDGET,"E110-F 2.5 GPUh cap"
  if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()/2**20<PEAK,"peak estimate exceeded"
  self.last=elapsed;self.meta.update(elapsed_s=elapsed,gpu_seconds_total=self.prior+elapsed,last_gpu_row=r);atomic(self.path,self.meta);return elapsed
 def watch(self):
  while not self.stop.wait(2):
   try:self.check(force=True)
   except BaseException as e:self.meta.update(status="stopped_guard",error=repr(e),elapsed_s=time.monotonic()-self.start);atomic(self.path,self.meta);os._exit(88)
 def close(self):
  self.stop.set();self.thread.join(5);self.meta.update(status="closed",elapsed_s=time.monotonic()-self.start,finished_brt=brt());atomic(self.path,self.meta)
def float_batch(b):return {k:(v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k,v in b.items()}
def transitions(s,initial):
 x=np.r_[int(initial),np.asarray(s,dtype=np.int8)]
 return [(int(i),int(x[i])) for i in np.flatnonzero(x[1:]!=x[:-1])+1]
def match(pred,true,tol):
 n,m=len(pred),len(true);dp=[[(-1,10**9,[]) for _ in range(m+1)] for _ in range(n+1)];dp[0][0]=(0,0,[])
 def put(i,j,v):
  o=dp[i][j]
  if v[0]>o[0] or (v[0]==o[0] and v[1]<o[1]):dp[i][j]=v
 for i in range(n+1):
  for j in range(m+1):
   v=dp[i][j]
   if v[0]<0:continue
   if i<n:put(i+1,j,v)
   if j<m:put(i,j+1,v)
   if i<n and j<m and pred[i][1]==true[j][1] and abs(pred[i][0]-true[j][0])<=tol:
    d=pred[i][0]-true[j][0];put(i+1,j+1,(v[0]+1,v[1]+abs(d),v[2]+[d]))
 return dp[n][m][2]
def main():
 assert not OUT.exists();OUT.mkdir()
 uuid=os.environ.get("CUDA_VISIBLE_DEVICES");assert uuid and uuid.startswith("GPU-")
 r=next(x for x in gpu_rows() if x["uuid"]==uuid);assert r["free"]>=PEAK*1.5
 guard=Guard(OUT,uuid)
 try:
  torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.cuda.reset_peak_memory_stats()
  _,_,tr,dev=datasets();lock=json.loads((CONTROL/"PRETRAIN_LOCK.json").read_text());entries=lock["entries"]
  assert len(entries)==211 and sum(e["original48"] for e in entries)==48 and len({e["parent"] for e in entries})==27
  assert all(e["cid"]!="piper_4dc107335389f5711a0f65e0" for e in entries)
  ddir=T/"results/e110f_d_20261116_05";dcp=torch.load(ddir/"ckpt_04000.pt",map_location="cpu",weights_only=False);fpath=Path(sys.argv[1]);fcp=torch.load(fpath,map_location="cpu",weights_only=False)
  assert dcp["sample_stream_sha256"]=="38c5e37ede09b613a15786d0dd28f2f1df281493d585a3794b691f176092e6b3","E110-F D stream mismatch"
  assert fcp.get("additional_step") in (4000,8000,12000,16000),"unexpected continuation checkpoint"
  c=contract();onset_cache={};action_rows=[];event_rows=[];world_rows=[];latency={}
  for group,cp in (("D",dcp),("F",fcp)):
   model=WorldModel(tr.norm);model.load_state_dict(cp["model"],strict=True);model=model.to("cuda").eval()
   tags=("D",) if group=="D" else ("S1","S2","S4","S8")
   with torch.inference_mode():
    for start in range(0,len(entries),4):
     es=entries[start:start+4];ids=[e["index"] for e in es]
     b=move(float_batch(dev.batch(ids,(10,),"short")),"cuda");truth=b["action"];qcur=b["q_current"];actions={}
     if group=="D":actions["D"]=[infer_d(model,b,4,90)]
     else:
      for s in (1,2,4,8):actions[f"S{s}"]=[infer_f(model,b,s,90+k) for k in range(8)]
     assert torch.isfinite(truth).all()
     assert all(torch.isfinite(a).all() for samples in actions.values() for a in samples),"nonfinite action prediction"
     for tag,aa in actions.items():
      stack=torch.stack(aa,0);one=stack[0];mean=stack.mean(0);median=stack.median(0).values
      ji=torch.as_tensor(JOINTS,device="cuda")
      def l1(x,h):return (x[:,:h,ji]-truth[:,:h,ji]).abs().mean((1,2))
      cp8=(qcur[:,None,ji]-truth[:,:8,ji]).abs().mean((1,2));cp50=(qcur[:,None,ji]-truth[:,:50,ji]).abs().mean((1,2))
      best8=torch.stack([l1(x,8) for x in aa]).min(0).values;best50=torch.stack([l1(x,50) for x in aa]).min(0).values
      std=stack[:,:,:,ji].std(0,unbiased=True).mean((1,2)) if len(aa)>1 else torch.zeros(len(es),device="cuda")
      rmse=((mean[:,:,ji]-truth[:,:,ji]).square().mean((1,2))).sqrt();ratio=std/rmse.clamp_min(1e-9)
      for k,e in enumerate(es):
       rec=dict(group=tag,index=e["index"],parent=e["parent"],original48=e["original48"],l1_8=float(l1(one,8)[k]),l1_50=float(l1(one,50)[k]),copy8=float(cp8[k]),copy50=float(cp50[k]))
       if len(aa)>1:rec.update(k8mean8=float(l1(mean,8)[k]),k8mean50=float(l1(mean,50)[k]),k8median8=float(l1(median,8)[k]),k8median50=float(l1(median,50)[k]),best8=float(best8[k]),best50=float(best50[k]),spread=float(std[k]),ensemble_rmse=float(rmse[k]),spread_skill=float(ratio[k]),collapsed=bool(ratio[k]<.5))
       action_rows.append(rec)
       pred_states=(one[k,:,torch.as_tensor([6,13],device="cuda")].cpu().numpy()>np.asarray(c["threshold_m"])).astype("i1")
       truth_states=b["grip_label"][k].cpu().numpy().astype("i1");cur=(b["grip_current"][k].cpu().numpy()>.5).astype("i1")
       z,q,pose,good=dev.states.get(dev,e["cid"])
       if e["cid"] not in onset_cache:onset_cache[e["cid"]]=onset_labels(q,z["continuous_segment"],good,c)[0]
       init=onset_cache[e["cid"]][e["anchor"]]
       ev={str(t):dict(tp=0,pred=0,true=0,minutes=50/(30*60)) for t in (5,10,20)}
       for side in range(2):
        pe=transitions(pred_states[:,side],cur[side]);te=transitions(truth_states[:,side],init[side])
        for tol in (5,10,20):
         tp=len(match(pe,te,tol));ev[str(tol)]["tp"]+=tp;ev[str(tol)]["pred"]+=len(pe);ev[str(tol)]["true"]+=len(te)
       event_rows.append(dict(group=tag,index=e["index"],parent=e["parent"],original48=e["original48"],state_accuracy=float((pred_states==truth_states).mean()),events=ev))
     for h in (25,50,100):
      pos=[k for k,e in enumerate(es) if h in e["horizons"]]
      if not pos:continue
      eh=[es[k] for k in pos];hid=[e["index"] for e in eh]
      bh=move(float_batch(dev.batch(hid,(h,),"short")),"cuda");Y=bh["world"].double()
      assert bh["world_valid"].all() and torch.isfinite(Y).all()
      Pno=model.forward_mean(dict(bh,robot=torch.zeros_like(bh["robot"]),robot_valid=torch.zeros_like(bh["robot_valid"])))["mean"].double()
      copy_mse=Y.square().mean((1,2));no_mse=(Pno-Y).square().mean((1,2))
      for tag,aa in actions.items():
       use=[a[pos] for a in aa];wp=[]
       for a in use:
        predw=model.forward_mean(dict(bh,robot=trajectory(bh["context"],a)))["mean"].double()
        assert torch.isfinite(predw).all(),"nonfinite action-conditioned world prediction"
        wp.append(predw)
       for method,P in (("N1",wp[0]),("N8",torch.stack(wp).mean(0))):
        mse=(P-Y).square().mean((1,2))
        for k,e in enumerate(eh):world_rows.append(dict(group=tag,index=e["index"],parent=e["parent"],original48=e["original48"],h=h,method=method,mse=float(mse[k]),copy=float(copy_mse[k]),no_traj_mse=float(no_mse[k])))
     if start%40==0:atomic(OUT/"status.json",dict(status="evaluation",windows=min(start+4,len(entries)),total=len(entries),elapsed_s=guard.check(),brt=brt()))
   # Cached-feature inference latency for each deployed sampler setting.
   b=move(float_batch(dev.batch([entries[0]["index"]],(10,),"short")),"cuda")
   for tag in tags:
    fn=(lambda:infer_d(model,b,4,90)) if tag=="D" else (lambda:infer_f(model,b,int(tag[1:]),90))
    for _ in range(10):fn()
    torch.cuda.synchronize();times=[]
    for _ in range(100):
     guard.check();t=time.perf_counter();fn();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
    latency[tag]=dict(batch=1,median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),repeats=100)
   del model;torch.cuda.empty_cache()
  result=dict(status="raw_complete",d_sha=hashlib.sha256((ddir/"ckpt_04000.pt").read_bytes()).hexdigest(),f_sha=hashlib.sha256(fpath.read_bytes()).hexdigest(),sample_stream_sha256=dcp["sample_stream_sha256"],n_windows=211,n_parents=27,original48=48,actions=action_rows,events=event_rows,world=world_rows,latency=latency,peak_mib=torch.cuda.max_memory_allocated()/2**20,gpu_seconds=guard.prior+time.monotonic()-guard.start,brt=brt())
  atomic(OUT/"raw_eval.json",result);atomic(OUT/"status.json",dict(status="raw_complete",windows=211,total=211,elapsed_s=guard.check(),brt=brt()))
  print(json.dumps(dict(actions=len(action_rows),events=len(event_rows),world=len(world_rows),latency=latency),ensure_ascii=False),flush=True)
 finally:guard.close()
if __name__=="__main__":
 try:main()
 except BaseException: 
  if OUT.exists():atomic(OUT/"error.json",dict(status="failed",error=traceback.format_exc(),brt=brt()))
  raise
