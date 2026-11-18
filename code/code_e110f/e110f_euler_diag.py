import os,sys,json,time,hashlib,subprocess,threading,datetime,traceback
from pathlib import Path
import numpy as np,torch
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");ROOT=T/"code_e110f";CONTROL=T/"results/e106c_control_20261115_02";RUN=T/"results/e106c_f2_s90_20261115_02";OUT=T/"results/e110f_d_20261116"
if str(T/"code_e102a") not in sys.path:sys.path.insert(0,str(T/"code_e102a"))
sys.path.insert(0,str(ROOT))
from e99w2_data import load
from e99w2_model import WorldModel
from e86.data import move
from e110f_core import infer_f,infer_d

def brt():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def gpu():
 s=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
 return [dict(index=int(a),uuid=b.strip(),free=int(c)) for a,b,c in (x.split(",") for x in s.strip().splitlines())]
def atomic(p,x):
 q=Path(p);tmp=q.with_suffix(q.suffix+".tmp");tmp.write_text(json.dumps(x,indent=2));os.replace(tmp,q)
class Guard:
 def __init__(self,phase,peak=6144):
  self.start=time.monotonic();self.peak=peak;self.uuid=os.environ["CUDA_VISIBLE_DEVICES"];self.path=OUT/"gpu_ledger.json";self.stop=threading.Event();self.meta=dict(phase=phase,status="active",gpu=self.uuid,peak_estimate_mib=peak,started_brt=brt(),elapsed_s=0.)
  assert not self.path.exists();self.check();atomic(self.path,self.meta);self.th=threading.Thread(target=self.watch,daemon=True);self.th.start()
 def check(self):
  row=next(x for x in gpu() if x["uuid"]==self.uuid);assert row["free"]>=self.peak*1.5,f"shared GPU memory below guard: {row}"
  assert time.monotonic()-self.start<9000,"E110-F GPU budget 2.5h exhausted"
  if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()/2**20<=self.peak,"peak memory estimate exceeded"
  self.meta["last_gpu_row"]=row;self.meta["elapsed_s"]=time.monotonic()-self.start;atomic(self.path,self.meta)
 def watch(self):
  while not self.stop.wait(2):
   try:self.check()
   except BaseException as e:self.meta.update(status="stopped_guard",error=repr(e),elapsed_s=time.monotonic()-self.start);atomic(self.path,self.meta);os._exit(88)
 def close(self):
  self.stop.set();self.th.join(4);self.meta.update(status="closed",elapsed_s=time.monotonic()-self.start,finished_brt=brt());atomic(self.path,self.meta)

def main():
 assert not OUT.exists();OUT.mkdir()
 guard=Guard("pretraining_euler_diagnostic")
 try:
  torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.cuda.reset_peak_memory_stats();device="cuda"
  _,_,tr,dev=load();lock=json.loads((CONTROL/"PRETRAIN_LOCK.json").read_text());entries=lock["entries"]
  assert len(entries)==211 and len({x["parent"] for x in entries})==27
  assert all(x["cid"]!="piper_4dc107335389f5711a0f65e0" for x in entries)
  model=WorldModel(tr.norm);cp=torch.load(RUN/"ckpt_20000.pt",map_location="cpu",weights_only=False);model.load_state_dict(cp["model"],strict=True);model=model.to(device).eval();del cp
  by={s:[] for s in ("D","S1","S2","S4","S8")};ji=torch.as_tensor([0,1,2,3,4,5,7,8,9,10,11,12],device=device)
  for start in range(0,len(entries),4):
   es=entries[start:start+4];ids=[x["index"] for x in es];b=move(dev.batch(ids,(10,),"short"),device);truth=b["action"]
   pred=infer_d(model,b,4,90);by["D"].append((es,pred,truth))
   for s in (1,2,4,8):by[f"S{s}"].append((es,infer_f(model,b,s,90),truth))
   if start%40==0:atomic(OUT/"status.json",dict(status="diagnostic",windows=min(start+4,len(entries)),total=len(entries),elapsed_s=guard.check(),brt=brt()))
  result={}
  for name,parts in by.items():
   parent={}
   for es,pred,truth in parts:
    err=(pred[:,:,ji]-truth[:,:,ji]).abs()
    for k,e in enumerate(es):
     a=err[k].mean(-1);parent.setdefault(e["parent"],[]).append([float(a[:8].mean()),float(a[:50].mean())])
   pv=np.array([np.array(v).mean(0) for v in parent.values()]);result[name]={"parents":len(parent),"l1_8_rad":float(pv[:,0].mean()),"l1_50_rad":float(pv[:,1].mean())}
  atomic(OUT/"euler_diagnostic.json",dict(status="complete",note="pretraining Euler check only; no step selection",checkpoint=str(RUN/"ckpt_20000.pt"),checkpoint_sha=hashlib.sha256((RUN/"ckpt_20000.pt").read_bytes()).hexdigest(),n_windows=len(entries),n_parents=27,metrics=result,peak_mib=torch.cuda.max_memory_allocated()/2**20,brt=brt()))
  print(json.dumps(result))
 finally:guard.close()
if __name__=="__main__":
 try:main()
 except BaseException as e:atomic(OUT/"error.json",dict(status="failed",error=traceback.format_exc(),brt=brt()));raise
