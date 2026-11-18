import os,sys,json,time,threading,signal,subprocess
os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",CUDA_VISIBLE_DEVICES="4",PYTHONDONTWRITEBYTECODE="1")
from pathlib import Path
import numpy as np
import io,struct
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu"); R=T/"results/e116p3_20261117_01"; CODE=T/"code_e116p3"; REPORT=T/"notes/e116p3_language_alignment_2026-11-17.md"
def brt():return time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime(time.time()-3*3600))
def status(stage,**kw):
 d=dict(stage=stage,brt=brt(),**kw);(R/"status.json").write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n");print(json.dumps(d,ensure_ascii=False),flush=True)
def gpu():
 return [[int(y.strip()) for y in x.split(",")] for x in subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free,memory.used","--format=csv,noheader,nounits"],text=True).splitlines()]
def main():
 assert (CODE/"TEXT_LOCKED.json").exists() if False else True
 assert not (R/"COMPLETE.json").exists()
 # Admission: estimated full-model inference peak 10 GiB; require 15 GiB free.
 rows=gpu(); candidates=[x for x in rows if x[0]==4 and x[1]>=15360]; assert candidates,rows
 dev_id,initial_free,_=candidates[0]; peak=10240
 import torch
 from PIL import Image
 torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_device(0)
 started=time.monotonic(); closed=False; stop=threading.Event()
 def watch():
  while not stop.wait(2):
   try:
    free=next(x[1] for x in gpu() if x[0]==dev_id)
    if free<15360 or time.monotonic()-started>1790:
     status("aborted_shared_gpu_pressure_or_budget",gpu=dev_id,free_mib=free,elapsed_s=time.monotonic()-started);os.kill(os.getpid(),signal.SIGTERM)
   except Exception:pass
 th=threading.Thread(target=watch,daemon=True);th.start()
 ledger=R/"gpu_ledger.json"; ledger.write_text(json.dumps([dict(gpu=dev_id,initial_free_mib=initial_free,estimated_peak_mib=peak,start_brt=brt(),seconds=0,status="running")],indent=2)+"\n")
 try:
  from transformers import AutoTokenizer,SiglipModel
  sys.path.insert(0,str(T/"code_e86"));from e86.vision import preprocess,MODEL
  def selected_frames(path,ids):
   ids=list(ids);expr="+".join(f"eq(n\\,{int(i)})" for i in ids)
   cmd=["ffmpeg","-hide_banner","-loglevel","error","-i",str(path),"-vf",f"select={expr}","-vsync","0","-an","-sn","-f","image2pipe","-vcodec","png","pipe:1"]
   raw=subprocess.check_output(cmd,stderr=subprocess.DEVNULL);out=[];sig=bytes([137,80,78,71,13,10,26,10]);pos=0
   while pos<len(raw):
    assert raw[pos:pos+8]==sig,(path,pos,len(raw))
    q=pos+8
    while True:
     n=struct.unpack(">I",raw[q:q+4])[0];typ=raw[q+4:q+8];q+=12+n
     if typ==b"IEND":break
    out.append(np.array(Image.open(io.BytesIO(raw[pos:q])).convert("RGB")));pos=q
   assert len(out)==len(ids),(path,len(ids),len(out))
   return out
  model=SiglipModel.from_pretrained(MODEL,local_files_only=True).eval().cuda(); model.requires_grad_(False)
  tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=True)
  torch.backends.mha.set_fastpath_enabled(False)
  # Exercise exact E86 aspect-padding preprocessing and repeat-image embedding before reading dev labels.
  src=json.loads((T/"results/e99b3_20261115_01/sources.json").read_text());tr=next(s for s in src if s["pool"]=="train")
  cam="cam_high";frame=17
  rgb=selected_frames(tr["videos"][cam],[frame])[0];pixel,_,_=preprocess(rgb)
  with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
   z1=model.get_image_features(pixel_values=pixel[None].cuda()).float();z2=model.get_image_features(pixel_values=pixel[None].cuda()).float()
  diff=float((z1-z2).abs().max());assert torch.isfinite(z1).all() and diff<=1e-6,diff
  # Lock all stage descriptions and predicate text. Required tokenizer padding and fixed context length.
  lock=json.loads((R/"text_lock.json").read_text());texts=[];text_meta=[]
  for s in lock["stages"]:
   for group,field in (("name","name_en"),("start","visible_start_en"),("start_plus_end","visible_start_en")):
    texts.append(s[field] if group!="start_plus_end" else s[field]+" "+s["visible_end_en"]);text_meta.append((s["id"],group,"en"))
   for group,field in (("name","name_zh"),("start","visible_start_zh"),("start_plus_end","visible_start_zh")):
    texts.append(s[field] if group!="start_plus_end" else s[field]+" "+s["visible_end_zh"]);text_meta.append((s["id"],group,"zh"))
  for p in lock["predicates"]:
   for lang,field in (("en","question_en"),("zh","question_zh")):
    texts.append(p[field]);text_meta.append((p["id"],"predicate",lang))
  tx=tok(texts,padding="max_length",max_length=64,truncation=False,return_tensors="pt").to("cuda")
  with torch.inference_mode():tz=model.get_text_features(**tx).float()
  assert torch.isfinite(tz).all(); ti=torch.nn.functional.normalize(tz,dim=-1)
  image_norm=torch.nn.functional.normalize(z1,dim=-1); logits=(image_norm@ti.T)*model.logit_scale.exp()+model.logit_bias; sims=logits.sigmoid()
  test=dict(image_repeat_max_abs_diff=diff,text_count=len(texts),all_text_max_length=64,padding="max_length",embedding_dim=int(z1.shape[-1]),logit_scale=float(model.logit_scale.exp()),logit_bias=float(model.logit_bias),sample_similarity_min=float(sims.min()),sample_similarity_max=float(sims.max()),mha_fastpath=False,finite=True,estimated_peak_mib=peak,admission_free_mib=initial_free)
  (R/"embedding_test.json").write_text(json.dumps(test,indent=2)+"\n");status("embedding_test_passed",**test)
  # No dev reference labels/stages are loaded until text is locked and test passes.
  specs=json.loads((T/"results/e99b3_20261115_01/sources.json").read_text());devsrc=[s for s in specs if s["pool"]=="dev" and s["cid"]!="piper_4dc107335389f5711a0f65e0"]
  frames={s["parent"]:sorted(set(range(0,s["n"],30))) for s in devsrc}
  causal=json.loads((T/"results/e99b3_20261115_01/dev_labels.json").read_text())
  for r in causal:
   if r["causal_valid"] and r["parent"] in frames:frames[r["parent"]].append(int(r["frame"]))
  frames={p:sorted(set(v)) for p,v in frames.items()}
  cams=("cam_high","cam_left_wrist","cam_right_wrist"); img_by={};nframes=0;peak_alloc=0
  for si,s in enumerate(devsrc):
   fids=frames[s["parent"]]; arr=np.empty((len(fids),3,int(z1.shape[-1])),dtype=np.float32)
   for ci,cam in enumerate(cams):
    decoded=selected_frames(s["videos"][cam],fids);batch=[];ix=[]
    for j,rgb in enumerate(decoded):
     px,_,_=preprocess(rgb);batch.append(px);ix.append(j)
     if len(batch)>=16:
      with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):out=model.get_image_features(pixel_values=torch.stack(batch).cuda()).float()
      assert torch.isfinite(out).all();arr[ix,ci]=out.cpu().numpy();peak_alloc=max(peak_alloc,int(torch.cuda.max_memory_allocated()));batch=[];ix=[]
    if batch:
     with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):out=model.get_image_features(pixel_values=torch.stack(batch).cuda()).float()
     assert torch.isfinite(out).all();arr[ix,ci]=out.cpu().numpy();peak_alloc=max(peak_alloc,int(torch.cuda.max_memory_allocated()))
    del decoded
   img_by[s["parent"]]=(np.array(fids,dtype=np.int32),arr);nframes+=len(fids)
   np.savez_compressed(R/(s["parent"]+"_embeddings.npz"),frames=np.array(fids,dtype=np.int32),features=arr.astype("f2"))
   status("dev_embeddings",parent=si+1,total=len(devsrc),unique_frames=nframes)
  # Persist only compact scores, no source images or copied datasets.
  norm=torch.nn.functional.normalize(torch.tensor(img_by[devsrc[0]["parent"]][1],device="cuda"),dim=-1)
  txtnorm=torch.nn.functional.normalize(tz,dim=-1)
  scale=model.logit_scale.exp().detach();bias=model.logit_bias.detach()
  score_by={}
  ti_map={meta:i for i,meta in enumerate(text_meta)}
  for s in devsrc:
   fids,arr=img_by[s["parent"]];x=torch.nn.functional.normalize(torch.tensor(arr,device="cuda"),dim=-1)
   vals=(x@txtnorm.T)*scale+bias; vals=vals.sigmoid().cpu().numpy()
   # cube frame x camera x text; store all locked variants and predicates
   score_by[s["parent"]]=(fids,vals)
   np.savez_compressed(R/(s["parent"]+"_similarities.npz"),frames=fids,prob=vals.astype("f2"))
   del x,vals
  peak_alloc=max(peak_alloc,int(torch.cuda.max_memory_allocated()));torch.cuda.synchronize()
  elapsed=time.monotonic()-started
  ledger.write_text(json.dumps([dict(gpu=dev_id,initial_free_mib=initial_free,estimated_peak_mib=peak,start_brt="see status",seconds=elapsed,status="inference_complete",peak_allocated_mib=peak_alloc)],indent=2)+"\n")
  status("features_complete",dev_parents=len(devsrc),unique_parent_frames=nframes,elapsed_gpu_s=elapsed,peak_allocated_mib=peak_alloc,disk_bytes=sum(p.stat().st_size for p in R.iterdir() if p.is_file()))
  # All subsequent metric computation is CPU-side.
  torch.cuda.empty_cache();stop.set();th.join(timeout=3)
 except BaseException:
  stop.set();elapsed=time.monotonic()-started
  ledger.write_text(json.dumps([dict(gpu=dev_id,initial_free_mib=initial_free,estimated_peak_mib=peak,seconds=elapsed,status="failed_or_interrupted")],indent=2)+"\n")
  raise
if __name__=="__main__":main()
