# E113-V: Qwen3-VL-4B zero-shot predicate answers with history frames and detailed prompts. Self-run, no Codex.
import os,sys,json,time,datetime,subprocess,collections
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2")
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");B3=T/"results/e99b3_20261115_01";R=T/"results/e113v_vlm_multiframe_20261116_01";MODEL="/mnt/workspace/dengjianfeng/models/qwen3_vl_4b"
CAMS=("cam_high","cam_left_wrist","cam_right_wrist");CAMNAME={"cam_high":"overhead","cam_left_wrist":"left wrist","cam_right_wrist":"right wrist"}
FPS=30;OFFS=[60,15]  # history: 2.0 s and 0.5 s before the current frame (frame index = source frame, nominal 30 Hz)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def jread(p):return json.loads(Path(p).read_text())
def put(name,x):
 p=R/name;tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(x,ensure_ascii=False,indent=1,allow_nan=False)+"\n");os.replace(tmp,p)
spec=jread(B3/"spec.json");PRED=spec["predicates"];PIDS=[p["id"] for p in PRED]
ORIG=jread(B3/"PROTOCOL_LOCK.json")["prompt"]
INSTR="Answer the following question about the current zip tie. Output exactly yes or no. If uncertain, give your best binary judgment."
def tail(variant,p):
 q=p["question_en"]
 if variant in ("V0","V1"):return "Question: "+q
 return ("Question: "+q+"\nMost informative camera for this question: "+CAMNAME[p["primary_camera"]]+" (the other views give context)."
  +"\nWhat to look for (decision evidence, in Chinese): "+p["decision_evidence_zh"]
  +"\nEasily confused with (in Chinese): "+p["common_confusions_zh"])
def content(variant):
 hist=variant in ("V1","V3");c=[]
 if not hist:
  c+=[dict(type="image") for _ in CAMS]
  head=ORIG.split("Question: {question_en}")[0] if variant=="V0" else "Three synchronized views at the current source frame: overhead, left wrist, right wrist. "+INSTR+" "
  return c,head
 labels=["2.0 s ago","0.5 s ago","current"]
 for lab in labels:
  c.append(dict(type="text",text="Views "+lab+" (overhead, left wrist, right wrist):"));c+=[dict(type="image") for _ in CAMS]
 head="The nine images are three synchronized time points (2.0 s ago, 0.5 s ago, current), each with overhead, left wrist and right wrist views, in that order. Use the earlier frames to judge motion and contact (for example whether the strap moves with a gripper). "+INSTR+" "
 return c,head
def main(mode):
 import numpy as np,torch,cv2
 from PIL import Image
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 gpu=os.environ.get("CUDA_VISIBLE_DEVICES","?");torch.set_num_threads(2);cv2.setNumThreads(1)
 proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True,min_pixels=56*56,max_pixels=448*448);tok=proc.tokenizer;tok.padding_side="left"
 m=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval();m.requires_grad_(False)
 yes=tok.encode("yes",add_special_tokens=False);no=tok.encode("no",add_special_tokens=False);assert len(yes)==len(no)==1
 V={}
 for v in ["V0","V1","V2","V3"]:
  c,head=content(v);strings=[]
  for p in PRED:
   msg=[dict(role="user",content=c+[dict(type="text",text=head+tail(v,p))])];strings.append(proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True))
  raw=[tok(s,add_special_tokens=False)["input_ids"] for s in strings];common=0
  while common<min(map(len,raw)) and len({x[common] for x in raw})==1:common+=1
  V[v]=dict(strings=strings,raw=raw,common=common)
 def predict(v,images,audit=False):
  S=V[v];raw=S["raw"];common=S["common"]
  inp=proc(text=[S["strings"][0]],images=images,return_tensors="pt").to("cuda");base=inp["input_ids"].shape[1]-len(raw[0])+common
  assert inp["input_ids"][0,base:].tolist()==raw[0][common:]
  with torch.inference_mode():
   if audit:full=m(**inp,use_cache=False,logits_to_keep=1).logits[0,-1,[no[0],yes[0]]].float().cpu().numpy()
   prefix={k:x for k,x in inp.items()}
   for k in ("input_ids","attention_mask","mm_token_type_ids"):
    if k in prefix:prefix[k]=prefix[k][:,:base]
   m.model.rope_deltas=None
   first=m(**prefix,use_cache=True,logits_to_keep=1);cache=first.past_key_values;delta=int(first.rope_deltas.reshape(-1)[0]);cache.batch_repeat_interleave(len(raw));del first
   suffix=tok.pad(dict(input_ids=[r[common:] for r in raw]),padding=True,return_tensors="pt").to("cuda");mask=suffix["attention_mask"];pos=(mask.cumsum(-1)-1+base+delta).clamp_min(0)[None].expand(3,-1,-1)
   attention=torch.cat([torch.ones(len(raw),base,device="cuda",dtype=mask.dtype),mask],1)
   out=m(input_ids=suffix["input_ids"],attention_mask=attention,position_ids=pos,past_key_values=cache,use_cache=False,logits_to_keep=1);lg=out.logits[:,-1,[no[0],yes[0]]].float();p=lg.softmax(-1)[:,1].cpu().numpy();l=lg.cpu().numpy();del out,cache
  assert np.isfinite(l).all()
  if audit:
   gap=abs(float((l[0,1]-l[0,0])-(full[1]-full[0])));return p,dict(prefix_len=base,gap=gap,images=int(inp["image_grid_thw"].shape[0]))
  return p
 def run_parent(s,frames,variants,log):
  caps=[cv2.VideoCapture(s["videos"][c]) for c in CAMS];assert all(c.isOpened() for c in caps);fps=[c.get(cv2.CAP_PROP_FPS) for c in caps]
  buf=[{} for _ in CAMS];cursor=-1;out={v:[] for v in variants}
  for f in sorted(frames):
   while cursor<f:
    for k,cap in enumerate(caps):
     ok,bgr=cap.read();assert ok;buf[k][cursor+1]=bgr;buf[k].pop(cursor+1-70,None)
    cursor+=1
   def img(k,i):return Image.fromarray(cv2.cvtColor(buf[k][max(0,i)] if max(0,i) in buf[k] else buf[k][min(buf[k])],cv2.COLOR_BGR2RGB))
   cur=[img(k,f) for k in range(3)];hist=[[img(k,f-o) for k in range(3)] for o in OFFS]
   for v in variants:
    images=cur if v in ("V0","V2") else sum(hist,[])+cur
    out[v].append(predict(v,images))
   log(f)
  for c in caps:c.release()
  return {v:np.array(x) for v,x in out.items()},fps
 if mode=="smoke":
  s=next(x for x in jread(B3/"sources.json") if x["pool"]=="train");res={}
  caps=[cv2.VideoCapture(s["videos"][c]) for c in CAMS];fr=[]
  for cap in caps:
   for _ in range(70):ok,bgr=cap.read()
   fr.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
  t0=time.monotonic()
  for v in ["V0","V1","V2","V3"]:
   images=fr if v in ("V0","V2") else fr*3;t1=time.monotonic();p,a=predict(v,images,True);a["seconds"]=time.monotonic()-t1;a["common"]=V[v]["common"];res[v]=a
  res["fps"]=[c.get(cv2.CAP_PROP_FPS) for c in caps];res["gpu"]=gpu;res["peak_mib"]=torch.cuda.max_memory_allocated()/2**20;res["brt"]=now();put("SMOKE.json",res);print(json.dumps(res))
  return
 if mode=="train":
  causal=collections.defaultdict(dict);src=jread(B3/"sources.json");cid2p={x["cid"]:x["parent"] for x in src}
  for l in open(T/"data/e98_predicate_labels_train_causal_v1/predicates_causal.jsonl"):
   r=json.loads(l);p=cid2p.get(r["canonical_id"])
   if p is not None and r["causal_observable_answer"] in ("yes","no"):causal[p,r["source_frame"]][r["predicate"]]=r["causal_observable_answer"]=="yes"
  vals=[x for x in src if x["pool"]=="train" and x["validation"]];rng=__import__("numpy").random.default_rng(11300)
  moments=[(x["parent"],f) for x in vals for f in x["annotation_frames"] if (x["parent"],f) in causal];idx=rng.permutation(len(moments))[:300];chosen=sorted(moments[i] for i in idx)
  put("TRAIN_SUBSET_LOCK.json",dict(brt=now(),n=len(chosen),validation_parents=len(vals),source="E98-C causal first-pass labels (model-generated, non-human), 12 validation parents",moments=chosen,variants=["V0","V1","V2","V3"],selection="highest macro AUROC over 21 predicates, ties to simpler variant"))
  by=collections.defaultdict(list)
  for p,f in chosen:by[p].append(f)
  t0=time.monotonic()
  for pi,x in enumerate(vals):
   frames=sorted(by[x["parent"]])
   if not frames:continue
   o,fps=run_parent(x,frames,["V0","V1","V2","V3"],lambda f:None);np_=__import__("numpy");np_.savez_compressed(R/(x["parent"]+"_train.npz"),frames=np_.array(frames),**o);print("train parent",pi+1,len(frames),"elapsed",time.monotonic()-t0,flush=True)
  put("TRAIN_DONE.json",dict(brt=now(),seconds=time.monotonic()-t0,gpu=gpu))
 if mode=="dev":
  lock=jread(R/"VARIANT_LOCK.json");v=lock["variant"];dl=jread(B3/"dev_labels.json");need=collections.defaultdict(set)
  for r in dl:
   if r["causal_valid"]:need[r["parent"]].add(r["frame"])
  t0=time.monotonic();np_=__import__("numpy")
  for pi,x in enumerate([x for x in jread(B3/"sources.json") if x["pool"]=="dev"]):
   frames=sorted(need[x["parent"]])
   o,fps=run_parent(x,frames,[v],lambda f:None);np_.savez_compressed(R/(x["parent"]+"_dev_"+v+".npz"),frames=np_.array(frames),prob=o[v]);print("dev parent",pi+1,len(frames),"elapsed",time.monotonic()-t0,flush=True)
  put("DEV_DONE.json",dict(brt=now(),variant=v,seconds=time.monotonic()-t0))
if __name__=="__main__":main(sys.argv[1])
