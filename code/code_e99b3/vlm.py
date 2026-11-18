from common import *
import numpy as np
MODEL='/mnt/workspace/dengjianfeng/models/qwen3_vl_4b'
def main(mode):
 guard=Guard('qwen_'+mode,20)
 try:
  import torch,cv2
  from PIL import Image
  from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
  torch.set_num_threads(2);cv2.setNumThreads(1)
  proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True,min_pixels=56*56,max_pixels=448*448);tok=proc.tokenizer;tok.padding_side='left'
  m=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();m.requires_grad_(False)
  spec=read('spec.json');questions=[p['question_en'] for p in spec['predicates']];pids=[p['id'] for p in spec['predicates']]
  template=read('PROTOCOL_LOCK.json')['prompt'];strings=[]
  for q in questions:
   msg=[dict(role='user',content=[dict(type='image') for _ in CAMS]+[dict(type='text',text=template.format(question_en=q))])]
   strings.append(proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True))
  raw=[tok(s,add_special_tokens=False)['input_ids'] for s in strings];common=0
  while common<min(map(len,raw)) and len({x[common] for x in raw})==1:common+=1
  assert common>10
  yes=tok.encode('yes',add_special_tokens=False);no=tok.encode('no',add_special_tokens=False);assert len(yes)==len(no)==1
  def predict(images,audit=False):
   inp=proc(text=[strings[0]],images=images,return_tensors='pt').to('cuda');base=inp['input_ids'].shape[1]-len(raw[0])+common
   assert inp['input_ids'][0,base:].tolist()==raw[0][common:]
   with torch.inference_mode():
    if audit:full=m(**inp,use_cache=False,logits_to_keep=1).logits[0,-1,[no[0],yes[0]]].float().cpu().numpy()
    prefix={k:v for k,v in inp.items()}
    for k in ('input_ids','attention_mask','mm_token_type_ids'):
     if k in prefix:prefix[k]=prefix[k][:,:base]
    m.model.rope_deltas=None
    first=m(**prefix,use_cache=True,logits_to_keep=1);cache=first.past_key_values;delta=int(first.rope_deltas.reshape(-1)[0]);cache.batch_repeat_interleave(len(raw));del first
    suffix=tok.pad(dict(input_ids=[r[common:] for r in raw]),padding=True,return_tensors='pt').to('cuda');mask=suffix['attention_mask'];pos=mask.cumsum(-1)-1+base+delta;pos=pos.clamp_min(0)[None].expand(3,-1,-1)
    attention=torch.cat([torch.ones(len(raw),base,device='cuda',dtype=mask.dtype),mask],1)
    out=m(input_ids=suffix['input_ids'],attention_mask=attention,position_ids=pos,past_key_values=cache,use_cache=False,logits_to_keep=1);logits=out.logits[:,-1,[no[0],yes[0]]].float();p=logits.softmax(-1)[:,1].cpu().numpy();l=logits.cpu().numpy();del out,cache
   assert np.isfinite(l).all()
   if audit:
    gap=abs(float((l[0,1]-l[0,0])-(full[1]-full[0])));assert gap<=.25,(gap,l[0].tolist(),full.tolist())
    return p,l,dict(prefix_length=base,rope_delta=delta,full_vs_prefix_yes_no_logit_gap_error=gap,full_logits=full.tolist(),cached_logits=l[0].tolist(),yes_id=yes[0],no_id=no[0],image_grid=inp['image_grid_thw'].cpu().tolist())
   return p,l
  if mode=='smoke':
   s=next(s for s in read('sources.json') if s['pool']=='train');images=[]
   for cam in CAMS:
    cap=cv2.VideoCapture(s['videos'][cam]);ok,bgr=cap.read();cap.release();assert ok;images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
   start=time.monotonic();_,_,audit=predict(images,True);audit.update(brt=now(),wall_seconds=time.monotonic()-start,model=MODEL,transformers=__import__('transformers').__version__,prompt=template,current_frame_only=True);put('VLM_SMOKE.json',audit);report('## VLM 环境与缓存合同 '+now()+'\n'+json.dumps(audit,ensure_ascii=False)+'\n首帧 prefix KV 与完整原提示等价性在训练 parent 上核验；未以 dev 调提示。')
  else:
   assert (R/'DEV_ONCE_LOCK.json').exists() and (R/'VLM_SMOKE.json').exists()
   for si,s in enumerate([s for s in read('sources.json') if s['pool']=='dev']):
    frames=s['annotation_frames'];caps=[cv2.VideoCapture(s['videos'][cam]) for cam in CAMS];assert all(c.isOpened() for c in caps);cursor=-1;rows=[];logs=[]
    for fi,f in enumerate(frames):
     guard.check()
     while cursor<f:
      for cap in caps:assert cap.grab()
      cursor+=1
     images=[]
     for cap in caps:
      ok,bgr=cap.retrieve();assert ok;images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
     p,l=predict(images);rows.append(p);logs.append(l)
     if fi%20==0:status('qwen_dev',parent=si+1,parents=20,frame=fi,total_frames=len(frames),gpu_seconds=guard.used+time.monotonic()-guard.started)
    for cap in caps:cap.release()
    np.savez_compressed(R/(s['parent']+'_vlm.npz'),frames=frames,prob=np.array(rows),logits=np.array(logs));status('qwen_parent_complete',parent=si+1,parents=20)
   put('VLM_COMPLETE.json',dict(brt=now(),frames=sum(len(s['annotation_frames']) for s in read('sources.json') if s['pool']=='dev'),predicates=21,all_scores_once=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
 finally:guard.finish()
if __name__=='__main__':main(sys.argv[1])
