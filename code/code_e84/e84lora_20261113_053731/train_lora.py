import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import sys,time,traceback,numpy as np,torch,cv2,hashlib
sys.dont_write_bytecode=True
import torch.nn.functional as F
from common import *
from encoder import Encoder,WORDS,CONTROLS
from evaluation_data import candidate_texts
from lora_module import inject,adapter_state
from resource_guard import Guard
from contract import sha
torch.set_num_threads(2);cv2.setNumThreads(1)
STEPS=768;SEED=20261112

def text_features(enc,texts):
 inp=enc.tok([s.lower() for s in texts],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda:0');t=enc.model.get_text_features(**inp)
 if not isinstance(t,torch.Tensor):t=t.pooler_output
 return F.normalize(t.float(),dim=-1)
def patch_project(model,h):
 head=model.vision_model.head;att=head.attention;d=h.shape[-1];v=F.linear(h,att.in_proj_weight[2*d:],None if att.in_proj_bias is None else att.in_proj_bias[2*d:]);v=att.out_proj(v)
 return F.normalize(v+head.mlp(head.layernorm(v)),dim=-1)
def swap(s):return s.replace('right','LEFT').replace('left','right').replace('LEFT','left')
def main():
 items=json.loads((L/'training_items.json').read_text());pixels=np.load(L/'train_pixels.npy',mmap_mode='r');teacher=np.load(L/'teacher_globals.npy');ready=json.loads((L/'READY_training_data.json').read_text());assert ready['selection_sha256']==sha(P/'selection_full_v4.json')
 fkids=np.array([i for i,r in enumerate(items) if r['kind']=='fk']);eids=np.array([i for i,r in enumerate(items) if r['kind']=='e83']);labels=np.array([r.get('label',0) for r in items]);perm_labels=labels.copy();perm_labels[fkids]=labels[np.random.default_rng(SEED).permutation(fkids)];ecaptions={i:r['text'] for i,r in enumerate(items) if r['kind']=='e83'};epers=np.random.default_rng(SEED+1).permutation(eids);shuffle_caption=dict(zip(eids,[items[i]['text'] for i in epers]));assert len(eids)>0
 base_texts=sum([candidate_texts(a,'robot',lang) for lang in ['zh','en'] for a in ['left','right']],[])+[f'This is a photo of the {s}.' for s in WORDS+CONTROLS]
 guard=Guard();started=time.time();globals()['_guard']=guard
 for variant in ['lora','shuffled']:
  out=L/variant;out.mkdir(exist_ok=True)
  if (out/'training_result.json').exists():continue
  torch.manual_seed(SEED);enc=Encoder('cuda:0');base_static=torch.as_tensor(enc.text(base_texts),device='cuda:0').clone();base_e83={}
  for i in eids:
   text=shuffle_caption[i] if variant=='shuffled' else ecaptions[i]
   if text not in base_e83:base_e83[text]=enc.text([text])[0]
  config=inject(enc.model,rank=8,layers=4);enc.model.eval();parameters=[p for p in enc.model.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=.01)
  atomic(out/'training_label_assignment.json',[dict(index=i,kind=r['kind'],parent=r['parent'],canonical_id=r['canonical_id'],anchor_source_row=r['anchor_source_row'],assigned_label=int((perm_labels if variant=='shuffled' else labels)[i]) if r['kind']=='fk' else None,assigned_text=(shuffle_caption[i] if variant=='shuffled' else ecaptions[i]) if r['kind']=='e83' else None) for i,r in enumerate(items)])
  ckpt=out/'resume.pt';step0=0;logs=[]
  if ckpt.exists():
   ck=torch.load(ckpt,map_location='cuda:0',weights_only=False);enc.model.load_state_dict(ck['adapter'],strict=False);opt.load_state_dict(ck['optimizer']);step0=ck['step'];logs=ck['logs']
  atomic(out/'training_manifest.json',dict(timestamp_brt=brt(),variant=variant,config=config,seed=SEED,steps=STEPS,batch=4,lr=1e-5,drift_weight=.1,hard_weight=.25,color_weight=.25,e83_weight=.5,e83_every_n_steps=8,strict_e83_exact_anchor=True,train_parents=ready['parents'],fk_pairs=len(fkids),e83_pairs=len(eids),selection_sha256=ready['selection_sha256'],selection_rule='up to4 FK pairs per parent per arm, up to2 existing E83 captions per train parent, no val selection',checkpoint='fixed final step768; no val/GT threshold or checkpoint use',control='same initialization/order/steps; FK labels permuted across FK rows, E83 captions permuted across E83 rows. Orange HSV pseudo-label invariant under permutation (all orange), reported limitation',color='deterministic HSV orange membership only, not accepted instance GT',objective='normalized mean(top+own wrist) end-start -> 48 FK candidates each zh/en; E83 view-mean -> caption; left/right reversal and orange/blue/green hard negatives; cosine drift to cached frozen globals and original text'))
  for step in range(step0,STEPS):
   rng=np.random.default_rng(SEED+step);n_e=1 if len(eids) and step%8==0 else 0;ids=np.r_[rng.choice(fkids,4-n_e,replace=False),rng.choice(eids,n_e,replace=False)];xx=torch.from_numpy(pixels[ids].copy()).to('cuda:0',dtype=torch.float32)/127.5-1;tt=torch.as_tensor(teacher[ids],device='cuda:0');opt.zero_grad(set_to_none=True)
   captions=[(shuffle_caption[i] if variant=='shuffled' else ecaptions[i]) for i in ids if items[i]['kind']=='e83'];ctext=sum([[s,swap(s),s.replace('orange','blue'),s.replace('orange','green')] for s in captions],[]);texts=base_texts+ctext
   static_teacher=torch.cat([base_static,torch.as_tensor(np.stack([base_e83[s] for s in captions for _ in range(4)]),device='cuda:0')],0) if captions else base_static
   with torch.autocast('cuda',dtype=torch.bfloat16):
    vision=enc.model.vision_model(pixel_values=xx.flatten(0,1));v=vision.pooler_output.float().reshape(len(ids),4,-1);t=text_features(enc,texts)
    n_fk=4-n_e;start=F.normalize(v[:n_fk,:2].mean(1),dim=-1);end=F.normalize(v[:n_fk,2:].mean(1),dim=-1);delta=F.normalize(end-start,dim=-1);truth=torch.as_tensor((perm_labels if variant=='shuffled' else labels)[ids[:n_fk]],device='cuda:0')
    ce=0.;hard=0.
    for offset in [0,48]:
     logits=delta@t[offset:offset+48].T/.07;ce=ce+F.cross_entropy(logits,truth)/2
     cls=truth%24;arm=truth//24;reverse=arm*24+((cls//3+4)%8)*3+cls%3;sw=(1-arm)*24+cls;positive=logits.gather(1,truth[:,None]).squeeze(1)
     hard=hard+(F.relu(.5+logits.gather(1,reverse[:,None]).squeeze(1)-positive)+F.relu(.5+logits.gather(1,sw[:,None]).squeeze(1)-positive)).mean()/4
    e83=v.sum()*0
    if n_e:
     for ei,caption in enumerate(captions):
      ev=F.normalize(v[n_fk+ei,:3].mean(0),dim=-1);bank=torch.cat([t[96:109],t[109+ei*4:109+(ei+1)*4]],0);sim=ev@bank.T/.07
      # Identical substitutions are duplicate positives, not false negatives.
      banktexts=base_texts[96:109]+ctext[ei*4:(ei+1)*4];positive=torch.tensor([s==caption for s in banktexts],device='cuda:0');e83=e83+torch.logsumexp(sim,0)-torch.logsumexp(sim[positive],0)
    masks=[]
    for pi in range(n_fk):
     rgb=pixels[ids[pi],0].transpose(1,2,0).copy();hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);m=((hsv[:,:,0]>=3)&(hsv[:,:,0]<=25)&(hsv[:,:,1]>=90)&(hsv[:,:,2]>=60)).astype('f4');masks.append(cv2.resize(m,(14,14),interpolation=cv2.INTER_AREA).ravel())
    mask=torch.as_tensor(np.stack(masks),device='cuda:0');valid=mask.sum(-1)>=.5;color=v.sum()*0
    if valid.any():
     h=vision.last_hidden_state.float().reshape(4,4,196,768)[:n_fk,0];p=patch_project(enc.model,h);pooled=F.normalize((p*mask[:,:,None]).sum(1)/mask.sum(-1,keepdim=True).clamp_min(1e-6),dim=-1);color=F.cross_entropy(pooled[valid]@t[[100,103,104]].T/.07,torch.zeros(int(valid.sum()),dtype=torch.long,device='cuda:0'))
    vd=(1-F.cosine_similarity(v,tt,dim=-1)).mean();td=(1-(t[:109]*base_static).sum(-1)).mean();loss=ce+.25*hard+.25*color+.5*e83+.1*(vd+td)
   assert torch.isfinite(loss),'nonfinite loss';loss.backward();
   if step==0:
    audit={tower:dict(nonzero_parameters=sum(bool(p.grad is not None and p.grad.abs().sum()>0) for n,p in enc.model.named_parameters() if n.startswith(tower) and p.requires_grad),trainable_parameters=sum(p.numel() for n,p in enc.model.named_parameters() if n.startswith(tower) and p.requires_grad)) for tower in ['vision_model','text_model']};assert all(z['nonzero_parameters']>0 for z in audit.values());atomic(out/'gradient_audit.json',audit)
   torch.nn.utils.clip_grad_norm_(parameters,1);opt.step()
   row=dict(step=step+1,loss=float(loss.detach()),contrastive=float(ce.detach()),hard=float(hard.detach()),color=float(color.detach()),e83=float(e83.detach()),visual_cosine_deviation=float(vd.detach()),text_cosine_deviation=float(td.detach()));logs.append(row)
   if (step+1)%16==0 or step==step0:
    guard.check();torch.save(dict(adapter=adapter_state(enc.model),optimizer=opt.state_dict(),step=step+1,logs=logs),str(ckpt)+'.tmp');os.replace(str(ckpt)+'.tmp',ckpt);atomic(out/'progress.json',dict(timestamp_brt=brt(),status='training',step=step+1,steps=STEPS,last=row));update(status='training_'+variant,variant=variant,step=step+1)
   if (step+1)%64==0:print('TRAIN',variant,step+1,round(row['loss'],4),'GPU elapsed',round(time.time()-started),flush=True)
  torch.save(adapter_state(enc.model),out/'adapter.pt');atomic(out/'training_result.json',dict(status='completed_fixed768',timestamp_brt=brt(),variant=variant,steps=STEPS,config=config,adapter_sha256=sha(out/'adapter.pt'),logs=logs));del enc,opt,parameters,vision,v,t,loss;torch.cuda.empty_cache()
 update(status='both_adapters_trained');print('TRAINING_DONE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(P/'training_failure.json',dict(timestamp_brt=brt(),traceback=traceback.format_exc()));update(status='training_failed');raise
 finally:
  if globals().get('_guard'):globals()['_guard'].finish()
