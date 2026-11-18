import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import sys
sys.dont_write_bytecode=True
import numpy as np,collections
from common import *
from metrics import parent_ci,paired_ci,normalize,planar_follow
from encoder import Encoder,WORDS,CONTROLS
from evaluation_data import candidate_texts
from contract import sha

def cka(a,b):
 a=a-a.mean(0);b=b-b.mean(0);cross=a.T@b
 return float((cross*cross).sum()/np.sqrt(((a.T@a)**2).sum()*((b.T@b)**2).sum()))
def drift():
 sel=json.loads((P/'selection_full_v4.json').read_text());by=collections.defaultdict(list)
 for r in sel['samples']:by[r['canonical_id']].append(r)
 chosen=[]
 # fixed hash one anchor per capture; all3 views and observed endpoints, no GT use.
 import hashlib
 for cid,rows in sorted(by.items()):
  r=min(enumerate(rows),key=lambda ir:hashlib.sha256(str((cid,ir[1]['anchor_source_row'])).encode()).digest());chosen.append((cid,r[0],r[1]))
 basetext=sum([candidate_texts(a,'robot',lang) for lang in ['zh','en'] for a in ['left','right']],[])+[f'This is a photo of the {s}.' for s in WORDS+CONTROLS]
 enc=Encoder('cpu');bt=enc.text(basetext);del enc
 result={}
 for v in ['lora','shuffled']:
  aa=[];bb=[];ps=[];split=[]
  for cid,i,r in chosen:
   f=np.load(R/'features'/OLD/'frozen'/(cid+'.npz'));g=np.load(R/'features'/RUN/v/(cid+'.npz'));valid=f['history_valid'][i]
   aa.append(f['global_features'][i][valid].astype('f4'));bb.append(g['global_features'][i][valid].astype('f4'));n=int(valid.sum());ps.extend([r['parent_component']]*n);split.extend([r['probe_split']]*n);f.close();g.close()
  a=normalize(np.concatenate(aa));b=normalize(np.concatenate(bb));ps=np.array(ps);split=np.array(split);cos=(a*b).sum(-1);enc=Encoder('cpu',str(L/v/'adapter.pt'));t=enc.text(basetext);del enc
  obj={'visual':{},'text_cosine':float((bt*t).sum(-1).mean()),'text_cosine_min':float((bt*t).sum(-1).min()),'text_linear_CKA':cka(bt,t),'text_count':len(t),'generic_retrieval':'N/A: no supplied local generic image-text retrieval sample; no downloads; Piper in-domain drift cannot certify general retention'}
  for s in ['probe-train','probe-val']:
   keep=split==s;obj['visual'][s]={'cosine':parent_ci(cos[keep],ps[keep]),'linear_CKA':cka(a[keep],b[keep]),'instances':int(keep.sum())}
  result[v]=obj
 atomic(P/'drift_results.json',dict(timestamp_brt=brt(),selection='one fixed-hash selected anchor per capture, all valid current/history camera globals',results=result))

def compare():
 results={'motion':{},'behavior':{},'pixel':{},'timestamp_brt':brt()}
 for arm in ['left','right']:
  for h in [50,10]:
   stem=f'motion_robot_{arm}_h{h}';f=np.load(OLDP/'frozen'/(stem+'_predictions.npz'));truth=f['labels'];parents=f['parents'];allsc={'frozen_zero':f['zero_difference_zh'],'frozen_linear':f['linear'],'frozen_contrastive':f['contrastive_zh'],'random':f['random'],'majority':f['majority']}
   for v in ['lora','shuffled']:
    a=np.load(P/v/(stem+'_predictions.npz'));assert np.array_equal(truth,a['labels']) and np.array_equal(parents,a['parents']);assert np.array_equal(f['feature_index'],a['feature_index'])
    for name in ['zero_difference_zh','linear','contrastive_zh','contrastive_zh_shuffled']:allsc[v+'_'+name]=a[name]
   if h==50:
    m=np.load(P/'matched_frozen'/(stem+'_predictions.npz'));assert np.array_equal(m['labels'],truth) and np.array_equal(m['parents'],parents);allsc['matched_frozen']=m['true']
   gaps={}
   comparisons=[('lora_zero_difference_zh','random'),('lora_zero_difference_zh','majority'),('lora_contrastive_zh','lora_contrastive_zh_shuffled'),('lora_zero_difference_zh','frozen_zero'),('lora_zero_difference_zh','shuffled_zero_difference_zh'),('lora_linear','frozen_linear'),('lora_contrastive_zh','frozen_contrastive'),('lora_contrastive_zh','shuffled_contrastive_zh')]
   if h==50:comparisons += [('lora_zero_difference_zh','matched_frozen'),('lora_contrastive_zh','matched_frozen')]
   for a,b in comparisons:
    pa=allsc[a].argmax(-1);pb=allsc[b].argmax(-1);gaps[a+' minus '+b]={}
    for metric,aa,bb in [('top1',pa==truth,pb==truth),('direction_top1',pa//3==truth//3,pb//3==truth//3),('amplitude_top1',pa%3==truth%3,pb%3==truth%3)]:gaps[a+' minus '+b][metric]=paired_ci(aa,bb,parents)
    ta=np.argsort(allsc[a],axis=-1)[:,-3:];tb=np.argsort(allsc[b],axis=-1)[:,-3:];gaps[a+' minus '+b]['top3']=paired_ci((ta==truth[:,None]).any(-1),(tb==truth[:,None]).any(-1),parents)
    ea=np.minimum((pa//3-truth//3)%8,(truth//3-pa//3)%8)*45;eb=np.minimum((pb//3-truth//3)%8,(truth//3-pb//3)%8)*45;gaps[a+' minus '+b]['angle_error_reduction_deg']=paired_ci(eb,ea,parents)
   results['motion'][arm+'_h'+str(h)]=gaps
  stem=f'behavior_{arm}_predictions.npz';f=np.load(OLDP/'frozen'/stem);parents=f['parents'];banks={'frozen':f['cf_language']}
  for v in ['lora','shuffled']:
   a=np.load(P/v/stem);assert np.array_equal(parents,a['parents']) and np.array_equal(f['feature_index'],a['feature_index']);assert np.array_equal(f['true_displacement'],a['true_displacement']);banks[v]=a['cf_language']
  dirs=np.stack([np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)],-1);gaps={}
  for a,b in [('lora','frozen'),('lora','shuffled')]:
   aa=banks[a][:,:,:2];bb=banks[b][:,:,:2];ps=np.repeat(parents,8)
   # Preserve the inherited nonzero-only cosine estimand, then bootstrap paired parent means.
   ca=(normalize(aa)*dirs[None]).sum(-1).ravel();cb=(normalize(bb)*dirs[None]).sum(-1).ravel();ca[np.linalg.norm(aa,axis=-1).ravel()<=.001]=np.nan;cb[np.linalg.norm(bb,axis=-1).ravel()<=.001]=np.nan
   up=np.unique(ps);ma=np.array([np.nanmean(ca[ps==p]) for p in up]);mb=np.array([np.nanmean(cb[ps==p]) for p in up]);cosgap=paired_ci(ma,mb,up)
   gaps[a+' minus '+b]={'follow_rate':paired_ci(planar_follow(aa,dirs[None],.001).ravel(),planar_follow(bb,dirs[None],.001).ravel(),ps),'prediction_instruction_cosine':cosgap}
  results['behavior'][arm]=gaps
 keys=['id','parent','split','stratum','word','quality','radius_multiplier','condition']
 def readpixel(root):
  rows=[json.loads(x) for x in (root/'pixel_per_target.jsonl').read_text().splitlines()];return {tuple(r[k] for k in keys):r for r in rows if r['split']=='probe-val' and r['quality']=='main'}
 pixels={'frozen':readpixel(OLDP/'frozen'),**{v:readpixel(P/v) for v in ['lora','shuffled']}}
 assert all(set(x)==set(pixels['frozen']) for x in pixels.values())
 for word in WORDS:
  results['pixel'][word]={}
  for radius in [1,2]:
   rr=[r for k,r in pixels['frozen'].items() if r['word']==word and r['radius_multiplier']==radius and r['condition']=='correct'];ps=[r['parent'] for r in rr];ks=[tuple(r[k] for k in keys) for r in rr];out={}
   for a,b in [('lora','frozen'),('lora','shuffled')]:
    out[a+' minus '+b]={metric:paired_ci([pixels[a][k][metric] for k in ks],[pixels[b][k][metric] for k in ks],ps) for metric in ['pointing','auroc']}
   results['pixel'][word][str(radius)]=out
 # Difference-in-differences: semantic contrast gap itself, not merely absolute localization.
 semantic={}
 for word in WORDS:
  rr=[r for r in pixels['frozen'].values() if r['word']==word and r['radius_multiplier']==1 and r['condition']=='correct'];ps=[r['parent'] for r in rr];ks=[tuple(r[k] for k in keys) for r in rr];out={}
  controls=['left_right_swap'] if ('arm' in word or 'gripper' in word) else ['blue','green'] if word=='orange zip tie' else []
  for control in controls:
   ck=[tuple(control if name=='condition' else value for name,value in zip(keys,k)) for k in ks]
   if not ck:continue
   out[control]={metric:paired_ci([pixels['lora'][k][metric]-pixels['lora'][c][metric] for k,c in zip(ks,ck)],[pixels['frozen'][k][metric]-pixels['frozen'][c][metric] for k,c in zip(ks,ck)],ps) for metric in ['pointing','auroc']}
  if out:semantic[word]=out
 results['semantic_contrast_gain']=semantic
 atomic(P/'paired_comparisons.json',results)
if __name__=='__main__':drift();compare();print('ANALYSIS_COMPLETE',flush=True)
