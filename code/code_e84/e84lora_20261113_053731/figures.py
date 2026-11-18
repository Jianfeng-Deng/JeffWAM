import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,cv2
from common import *

def main():
 figdir=P/'figures';figdir.mkdir(exist_ok=True);rows={r['id']:r for r in [json.loads(s) for s in (R/'pixel_gt/pixel_gt.jsonl').read_text().splitlines()]};selected=['G002','G027','G103','G126'];roots=[OLDP/'frozen',P/'lora',P/'shuffled'];names=['Frozen','LoRA','Shuffled LoRA'];cache=[]
 for root in roots:
  f=np.load(root/'pixel_patch_text_maps.npz');cache.append({k:f[k] for k in f.files});f.close()
 for word,qi in [('gripper_right',2),('orange_tie',4)]:
  fig,axes=plt.subplots(4,4,figsize=(13,10),constrained_layout=True)
  for ri,uid in enumerate(selected):
   row=rows[uid];im=cv2.cvtColor(cv2.imread(row['decoded_frame']),cv2.COLOR_BGR2RGB);w,h=row['resolution'];mm=[]
   for f in cache:
    i=f['ids'].tolist().index(uid);square=cv2.resize(f['patch_cosine'][i].reshape(14,14,-1),(224,224));mm.append(cv2.warpAffine(square,f['affine'][i,:2],(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)[:,:,qi])
   lo=min(m.min() for m in mm);hi=max(m.max() for m in mm)
   for ci in range(4):
    ax=axes[ri,ci];ax.imshow(im)
    if ci:ax.imshow(mm[ci-1],cmap='magma',alpha=.55,vmin=lo,vmax=hi)
    for key,col in [('L','cyan'),('R','lime'),('HEAD','red'),('TIP','white')]:
     a=row['annotations'][key]
     if a['point_xy'] is not None:ax.plot(*a['point_xy'],'+',color=col,markersize=9)
    ax.set_title(f'{uid} {row["probe_split"]}: '+(['Original']+names)[ci],fontsize=9);ax.axis('off')
  fig.savefig(figdir/f'fixed_{word}_comparison.png',dpi=130);plt.close(fig)
 if os.environ.get('E84_QUAL_ONLY')=='1':return
 fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
 for ai,arm in enumerate(['left','right']):
  for k,(name,root) in enumerate(zip(names,roots)):
   mo=json.loads((root/f'motion_robot_{arm}_h50.json').read_text());be=json.loads((root/f'behavior_{arm}.json').read_text());metrics=[mo['zero_shot']['zh']['difference']['direction_top1'],mo['learned']['contrastive_zh']['metrics']['direction_top1'],be['counterfactual']['language']['follow_rate']]
   for ax,m in zip(axes,metrics):
    x=ai*4+k;e=m['estimate']*100;lo,hi=np.array(m['ci95'])*100;ax.errorbar(x,e,yerr=np.array([[e-lo],[hi-e]]),fmt='o',capsize=3,color=['#555555','#087f8c','#c48b2c'][k],label=name if ai==0 else None)
 for ax,title in zip(axes,['H50 zero-shot direction','H50 contrastive-head direction','Counterfactual follow rate']):
  ax.set_title(title);ax.set_xticks([1,5],['Left','Right']);ax.set_ylabel('% (equal-parent 95% CI)');ax.axhline(12.5,color='gray',ls='--',lw=.8);ax.grid(axis='y',alpha=.2)
 axes[0].legend(fontsize=8);fig.savefig(figdir/'primary_comparison.png',dpi=150);plt.close(fig);print('FIGURES_READY',figdir)
def confusion_figure():
 figdir=P/'figures';figdir.mkdir(exist_ok=True);fig,axes=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
 labels=['F','FL','L','BL','B','BR','R','FR']
 for ri,arm in enumerate(['left','right']):
  frozen=json.loads((OLDP/'frozen'/f'motion_robot_{arm}_h50.json').read_text());lo=json.loads((P/'lora'/f'motion_robot_{arm}_h50.json').read_text());sh=json.loads((P/'shuffled'/f'motion_robot_{arm}_h50.json').read_text())
  cells=[('Frozen zero',frozen['zero_shot']['zh']['difference']),('LoRA zero',lo['zero_shot']['zh']['difference']),('LoRA+contrastive head',lo['learned']['contrastive_zh']['metrics']),('Shuffled-LoRA zero',sh['zero_shot']['zh']['difference'])]
  for ci,(title,m) in enumerate(cells):
   z=np.asarray(m['direction_confusion'],dtype=float);z=z/np.maximum(z.sum(1,keepdims=True),1);ax=axes[ri,ci];im=ax.imshow(z,vmin=0,vmax=1,cmap='Blues');ax.set_title(arm+' '+title,fontsize=9);ax.set_xticks(range(8),labels,fontsize=7);ax.set_yticks(range(8),labels,fontsize=7);ax.set_xlabel('Predicted');ax.set_ylabel('True own-base direction')
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='Per-sample row-normalized fraction');fig.savefig(figdir/'h50_direction_confusions.png',dpi=140);plt.close(fig)

if __name__=='__main__':
 main()
 if os.environ.get('E84_QUAL_ONLY')!='1':confusion_figure()
