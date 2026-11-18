import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,json
from common import *
root=P/'frozen';dest=root/'figures';dest.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,3,figsize=(16,4.4),constrained_layout=True)
colors=['#aab3c2','#6d8cbf','#5590a8','#277c61','#92a78e']
for ai,arm in enumerate(['left','right']):
 r=json.loads((root/f'motion_robot_{arm}_h50.json').read_text())
 names=['Random','Zero-shot ZH','Zero-shot EN','Linear','Contrastive']
 mm=[r['baseline']['random']['direction_top1'],r['zero_shot']['zh']['difference']['direction_top1'],r['zero_shot']['en']['difference']['direction_top1'],r['learned']['linear']['metrics']['direction_top1'],r['learned']['contrastive_zh']['metrics']['direction_top1']]
 y=np.array([m['estimate'] for m in mm])*100;ci=np.array([m['ci95'] for m in mm])*100
 axes[ai].bar(np.arange(5),y,color=colors,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),capsize=3)
 axes[ai].axhline(12.5,color='gray',ls='--',lw=1);axes[ai].set_xticks(np.arange(5),names,rotation=30,ha='right');axes[ai].set_ylim(0,100);axes[ai].set_ylabel('8-direction accuracy (%)');axes[ai].set_title(f'Robot-frame FK, {arm}, H50\n{r["val_samples"]} samples / {r["val_parents"]} validation parents')
for ai,arm in enumerate(['left','right']):
 r=json.loads((root/f'behavior_{arm}.json').read_text())
 for j,(condition,label) in enumerate([('language','Language'),('no_language','No language'),('shuffled_language_training','Shuffled language')]):
  m=r['counterfactual'][condition]['follow_rate'];y=m['estimate']*100;lo,hi=np.array(m['ci95'])*100;x=ai+(j-1)*.24
  axes[2].bar(x,y,width=.21,color=['#277c61','#9ba6b8','#b3c5bd'][j],label=label if ai==0 else None,yerr=[[y-lo],[hi-y]],capsize=3)
axes[2].set_xticks([0,1],['Left','Right']);axes[2].set_ylim(0,100);axes[2].set_ylabel('Counterfactual follow rate (%)');axes[2].set_title('Same observation, 8 text directions\nFixed medium magnitude; not future prediction');axes[2].legend(frameon=False,fontsize=8)
fig.savefig(dest/'motion_and_counterfactual.png',dpi=180);fig.savefig(dest/'motion_and_counterfactual.svg');plt.close(fig)
labels=['F','FL','L','BL','B','BR','R','FR'];fig,axes=plt.subplots(2,2,figsize=(9,8),constrained_layout=True)
for ri,arm in enumerate(['left','right']):
 r=json.loads((root/f'motion_robot_{arm}_h50.json').read_text())
 for ci,kind in enumerate(['linear','contrastive_zh']):
  c=np.array(r['learned'][kind]['metrics']['direction_confusion']);cm=c/np.maximum(c.sum(1,keepdims=True),1)
  ax=axes[ri,ci];im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(8),labels);ax.set_yticks(range(8),labels);ax.set_xlabel('Predicted direction');ax.set_ylabel('True direction');ax.set_title(f'{arm} H50 {kind}')
fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Row-normalized fraction');fig.savefig(dest/'h50_direction_confusions.png',dpi=180);plt.close(fig)
print('RESULT_FIGURES_READY')
