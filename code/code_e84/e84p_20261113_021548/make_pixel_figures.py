import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,cv2,json
from common import *
root=P/'frozen';figdir=root/'figures';figdir.mkdir(exist_ok=True)
f=np.load(root/'pixel_patch_text_maps.npz');maps=f['patch_cosine'];rows={r['id']:r for r in [json.loads(s) for s in (R/'pixel_gt/pixel_gt.jsonl').read_text().splitlines()]};ids=f['ids'].tolist()
selected=['G002','G027','G103','G126']
fig,axes=plt.subplots(4,4,figsize=(15,11),constrained_layout=True)
for ri,uid in enumerate(selected):
 row=rows[uid];i=ids.index(uid);im=cv2.cvtColor(cv2.imread(row['decoded_frame']),cv2.COLOR_BGR2RGB);w,h=row['resolution'];a=f['affine'][i]
 square=cv2.resize(maps[i].reshape(14,14,-1),(224,224));mm=cv2.warpAffine(square,a[:2],(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
 for ci,(title,qi,key) in enumerate([('Original + approximate points',None,None),('right gripper',2,'R'),('left gripper',3,'L'),('orange zip tie',4,'TIE')]):
  ax=axes[ri,ci];ax.imshow(im)
  if qi is not None:ax.imshow(mm[:,:,qi],cmap='magma',alpha=.55,vmin=mm[:,:,[2,3]].min() if qi in [2,3] else None,vmax=mm[:,:,[2,3]].max() if qi in [2,3] else None)
  for k,col in [('L','cyan'),('R','lime'),('HEAD','red'),('TIP','white')]:
   an=row['annotations'][k]
   if an['point_xy'] is not None:ax.plot(*an['point_xy'],'+',color=col,markersize=10)
  ax.set_title(f'{uid} {row["probe_split"]}\n{title}',fontsize=9);ax.axis('off')
fig.savefig(figdir/'pixel_qualitative_fixed_samples.png',dpi=140);plt.close(fig)
print('FIGURE',figdir/'pixel_qualitative_fixed_samples.png')
