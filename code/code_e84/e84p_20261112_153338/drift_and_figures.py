import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84p_20261112_153338/matplotlib_cache')
import sys
sys.dont_write_bytecode=True
import numpy as np,json,pathlib,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from evaluation_data import load_features,candidate_texts
from encoder import Encoder
from metrics import normalize,parent_ci,paired_ci,planar_follow
def main():
    figdir=P/'figures';figdir.mkdir(exist_ok=True);output={}
    if (R/'features'/RUN/'lora/READY_features.json').exists():
        a=load_features('frozen');b=load_features('lora');assert a['meta'].equals(b['meta'])
        x=a['now'];y=b['now'];cos=1-(normalize(x)*normalize(y)).sum(-1);relative=np.linalg.norm(y-x,axis=-1)/np.maximum(np.linalg.norm(x,axis=-1),1e-6)
        p=a['meta'].parent_component.to_numpy();s=a['meta'].probe_split.to_numpy()
        for which in ['probe-train','probe-val']:
            mask=s==which;output[which]={'image_global_cosine_deviation':parent_ci(cos[mask].mean(-1),p[mask]),'image_global_relative_l2':parent_ci(relative[mask].mean(-1),p[mask])}
        e0=Encoder('cpu');texts=sum([candidate_texts(arm,'robot','zh') for arm in ['left','right']],[]);t0=e0.text(texts);del e0
        e1=Encoder('cpu',str(R/'lora'/RUN/'adapter.pt'));t1=e1.text(texts);del e1
        tc=1-(t0*t1).sum(-1);output['text_motion_candidates']={'mean_cosine_deviation':float(tc.mean()),'max_cosine_deviation':float(tc.max()),'candidates':texts,'cosine_deviation':tc.tolist()}
        comparisons={}
        for arm in ['left','right']:
            fa=P/'frozen'/f'motion_robot_{arm}_h50_predictions.npz';fb=P/'lora'/f'motion_robot_{arm}_h50_predictions.npz'
            if fa.exists() and fb.exists():
                aa=np.load(fa);bb=np.load(fb);assert np.array_equal(aa['feature_index'],bb['feature_index'])
                yy=aa['labels'];pp=aa['parents'];comparisons[arm]={}
                for kind in ['linear','contrastive_zh']:
                    comparisons[arm][kind]={'joint_top1_LoRA_minus_frozen':paired_ci(bb[kind].argmax(-1)==yy,aa[kind].argmax(-1)==yy,pp),'direction_top1_LoRA_minus_frozen':paired_ci(bb[kind].argmax(-1)//3==yy//3,aa[kind].argmax(-1)//3==yy//3,pp)}
        output['paired_probe_comparisons']=comparisons
        behavioral={}
        dirs=np.stack([np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)],-1)
        for arm in ['left','right']:
            pa=P/'frozen'/f'behavior_{arm}_predictions.npz';pb=P/'lora'/f'behavior_{arm}_predictions.npz'
            if pa.exists() and pb.exists():
                aa=np.load(pa);bb=np.load(pb);assert np.array_equal(aa['feature_index'],bb['feature_index']);a=aa['cf_language'][:,:,:2];b=bb['cf_language'][:,:,:2];pp=np.repeat(aa['parents'],8)
                ac=(normalize(a)*dirs[None]).sum(-1);bc=(normalize(b)*dirs[None]).sum(-1)
                af=planar_follow(a,dirs[None],.001);bf=planar_follow(b,dirs[None],.001)
                behavioral[arm]={'follow_rate_LoRA_minus_frozen':paired_ci(bf.ravel(),af.ravel(),pp),'instruction_cosine_LoRA_minus_frozen':paired_ci(bc.ravel(),ac.ravel(),pp)}
        output['paired_behavior_comparisons']=behavioral
        atomic(P/'lora_drift.json',output)
        append_report('## LoRA编码偏离与配对差值\n\n'+json.dumps(output,ensure_ascii=False)+'。图像偏离按同一时刻/同一相机计算，CI以parent为单位；文本48候选无独立parent，不给伪bootstrap区间。')
    for arm in ['left','right']:
        specs=[('frozen','zero','Frozen zero-shot'),('frozen','linear','Frozen linear'),('frozen','contrast','Frozen contrast'),('lora','contrast','LoRA contrast')]
        fig,axes=plt.subplots(1,4,figsize=(14,3.4),constrained_layout=True)
        for ax,(version,kind,title) in zip(axes,specs):
            path=P/version/f'motion_robot_{arm}_h50.json'
            if not path.exists():ax.axis('off');continue
            r=json.loads(path.read_text())
            if r.get('status')!='completed':ax.axis('off');continue
            d=r['zero_shot']['zh']['difference'] if kind=='zero' else r['learned']['linear' if kind=='linear' else 'contrastive_zh']['metrics']
            c=np.array(d['confusion']).reshape(8,3,8,3).sum(axis=(1,3));c=c/np.maximum(c.sum(-1,keepdims=True),1)
            ax.imshow(c,vmin=0,vmax=1,cmap='Blues');ax.set_title(title);ax.set_xlabel('Predicted direction');ax.set_xticks(range(8));ax.set_yticks(range(8))
        axes[0].set_ylabel('True direction');fig.suptitle(f'{arm} arm, robot own-base XY, H50; row-normalized held-out confusion');fig.savefig(figdir/f'robot_direction_{arm}.png',dpi=160);plt.close(fig)
    # Fixed metadata selection for qualitative color-proxy heatmaps.
    gt=json.loads((R/'gt/image_index.json').read_text())['records'];selected=[];seen=set()
    for r in sorted(gt,key=lambda r:r.get('image_id','')):
        if r.get('status')!='ok' or r.get('probe_split')!='probe-val' or not r.get('eligible_for_e84') or r.get('camera')!='cam_high' or r['parent_component'] in seen:continue
        path=P/'frozen/patch_text_maps'/(r['image_id']+'.npz')
        if path.exists():selected.append(r);seen.add(r['parent_component'])
        if len(selected)==4:break
    if selected:
        from PIL import Image
        fig,axes=plt.subplots(len(selected),4,figsize=(10,2.4*len(selected)),squeeze=False,constrained_layout=True)
        for i,r in enumerate(selected):
            im=np.array(Image.open(r['frame_path']));mask=np.array(Image.open(r['mask_path']))
            axes[i,0].imshow(im);axes[i,0].set_title('Original held-out image')
            axes[i,1].imshow(mask,cmap='gray');axes[i,1].set_title('HSV orange proxy (not identity)')
            m=np.load(P/'frozen/patch_text_maps'/(r['image_id']+'.npz'))['singleton_MAP']
            axes[i,2].imshow(m[:,:,4],cmap='magma');axes[i,2].set_title('Frozen: orange zip tie')
            p=P/'lora/patch_text_maps'/(r['image_id']+'.npz')
            if p.exists():axes[i,3].imshow(np.load(p)['singleton_MAP'][:,:,4],cmap='magma');axes[i,3].set_title('LoRA: orange zip tie')
            else:axes[i,3].axis('off')
            for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
        fig.savefig(figdir/'color_proxy_examples.png',dpi=140);plt.close(fig)
    atomic(P/'figures_manifest.json',{'timestamp_brt':brt(),'files':[str(p) for p in sorted(figdir.glob('*.png'))],'warning':'coarse/HSV illustrations do not establish precise object truth'})
    print('FIGURES',len(list(figdir.glob('*.png'))),flush=True)
if __name__=='__main__':main()
