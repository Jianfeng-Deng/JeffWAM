import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import sys
sys.dont_write_bytecode=True
import numpy as np,cv2,json,collections,time,pathlib
from common import *
from fk_consumption import truth_contract
from evaluation_data import load_features
from encoder import Encoder,WORDS,CONTROLS
from metrics import normalize,localization,parent_ci,paired_ci
def main(version='frozen'):
    truth_contract();root=P/version;root.mkdir(exist_ok=True);out=root/'localization.json'
    if out.exists():return
    froot=R/'features'/RUN/version;ready=json.loads((froot/'READY_features.json').read_text());sel=json.loads(pathlib.Path(ready['selection_path']).read_text())
    # Only independent probe-val parents produce held-out localization scores.
    allowed={(r['canonical_id'],r['anchor_source_row']) for r in sel['samples'] if r['probe_split']=='probe-val'}
    records=[r for r in json.loads((R/'gt/image_index.json').read_text())['records'] if r['status']=='ok' and r['eligible_for_e84'] and (r['canonical_id'],r['anchor_source_row']) in allowed]
    encoder=Encoder('cpu',str(R/'lora'/RUN/'adapter.pt') if version=='lora' else None)
    texts=[f'This is a photo of the {s}.' for s in WORDS+CONTROLS];tv=encoder.text(texts);raw=[];coarse=[];skips=collections.Counter();prev=None;cache=None
    mapdir=root/'patch_text_maps';mapdir.mkdir(exist_ok=True)
    for ri,r in enumerate(records):
        cid=r['canonical_id']
        if cid!=prev:
            if cache is not None:cache.close()
            cache=np.load(froot/(cid+'.npz'));prev=cid
        positions=np.flatnonzero(cache['anchor_source_row']==r['anchor_source_row'])
        if not len(positions):skips['missing_feature']+=1;continue
        ni=int(positions[0]);assert cache['patch_available'][ni]
        ci=['cam_high','cam_left_wrist','cam_right_wrist'].index(r['camera']);actualrow=int(cache['camera_rows'][ni,-1,ci]);assert actualrow==r['source_row']
        patch=cache['patch_features'][ni,ci].astype('f4');af=cache['affine'][ni,ci];rect=cache['valid_rect'][ni,ci];h,w=cache['original_hw'][ni,ci]
        mask=cv2.imread(r['mask_path'],0);assert mask.shape==(h,w)
        valid=np.zeros((224,224),bool);x0,y0,x1,y1=map(int,rect);valid[y0:y1,x0:x1]=True
        warped=np.zeros((224,224),bool);warped[y0:y1,x0:x1]=cv2.resize(mask,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)>0
        saved_maps={}
        for method,vec in [('singleton_MAP',encoder.patch_vectors(patch)),('raw_patch',normalize(patch))]:
            scores=vec@tv.T;maps=np.stack([cv2.resize(scores[:,j].reshape(14,14),(224,224),interpolation=cv2.INTER_LINEAR) for j in range(len(texts))])
            saved_maps[method]=scores.reshape(14,14,len(texts)).astype('f2')
            # Exact HSV membership is a color proxy, not an instance identity label.
            for j in [4,7,8,9,10,11,12]:
                m=localization(maps[j],warped,valid)
                if m is None:skips['empty_or_full_hsv_mask']+=1;continue
                raw.append(dict(m,parent=r['parent_component'],camera=r['camera'],cohort=r['cohort'],image_id=r['image_id'],method=method,query=(WORDS+CONTROLS)[j],scope='fixed HSV orange-pixel proxy only'))
            if r['camera']=='cam_high' and r['cohort']=='demo4' and (h,w)==(480,640) and r.get('projection_time_valid') and r.get('projection_uv'):
                uv=np.array(r['projection_uv'],dtype=float);uv_net=np.c_[uv,np.ones(2)]@af.T
                yy,xx=np.mgrid[:224,:224]
                # Sweep coarse disk radii; none of these are approved instance masks.
                for radius in [8,16,32]:
                    discs=[((xx-u)**2+(yy-v)**2)<=(radius*float(af[0,0]))**2 for u,v,_ in uv_net]
                    for arm,ai,queryindex in [('left',0,1),('right',1,0)]:
                        point=uv_net[ai,:2];other=uv_net[1-ai,:2]
                        if np.linalg.norm(point-other)<2*radius*af[0,0]:continue
                        for noun,q in [('arm',queryindex),('gripper',queryindex+2)]:
                            m=localization(maps[q],discs[ai],valid)
                            swapped=localization(maps[(1-queryindex)+(2 if noun=='gripper' else 0)],discs[ai],valid)
                            if m is None or swapped is None:continue
                            vmax=np.where(valid,maps[q],-np.inf);vswap=np.where(valid,maps[(1-queryindex)+(2 if noun=='gripper' else 0)],-np.inf)
                            py,px=np.unravel_index(np.argmax(vmax),vmax.shape);sy,sx=np.unravel_index(np.argmax(vswap),vswap.shape)
                            p=np.array([px,py]);sp=np.array([sx,sy])
                            follows=bool(np.linalg.norm(p-point)<np.linalg.norm(p-other) and np.linalg.norm(sp-other)<np.linalg.norm(sp-point))
                            coarse.append(dict(m,parent=r['parent_component'],image_id=r['image_id'],camera=r['camera'],cohort=r['cohort'],arm=arm,noun=noun,method=method,radius_canonical_px=radius,swapped_pointing=swapped['pointing'],swapped_auroc=swapped['auroc'],text_switch_follows=follows,max_location_change_px=float(np.linalg.norm(p-sp)/af[0,0]),scope='UNAPPROVED coarse legacy projection disks; not whole-arm/gripper truth'))
        np.savez_compressed(mapdir/(r['image_id']+'.npz'),affine=af,valid_rect=rect,**saved_maps)
        if ri%100==0:print('LOCALIZATION',ri,len(records),flush=True)
    if cache is not None:cache.close()
    color_summary=[]
    for method in ['singleton_MAP','raw_patch']:
        for camera in ['all','cam_high','cam_left_wrist','cam_right_wrist']:
            for query in ['orange zip tie','blue zip tie','green zip tie','bicycle wheel','purple cloud','wooden spoon','random unrelated text']:
                rows=[r for r in raw if r['method']==method and (camera=='all' or r['camera']==camera) and r['query']==query]
                if rows:color_summary.append({'method':method,'camera':camera,'query':query,**{k:parent_ci([r[k] for r in rows],[r['parent'] for r in rows]) for k in ['pointing','auroc','uniform_pointing_chance']}})
    coarse_summary=[]
    control_gaps=[]
    for method in ['singleton_MAP','raw_patch']:
        for camera in ['all','cam_high','cam_left_wrist','cam_right_wrist']:
            groups=collections.defaultdict(dict)
            for r in raw:
                if r['method']==method and (camera=='all' or r['camera']==camera):groups[r['image_id']][r['query']]=r
            for control in ['blue zip tie','green zip tie','random_text_mean']:
                aa=[];bb=[];pp=[]
                for group in groups.values():
                    a=group.get('orange zip tie')
                    controls=[group.get(x) for x in (['bicycle wheel','purple cloud','wooden spoon','random unrelated text'] if control=='random_text_mean' else [control])]
                    if a is None or any(b is None for b in controls):continue
                    aa.append(a);bb.append({k:float(np.mean([b[k] for b in controls])) for k in ['pointing','auroc']});pp.append(a['parent'])
                if aa:control_gaps.append({'method':method,'camera':camera,'control':control,**{k+'_gap':paired_ci([r[k] for r in aa],[r[k] for r in bb],pp) for k in ['pointing','auroc']}})
    for method in ['singleton_MAP','raw_patch']:
        for radius in [8,16,32]:
            for noun in ['arm','gripper']:
                rows=[r for r in coarse if r['method']==method and r['radius_canonical_px']==radius and r['noun']==noun]
                if rows:coarse_summary.append({'method':method,'radius_canonical_px':radius,'noun':noun,**{k:parent_ci([r[k] for r in rows],[r['parent'] for r in rows]) for k in ['pointing','auroc','swapped_pointing','swapped_auroc','text_switch_follows','max_location_change_px']},'pointing_true_minus_swapped':paired_ci([r['pointing'] for r in rows],[r['swapped_pointing'] for r in rows],[r['parent'] for r in rows])})
    obj={'status':'completed_with_gt_limits','version':version,'heldout_images_requested':len(records),'color_proxy':color_summary,'color_control_gaps':control_gaps,'coarse_left_right':coarse_summary,'unsupported_exact_gt':['right arm','left arm','right gripper','left gripper','locking head','tip'],'scope':'color-membership proxy and unapproved coarse demo4 projection diagnostics only; no precision object localization conclusion','skips':dict(skips),'rows':raw,'coarse_rows':coarse,'map_grid':[14,14],'metric_grid':[224,224],'interpolation':'bilinear score, native binary mask warped to aspect-letterboxed 224; valid excludes pad','texts':texts}
    atomic(out,obj)
    append_report('## 语言—图像完成（受限） '+version+'\n\n'+str(len(records))+'张probe-val图像；7目标词均已生成patch相似图，但可信计算范围仅固定HSV橙色像素proxy。左右arm/gripper只给未认证demo4投影圆盘的诊断与左右切换率；locking head/tip无GT，标N/A。14×14分数上采样224，排除letterbox padding；原生细带mask重采样影响小目标指标。完整指标、随机/蓝绿对照和parent CI见 `'+str(out)+'`，不能拿proxy当精确对象对齐验收。')
    print('LOCALIZATION DONE',len(records),len(raw),len(coarse),flush=True)
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'frozen')
