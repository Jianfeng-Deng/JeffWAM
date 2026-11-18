import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import json,numpy as np,cv2,torch,collections
from common import *
from encoder import Encoder,WORDS,CONTROLS
from metrics import localization,parent_ci,paired_ci
from contract import sha
cv2.setNumThreads(1);torch.set_num_threads(2)
def main_unlocked(version='lora'):
    root=P/version;root.mkdir(exist_ok=True)
    out=root/'pixel_results.json'
    if out.exists():print('pixel results already complete');return
    features=R/'features'/RUN/version;ready=json.loads((features/'READY_pixel_features.json').read_text());assert sha(ready['path'])==ready['sha256']
    f=np.load(ready['path']);data=[json.loads(s) for s in (R/'pixel_gt/pixel_gt.jsonl').read_text().splitlines()]
    assert np.array_equal(f['ids'],[r['id'] for r in data])
    audit=json.loads((P/'pixel_metadata_audit.json').read_text());radii=audit['proposed_fixed_radii']
    encoder=Encoder('cpu',str(L/version/'adapter.pt')); text=[f'This is a photo of the {s}.' for s in WORDS+CONTROLS]
    tv=encoder.text(text);maps=[]
    for i in range(len(data)):
        vectors=encoder.patch_vectors(f['patch_features'][i]);maps.append(vectors@tv.T)
    maps=np.array(maps,dtype='f4');np.savez_compressed(root/'pixel_patch_text_maps.npz',ids=f['ids'],patch_cosine=maps,texts=np.array(text),text_vectors=tv,affine=f['affine'])
    records=[];switch=[]
    target_map={'right arm':'R','left arm':'L','right gripper':'R','left gripper':'L','orange zip tie':'TIE','locking head':'HEAD','tip':'TIP'}
    for i,r in enumerate(data):
        w,h=r['resolution'];a=f['affine'][i];yy,xx=np.mgrid[:h,:w]
        # Linear interpolation within the fixed14x14 patch map; never learned from pixel labels.
        square=cv2.resize(maps[i].reshape(14,14,-1),(224,224),interpolation=cv2.INTER_LINEAR)
        pixel_maps=cv2.warpAffine(square,a[:2],(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
        # Valid full original image; pad lies outside inverse-mapped image. Black sensor bars retained.
        for word,key in target_map.items():
            ann=r['annotations'][key]
            if ann['visibility']!='visible' or ann['ambiguous'] or (ann['point_xy'] is None and ann['bbox_xyxy'] is None):continue
            quality='low' if ann['confidence']=='low' else 'main'
            qi=WORDS.index(word)
            controls={'correct':qi,'random':len(WORDS)+2+(i%4)}
            if key in ['L','R']:controls['left_right_swap']=WORDS.index(word.replace('right','LEFT').replace('left','right').replace('LEFT','left'))
            if key=='TIE':controls.update(blue=7,green=8)
            if ann['point_xy'] is not None:
                x,y=ann['point_xy'];dist=np.hypot(xx-x,yy-y)
            else:
                x1,y1,x2,y2=ann['bbox_xyxy'];dx=np.maximum(np.maximum(x1-xx,xx-x2),0);dy=np.maximum(np.maximum(y1-yy,yy-y2),0);dist=np.hypot(dx,dy)
            for multiplier in [1,2]:
                rad=radii[r['stratum']]['r_primary_px']*multiplier
                hit=dist<=rad
                inside=hit if key!='TIE' else dist==0
                for cond,index in controls.items():
                    scores=pixel_maps[:,:,index]
                    pm=localization(scores,hit);am=localization(scores,inside)
                    if pm is None or am is None:continue
                    records.append({'id':r['id'],'parent':r['parent_id'],'split':r['probe_split'],'stratum':r['stratum'],'word':word,'target':key,'quality':quality,'radius_multiplier':multiplier,'radius_px':rad,'condition':cond,'pointing':pm['pointing'],'auroc':am['auroc'],'pointing_chance':pm['uniform_pointing_chance'],'auc_scope':'bbox_proxy' if key=='TIE' else 'point_neighborhood_proxy','arm_scope':'gripper_point_proxy_no_arm_truth' if 'arm' in word else 'named_target'})
        # Query switching uses two trusted visible gripper points; disjoint hit disks required.
        left,right=r['annotations']['L'],r['annotations']['R']
        good=lambda z:z['visibility']=='visible' and z['point_xy'] is not None and z['confidence']!='low' and not z['ambiguous']
        if good(left) and good(right):
            for mult in [1,2]:
                rad=radii[r['stratum']]['r_primary_px']*mult;pts=np.array([left['point_xy'],right['point_xy']])
                disjoint=np.linalg.norm(pts[0]-pts[1])>2*rad
                for scope,li,ri in [('gripper',3,2),('arm_proxy',1,0)]:
                    maxcoords=[]
                    for qi in [li,ri]:
                        scores=pixel_maps[:,:,qi];yx=np.unravel_index(np.argmax(scores),scores.shape);maxcoords.append([yx[1],yx[0]])
                    maxcoords=np.array(maxcoords);correct=np.linalg.norm(maxcoords-pts,axis=-1)<=rad;wrong=np.linalg.norm(maxcoords-pts[::-1],axis=-1)<=rad
                    switch.append({'id':r['id'],'parent':r['parent_id'],'split':r['probe_split'],'radius_multiplier':mult,'scope':scope,'disjoint_targets':bool(disjoint),'both_queries_follow':float(correct.all()),'mean_correct':float(correct.mean()),'mean_opposite':float(wrong.mean()),'argmax_changed':float(np.linalg.norm(maxcoords[0]-maxcoords[1])>1),'argmax_shift_px':float(np.linalg.norm(maxcoords[0]-maxcoords[1]))})
    (root/'pixel_per_target.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    (root/'pixel_switch_per_frame.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in switch))
    groups=collections.defaultdict(list)
    for r in records:groups[(r['split'],r['quality'],r['word'],r['radius_multiplier'],r['condition'])].append(r)
    summaries=[]
    for (split,q,word,mul,cond),rr in sorted(groups.items()):
        ps=[r['parent'] for r in rr];obj={'split':split,'quality':q,'word':word,'radius_multiplier':mul,'condition':cond,'frames':len(rr),'parents':len(set(ps))}
        for metric in ['pointing','auroc','pointing_chance']:obj[metric]=parent_ci([r[metric] for r in rr],ps)
        obj['pointing_minus_chance']=paired_ci([r['pointing'] for r in rr],[r['pointing_chance'] for r in rr],ps)
        obj['auc_minus_half']=parent_ci([r['auroc']-.5 for r in rr],ps)
        if cond!='correct':
            base={(r['id'],r['word'],r['radius_multiplier']):r for r in records if r['split']==split and r['quality']==q and r['condition']=='correct'}
            for metric in ['pointing','auroc']:
                obj['correct_minus_control_'+metric]=paired_ci([base[(r['id'],word,mul)][metric] for r in rr],[r[metric] for r in rr],ps)
        summaries.append(obj)
    switch_groups=collections.defaultdict(list)
    for r in switch:switch_groups[(r['split'],r['radius_multiplier'],r['scope'],r['disjoint_targets'])].append(r)
    ss=[]
    for (split,mul,scope,disjoint),rr in sorted(switch_groups.items()):
        obj={'split':split,'radius_multiplier':mul,'scope':scope,'disjoint_targets':disjoint,'frames':len(rr),'parents':len({r['parent'] for r in rr})}
        for k in ['both_queries_follow','mean_correct','mean_opposite','argmax_changed','argmax_shift_px']:obj[k]=parent_ci([r[k] for r in rr],[r['parent'] for r in rr])
        obj['correct_minus_opposite']=paired_ci([r['mean_correct'] for r in rr],[r['mean_opposite'] for r in rr],[r['parent'] for r in rr]);ss.append(obj)
    result={'status':'completed','run':RUN,'timestamp_brt':brt(),'frames':150,'frame_split_counts':audit['split_frames'],'counts':audit['target_counts'],'radii':radii,'summaries':summaries,'left_right_switch':ss,'statistics':'2000 equal-parent cluster bootstrap95%CI, fixed seed20261112; no checkpoint selection','method':'frozen singleton-MAP patch cosine, bilinear14x14->224 and inverse affine to native. For point targets pointing and AUROC use circular neighborhoods r/2r. TIE pointing distance to bbox<=r/2r; AUROC exact tight bbox proxy. Arm query uses gripper proxy, no arm truth. Random unrelated text assigned by frame ID index, fixed before scores.','limits':['Only40 probe-val frames32parent;110 training frames separately reported','Same model approximate GT, correlated annotation errors','No whole arm mask, no wrist GT, no accepted calibration; all localization results coarse exploratory','2r neighborhoods may overlap; switch disjoint subset separately reported','No pixel-model/probe training or pixel checkpoint/threshold selection']}
    atomic(out,result);update(pixel_evaluation_complete=True,pixel_result_path=str(out))
    append_report('## 150帧图文探针完成 '+brt()+'\n\n固定半径r/2r、correct/左右互换/颜色互换/随机文本，逐目标pointing与AUROC、低置信分层、左右换句响应和parent bootstrap均存 `'+str(out)+'`。probe-train110帧和probe-val40帧严格分报；ambiguous与不可见排除，low只作独立诊断。整臂只能使用夹爪点代理；TIE的AUROC是紧框代理，其余为点邻域，均非mask真值。不按此结果选择阈值/checkpoint。')
    print('PIXEL_EVAL_COMPLETE',len(records),'target-control records',len(switch),'switch records',flush=True)
def main(version='lora'):
    import fcntl
    root=P/version;root.mkdir(exist_ok=True)
    with (root/'pixel_eval.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return main_unlocked(version)
if __name__=='__main__':
 import sys
 main(sys.argv[1] if len(sys.argv)>1 else 'lora')
