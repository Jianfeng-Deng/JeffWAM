import os
import sys
sys.dont_write_bytecode=True
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,time,json,pathlib,collections,subprocess,traceback
import numpy as np,cv2,torch
from common import *
from contract import contracts,sha
from encoder import Encoder,preprocess,PROCESSOR,WORDS,CONTROLS
from gpu_budget import GPUBudget
_budget=None
cv2.setNumThreads(1);torch.set_num_threads(2)
def stat(p):
    s=pathlib.Path(p).stat();return [s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino]
def size(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
def admit(index):
    r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=[[int(x.strip()) for x in line.split(',')] for line in r.stdout.splitlines()]
    return next(x for x in rows if x[0]==index)
def main(mode,device,adapter=None):
    global _budget
    assert not json.loads((P/'progress.json').read_text()).get('scope_gate','').startswith('blocked_'),'Required pixel GT unavailable; resolve explicit scope gate before model experiments'
    split,caps,routes=contracts();selection=json.loads((P/('selection_'+mode+'_v4.json')).read_text());assert selection['split_sha256']==sha(R/'split/probe_split.json')
    from fk_consumption import truth_contract
    truth_contract()
    version='frozen' if adapter is None else 'lora';dest=R/'features'/RUN/version;dest.mkdir(parents=True,exist_ok=True)
    assert device == 'cuda:0' and os.environ['CUDA_VISIBLE_DEVICES'] == '4'
    from resource_guard import Guard
    guard=Guard()
    _budget=GPUBudget('frozen' if adapter is None else 'lora',device.startswith('cuda'))
    guard.budget=_budget
    encoder=Encoder(device,adapter);atomic(dest/'preprocessing.json',PROCESSOR)
    from pixel_features import extract_pixels
    extract_pixels(encoder,dest,guard,_budget)
    texts=[f'This is a photo of the {s}.' for s in WORDS+CONTROLS]
    np.savez_compressed(dest/'object_text.npz',texts=np.array(texts),features=encoder.text(texts).astype('f2'))
    grouped=collections.defaultdict(list)
    for row in selection['samples']:grouped[row['canonical_id']].append(row)
    completed=[];started=time.monotonic();gpu_elapsed=0
    baseline_total=size(R);baseline_features=size(R/'features'/RUN);written_bytes=0
    for cidx,(cid,rows) in enumerate(sorted(grouped.items())):
        out=dest/(cid+'.npz')
        if out.exists():completed.append(cid);continue
        estimated_shard=len(rows)*(3*196*768*2+7*3*768*2+2048)
        assert baseline_total+written_bytes+estimated_shard+128_000_000<29_000_000_000,'shared E84 artifacts approaching 30GB'
        assert baseline_features+written_bytes+estimated_shard<23_000_000_000,'P feature budget exhausted'
        _budget.heartbeat()
        n=len(rows);all_globals=np.zeros((n,7,3,768),dtype='f2');patches=np.zeros((n,3,196,768),dtype='f2');affine=np.zeros((n,3,3,3),dtype='f4');rect=np.zeros((n,3,4),dtype='i2');shapes=np.zeros((n,3,2),dtype='i2');source_before={}
        camera_rows=np.array([r['camera_rows']+r['endpoint_camera_rows'] for r in rows]);ages=np.array([r['camera_age_s']+r['endpoint_camera_age_s'] for r in rows],dtype='f4')
        for ci,cam in enumerate(['cam_high','cam_left_wrist','cam_right_wrist']):
            route=next(r for r in routes[cid]['cameras'] if r['camera']==cam);path=route['path'];source_before[path]=stat(path)
            wanted=sorted(set(camera_rows[:,:,ci].ravel())-{-1});lookup=collections.defaultdict(list)
            for ni in range(n):
                for hi in range(7):
                    rr=int(camera_rows[ni,hi,ci])
                    if rr>=0:lookup[rr].append((ni,hi))
            cap=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened(),path
            batch=[];batch_meta=[];found=0;cursor=-1;batch_size=8
            def flush():
                nonlocal batch,batch_meta,found
                if not batch:return
                guard.check()
                g,p=encoder.images(torch.stack(batch))
                for bi,(rr,af,rc,shape) in enumerate(batch_meta):
                    for ni,hi in lookup[rr]:
                        all_globals[ni,hi,ci]=g[bi]
                        if hi==4:
                            if version=='frozen' or (rows[ni]['probe_split']=='probe-val' and rows[ni]['mandatory_gt_coverage']):patches[ni,ci]=p[bi]
                            affine[ni,ci]=af;rect[ni,ci]=rc;shapes[ni,ci]=shape
                found+=len(batch);batch=[];batch_meta=[]
            for rr in wanted:
                while cursor<rr:
                    ok=cap.grab();cursor+=1
                    if not ok:raise RuntimeError(f'cannot decode {cid} {cam} row {rr}; no placeholder images')
                ok,bgr=cap.retrieve();assert ok
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);pixel,af,rc=preprocess(rgb)
                batch.append(pixel);batch_meta.append((rr,af,rc,rgb.shape[:2]))
                if len(batch)>=batch_size:flush()
            flush();cap.release();assert found==len(wanted)
            assert stat(path)==source_before[path],('source changed',path)
        tmp=str(out)+'.tmp'
        with open(tmp,'wb') as f:
            save=np.savez if version=='frozen' else np.savez_compressed
            save(f,global_features=all_globals[:,:5],endpoint_global_features=all_globals[:,5:],patch_features=patches,patch_available=np.array([version=='frozen' or (r['probe_split']=='probe-val' and r['mandatory_gt_coverage']) for r in rows]),affine=affine,valid_rect=rect,original_hw=shapes,camera_rows=camera_rows[:,:5],camera_age_s=ages[:,:5],history_valid=camera_rows[:,:5]>=0,endpoint_camera_rows=camera_rows[:,5:],endpoint_camera_age_s=ages[:,5:],endpoint_valid=camera_rows[:,5:]>=0,anchor_source_row=np.array([r['anchor_source_row'] for r in rows]),state=np.array([r['state'] for r in rows],dtype='f4'),parent_component=np.array([r['parent_component'] for r in rows]),probe_split=np.array([r['probe_split'] for r in rows]),cohort=np.array([r['cohort'] for r in rows]))
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,out);atomic(dest/(cid+'_receipt.json'),{'path':str(out),'sha256':sha(out),'samples':n,'source_stat_before_after_equal':True,'sources':source_before,'timestamp_brt':brt()})
        written_bytes+=out.stat().st_size+(dest/(cid+'_receipt.json')).stat().st_size
        completed.append(cid);elapsed=time.monotonic()-started
        atomic(dest/'progress.json',{'timestamp_brt':brt(),'completed_captures':completed,'total_captures':len(grouped),'last_samples':n,'elapsed_s':elapsed,'device':device,'mode':mode,'gpu_hours_this_process':elapsed/3600 if device.startswith('cuda') else 0})
        update(status='extracting_'+version,feature_completed_captures=len(completed),feature_total_captures=len(grouped))
        if cidx%10==0:print('EXTRACT',len(completed),len(grouped),'elapsed_s',round(elapsed),flush=True)
    atomic(dest/'READY_features.json',{'status':'ready','timestamp_brt':brt(),'mode':mode,'version':version,'selection_path':str(P/('selection_'+mode+'_v4.json')),'selection_sha256':sha(P/('selection_'+mode+'_v4.json')),'captures':len(completed),'samples':len(selection['samples']),'device':device,'bytes':size(dest),'elapsed_s':time.monotonic()-started,'patch_scope':'all selected current frames' if version=='frozen' else 'heldout mandatory GT anchors only; other patch arrays are unavailable placeholders and must never be scored'})
    guard.finish();append_report('## 特征完成 '+version+'\n\n'+brt()+' '+str(len(selection['samples']))+'时刻，'+str(len(completed))+'capture；mode='+mode+'；device='+device+'；fp16，历史只存global，当前保存global+196patch；所有源视频stat读取前后相同。空间变换采用完整视野保比例+edge pad到224，bilinear与(.5,.5)归一化；与官方直接方形缩放不同，显式披露此分布限制。')
    update(status='features_ready_wait_gt');print('DONE',len(selection['samples']),flush=True)
if __name__=='__main__':
    try:main(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
    except Exception:
        atomic(P/'extract_failure.json',{'timestamp_brt':brt(),'traceback':traceback.format_exc()});raise
    finally:
        if _budget is not None:_budget.close()
