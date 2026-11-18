import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys
sys.dont_write_bytecode=True
import numpy as np,torch,cv2,json,time,pathlib,collections,hashlib,traceback,subprocess
import torch.nn.functional as F
from common import *
from contract import sha
from fk_consumption import truth_contract
from encoder import Encoder,preprocess
from evaluation_data import load_features,motion_dataset,candidate_texts
from lora_module import inject,adapter_state
from gpu_budget import GPUBudget
_budget=None
torch.set_num_threads(2);cv2.setNumThreads(1)
def read_image(path,row):
    cap=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);cap.set(cv2.CAP_PROP_POS_FRAMES,int(row));ok,bgr=cap.read();pos=cap.get(cv2.CAP_PROP_POS_FRAMES);cap.release()
    if not ok or abs(pos-(row+1))>1:raise RuntimeError(f'exact decode failure {path} {row}')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
def tensor_feature(model,texts,tok,device):
    enc=tok([s.lower() for s in texts],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(device)
    z=model.get_text_features(**enc)
    if not isinstance(z,torch.Tensor):z=z.pooler_output
    return F.normalize(z.float(),dim=-1)
def patch_project(model,h):
    head=model.vision_model.head;att=head.attention;d=h.shape[-1]
    v=F.linear(h,att.in_proj_weight[2*d:],None if att.in_proj_bias is None else att.in_proj_bias[2*d:]);v=att.out_proj(v)
    return F.normalize(v+head.mlp(head.layernorm(v)),dim=-1)
def main(device='cpu'):
    global _budget
    assert not json.loads((P/'progress.json').read_text()).get('scope_gate','').startswith('blocked_'),'Required pixel GT unavailable; resolve explicit scope gate before model experiments'
    truth_contract();out=R/'lora'/RUN
    if (out/'adapter.pt').exists():return
    if device.startswith('cuda'):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        ad=next([int(t.strip()) for t in r.split(',')] for r in rows.splitlines() if int(r.split(',')[0])==gpu);assert ad[1]<512 and ad[2]<5
        atomic(out/'gpu_admission.json',{'timestamp_brt':brt(),'gpu':gpu,'row':ad})
    else:assert json.loads((P/'progress.json').read_text()).get('gpu_admission',{}).get('cpu_fallback_due'),'CPU fallback gate not met'
    _budget=GPUBudget('lora',device.startswith('cuda'))
    features=load_features('frozen');selection=json.loads(pathlib.Path(features['ready']['selection_path']).read_text());lookup={(r['canonical_id'],r['anchor_source_row']):r for r in selection['samples']};groups=collections.defaultdict(list)
    for arm in ['left','right']:
        df,*_=motion_dataset(features,'robot',arm,50)
        for r in df[df.probe_split=='probe-train'].to_dict('records'):groups[r['parent_component']].append(dict(r,arm=arm))
    chosen=[]
    for parent in sorted(groups,key=lambda x:hashlib.sha256(('20261112:lora:'+x).encode()).digest()):
        candidates=sorted(groups[parent],key=lambda r:hashlib.sha256(r['label_id'].encode()).digest())
        for arm in ['left','right']:
            same=[r for r in candidates if r['arm']==arm]
            if same:chosen.append(same[0])
    max_pairs=384 if device.startswith('cuda') else 128;chosen=chosen[:max_pairs]
    gt={}
    for r in json.loads((R/'gt/image_index.json').read_text())['records']:
        if r['status']=='ok' and r['eligible_for_e84'] and r['probe_split']=='probe-train':gt[(r['canonical_id'],r['camera'],r['source_row'])]=r
    enc=Encoder(device);model=enc.model
    texts=sum([candidate_texts(arm,'robot','zh') for arm in ['left','right']],[])+[f'This is a photo of the {c} zip tie.' for c in ['orange','blue','green']]
    base_text=torch.as_tensor(enc.text(texts),device=device).detach().clone()
    metadata=[];pixels=[];teachers=[];masks=[];labels=[];source_before={};cache={}
    for r in chosen:
        key=(r['canonical_id'],r['anchor_source_row']);s=lookup[key];ci=1 if r['arm']=='left' else 2;cam=['cam_high','cam_left_wrist','cam_right_wrist'][ci]
        if s['endpoint_camera_rows'][1][ci]<0:continue
        route=next(c for c in selection['source_routes'][r['canonical_id']]['cameras'] if c['camera']==cam);path=route['path'];st=pathlib.Path(path).stat();source_before[path]=[st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino]
        pair=[]
        for row in [s['camera_rows'][-1][ci],s['endpoint_camera_rows'][1][ci]]:
            im=read_image(path,row);p,af,rect=preprocess(im);pair.append(p)
        fcache=cache.get(r['canonical_id'])
        if fcache is None:fcache=np.load(features['root']/(r['canonical_id']+'.npz'));cache[r['canonical_id']]=fcache
        ni=int(np.flatnonzero(fcache['anchor_source_row']==r['anchor_source_row'])[0]);teacher=np.stack([fcache['global_features'][ni,-1,ci],fcache['endpoint_global_features'][ni,1,ci]]).astype('f4')
        g=gt.get((r['canonical_id'],cam,s['camera_rows'][-1][ci]));mw=np.zeros((14,14),dtype='f4')
        if g:
            mask=cv2.imread(g['mask_path'],0);_,af,rect=preprocess(np.zeros((g['height'],g['width'],3),dtype='u1'));x0,y0,x1,y1=map(int,rect);small=np.zeros((224,224),dtype='f4');small[y0:y1,x0:x1]=cv2.resize(mask.astype('f4')/255,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mw=cv2.resize(small,(14,14),interpolation=cv2.INTER_AREA)
        pixels.append(torch.stack(pair));teachers.append(teacher);masks.append(mw.ravel());labels.append((0 if r['arm']=='left' else 24)+int(r['label']));metadata.append({k:r[k] for k in ['label_id','parent_component','canonical_id','arm','label']})
    for c in cache.values():c.close()
    assert len(pixels)>=16
    pixels=torch.stack(pixels);teacher=torch.as_tensor(np.stack(teachers),dtype=torch.float32,device=device);mask=torch.as_tensor(np.stack(masks),dtype=torch.float32,device=device);labels=torch.as_tensor(labels,dtype=torch.long,device=device)
    config=inject(model,rank=8,layers=2);model.eval();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
    atomic(out/'training_manifest.json',{'timestamp_brt':brt(),'device':device,'pairs':len(metadata),'parents':len(set(r['parent_component'] for r in metadata)),'config':config,'lr':1e-5,'epochs':3,'drift_weight':.1,'hard_negative_weight':.25,'color_weight':.25,'inputs':'own wrist anchor O and future endpoint W50, for offline observed-motion encoder alignment only','labels':metadata,'texts':texts,'color_supervision':'weak HSV orange membership, not semantic instance truth','seed':20261112,'checkpoint_selection':'last fixed epoch only; no probe-val used'})
    checkpoint=out/'resume.pt';start_epoch=0;start_batch=0;elapsed_before=0;steps=0
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(ck['adapter'],strict=False);opt.load_state_dict(ck['optimizer']);start_epoch=ck['epoch'];start_batch=ck['batch'];elapsed_before=ck['elapsed_s'];steps=ck['steps']
    started=time.monotonic();batchsize=4;logs=[];budget_stopped=False
    for epoch in range(start_epoch,3):
        order=np.random.default_rng(20261112+epoch).permutation(len(pixels));batches=np.array_split(order,max(1,int(np.ceil(len(order)/batchsize))))
        for bi,ids in enumerate(batches):
            if epoch==start_epoch and bi<start_batch:continue
            elapsed=elapsed_before+time.monotonic()-started
            _budget.heartbeat()
            if device.startswith('cuda') and elapsed>3.25*3600:budget_stopped=True;break
            opt.zero_grad(set_to_none=True);inp=pixels[ids].flatten(0,1).to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
                vision=model.vision_model(pixel_values=inp);v=vision.pooler_output.float().reshape(len(ids),2,-1);delta=F.normalize(v[:,1]-v[:,0],dim=-1);t=tensor_feature(model,texts,enc.tok,device);logits=delta@t[:48].T/.07
                truth=labels[ids];ce=F.cross_entropy(logits,truth)
                cls=truth%24;arm=truth//24;reverse=arm*24+((cls//3+4)%8)*3+cls%3;swap=(1-arm)*24+cls
                positive=logits.gather(1,truth[:,None]).squeeze(1);hard=(F.relu(.5+logits.gather(1,reverse[:,None]).squeeze(1)-positive)+F.relu(.5+logits.gather(1,swap[:,None]).squeeze(1)-positive)).mean()/2
                pm=mask[ids];valid=pm.sum(-1)>.01;color=v.sum()*0
                if valid.any():
                    patch=vision.last_hidden_state.reshape(len(ids),2,196,-1)[:,0].float();pooled=(patch_project(model,patch)*pm[:,:,None]).sum(1)/pm.sum(-1,keepdim=True).clamp_min(1e-6);color=F.cross_entropy(pooled[valid]@t[48:].T/.07,torch.zeros(int(valid.sum()),dtype=torch.long,device=device))
                vd=(1-F.cosine_similarity(v,teacher[ids],dim=-1)).mean();td=(1-(t*base_text).sum(-1)).mean();drift=vd+td;loss=ce+.25*hard+.25*color+.1*drift
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step();steps+=1
            logs.append({'epoch':epoch,'batch':bi,'step':steps,'loss':float(loss.detach()),'contrastive':float(ce.detach()),'hard':float(hard.detach()),'color_proxy':float(color.detach()),'visual_cosine_deviation':float(vd.detach()),'text_cosine_deviation':float(td.detach())})
            elapsed=elapsed_before+time.monotonic()-started
            torch.save({'adapter':adapter_state(model),'optimizer':opt.state_dict(),'epoch':epoch,'batch':bi+1,'elapsed_s':elapsed,'steps':steps},str(checkpoint)+'.tmp');os.replace(str(checkpoint)+'.tmp',checkpoint)
            atomic(out/'progress.json',{'stage':'training','epoch':epoch,'batch':bi,'steps':steps,'elapsed_s':elapsed,'device':device,'timestamp_brt':brt(),'last':logs[-1]})
            if steps%20==0:print('LORA',epoch,bi,'loss',round(float(loss),4),'elapsed',round(elapsed),flush=True)
        start_batch=0
        if budget_stopped:break
    state=adapter_state(model);torch.save(state,out/'adapter.pt')
    for path,before in source_before.items():
        st=pathlib.Path(path).stat();assert before==[st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino],path
    atomic(out/'training_result.json',{'status':'stopped_at_budget' if budget_stopped else 'completed_fixed_3epochs','timestamp_brt':brt(),'device':device,'steps':steps,'elapsed_s':elapsed_before+time.monotonic()-started,'gpu_hours':(elapsed_before+time.monotonic()-started)/3600 if device.startswith('cuda') else 0,'adapter_sha256':sha(out/'adapter.pt'),'config':config,'train_pairs':len(metadata),'train_parents':len(set(r['parent_component'] for r in metadata)),'logs':logs,'source_stat_changes':0,'note':'merged-adapter equivalence is synthetic-tested; experimental adapter retained, no full stage-one model release'})
    append_report('## 条件LoRA训练完成\n\n'+brt()+' rank8，视觉/文本各最后2层Q/V，'+str(config['trainable_parameters'])+'可训练参数，'+str(len(metadata))+'个probe-train运动对、3固定epoch，lr1e-5；方向反转/换臂hard negative、橙色HSV弱监督的蓝绿负例，偏离原编码约束0.1。没有用probe-val选checkpoint；原始特征作teacher锚。'+device+'，耗时见training_result.json；CPU缩小版不冒称完整阶段一训练。')
if __name__=='__main__':
    try:main(sys.argv[1] if len(sys.argv)>1 else 'cpu')
    except Exception:
        atomic(R/'lora'/RUN/'failure.json',{'timestamp_brt':brt(),'traceback':traceback.format_exc()});raise
    finally:
        if _budget is not None:_budget.close()
