from common import *
import numpy as np

def extract(pool):
 guard=Guard('siglip2_and_O_'+pool,12)
 try:
  import torch,cv2,h5py,importlib.util
  from e86.vision import FrozenEncoder,preprocess,PREPROCESS
  from e98w_model import WorldModel
  torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.mha.set_fastpath_enabled(False)
  dest=R/'features';dest.mkdir(exist_ok=True)
  sources=[s for s in read('sources.json') if s['pool']==pool]
  enc=FrozenEncoder('cuda:0');put('vision_contract.json',PREPROCESS)
  for si,s in enumerate(sources):
   out=dest/s['parent'];out.mkdir(exist_ok=False);ids=np.array(s['feature_rows']);np.save(out/'rows.npy',ids)
   arr=np.lib.format.open_memmap(out/'raw.npy',mode='w+',dtype='f2',shape=(len(ids),3,8,768))
   for ci,cam in enumerate(CAMS):
    video=cv2.VideoCapture(s['videos'][cam]);assert video.isOpened();cursor=-1;pixels=[];masks=[];ix=[]
    def flush():
     if not pixels:return
     guard.check()
     with torch.inference_mode():z=enc.images(torch.stack(pixels),torch.stack(masks)).cpu().numpy().astype('f2')
     assert np.isfinite(z).all();arr[ix,ci]=z;pixels.clear();masks.clear();ix.clear()
    for j,f in enumerate(ids):
     while cursor<int(f):
      ok=video.grab();cursor+=1
      if not ok:raise RuntimeError(f'decode {s["parent"]} {cam} {f}')
     ok,bgr=video.retrieve();assert ok
     x,v,_=preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));pixels.append(x);masks.append(v);ix.append(j)
     if len(ix)==96:flush()
    flush();video.release()
   arr.flush();del arr
   status('vision_'+pool,parent=si+1,total=len(sources),last=s['parent'],gpu_seconds=guard.used+time.monotonic()-guard.started)
  del enc;torch.cuda.empty_cache()
  norm=json.loads((T/'code_e88/artifacts/cpu_audit_01/residual_stats.json').read_text())['scale']
  cp=torch.load(T/'results/e98w_a_20261114_04/ckpt_05000.pt',map_location='cpu',weights_only=False);assert cp['step']==5000
  model=WorldModel(norm,'A');model.load_state_dict(cp['model']);del cp;model=model.to('cuda').eval();model.requires_grad_(False)
  lang=T/'code_e86/artifacts/language_v1';ix=json.loads((lang/'index.json').read_text())['human_instruction'];text=np.load(lang/'vectors.npy',mmap_mode='r')[ix[0]:ix[1]].astype('f4');text=torch.tensor(text,device='cuda')
  # The causal O block is independent of R/W. Audit on a realistic masked batch before production.
  def batch(raw,ids,frames,times):
   frames=np.array(frames);idx=frames[:,None]+np.array([-24,-18,-12,-6,0])[None];valid=idx>=0;idx=idx.clip(0)
   # No observation crosses timestamp gaps.
   for j,f in enumerate(frames):
    for k,h in enumerate(idx[j]):
     if valid[j,k] and not ((np.diff(times[h:f+1])>0)&(np.diff(times[h:f+1])<=.050001)).all():valid[j,k]=False
   look=np.searchsorted(ids,idx);assert (ids[look[valid]]==idx[valid]).all()
   obs=np.asarray(raw[look],dtype='f4');obs[~valid]=0
   meta=np.zeros((len(frames),5,3,2),'f4');meta[:,:,:,0]=(times[idx]-times[frames,None])[:,:,None]
   return dict(obs=torch.tensor(obs,device='cuda'),obs_meta=torch.tensor(meta,device='cuda'),history_valid=torch.tensor(valid[:,:4],device='cuda'),current_valid=torch.ones((len(frames),3),dtype=torch.bool,device='cuda'),text=text[None].expand(len(frames),-1,-1),text_valid=torch.ones((len(frames),len(text)),dtype=torch.bool,device='cuda'))
  fkpath=Path('/mnt/workspace/dengjianfeng/demo4_zip_tie_active_gaze_20261030/bench/piper_bench/fk.py');sp=importlib.util.spec_from_file_location('fk_readonly_e99b3',fkpath);fk=importlib.util.module_from_spec(sp);sp.loader.exec_module(fk);chain=fk.load_chain()
  audit=None
  for si,s in enumerate(sources):
   out=dest/s['parent'];ids=np.load(out/'rows.npy');raw=np.load(out/'raw.npy',mmap_mode='r');frames=np.array(s['targets']);np.save(out/'target_rows.npy',frames)
   with h5py.File(s['source'],'r') as h:
    times=h['timestamp'][:];q=h['observations/qpos'][:].astype('f8');q[:,[6,13]]=h['observations/qpos_raw'][:,[6,13]];br=h['state_source_brackets'][:] if 'state_source_brackets' in h else None
   pose=[]
   for sl in (slice(0,6),slice(7,13)):
    mat=fk.fk_link6(q[:,sl],chain);xyz=mat[:,:3,3]+mat[:,:3,:3]@np.array([0,0,.1358]);rot=np.concatenate([mat[:,:3,0],mat[:,:3,1]],1);pose.append(np.concatenate([xyz,rot],1))
   qp=np.concatenate([q,*pose],1).astype('f4');available=np.maximum.accumulate(np.maximum(times,br[:,:,1].max(1))) if br is not None else times
   proprio=[];maxahead=-float('inf');missing=0
   for f in frames:
    feat=[]
    for off in (0.,.5,1.):
     cutoff=times[f]-off;j=min(f,int(np.searchsorted(available,cutoff,side='right')-1));good=j>=0 and np.isfinite(qp[max(0,j)]).all() and cutoff-times[max(0,j)]<=.100001
     feat.extend(qp[j].tolist() if good else [0.]*32);feat.append(float(good));missing+=not good
     if good:maxahead=max(maxahead,float(available[j]-times[f]));assert j<=f and available[j]<=cutoff+1e-7
    proprio.append(feat)
   np.save(out/'proprio.npy',np.array(proprio,dtype='f4'))
   b=np.lib.format.open_memmap(out/'b.npy',mode='w+',dtype='f2',shape=(len(frames),2304));c=np.lib.format.open_memmap(out/'c.npy',mode='w+',dtype='f2',shape=(len(frames),1024))
   with torch.inference_mode():
    for start in range(0,len(frames),64):
     guard.check();fs=frames[start:start+64];inp=batch(raw,ids,fs,times);o,ov,_=model.observation(inp);x=o
     for block in model.blocks:x=block(x,torch.zeros((o.shape[1],o.shape[1]),dtype=torch.bool,device='cuda'),~ov)
     x=model.norm(x);assert torch.isfinite(x).all()
     if audit is None:
      inp.update(robot=torch.zeros(len(fs),101,124,device='cuda'),robot_valid=torch.ones(len(fs),101,dtype=torch.bool,device='cuda'),world_horizons=torch.full((len(fs),24),10,dtype=torch.long,device='cuda'),world_positions=torch.arange(24,device='cuda'),world_valid=torch.ones(len(fs),24,dtype=torch.bool,device='cuda'))
      full=model(inp,w=torch.zeros(len(fs),24,768,device='cuda'))['observation'];err=float((full-x).abs().max());assert err<2e-5,err
      inp['robot'].normal_();again=model(inp,w=torch.randn(len(fs),24,768,device='cuda'))['observation'];leak=float((again-full).abs().max());assert leak<1e-6
      audit=dict(o_subgraph_max_error=err,future_robot_and_w_perturbation_error=leak,ckpt_sha256=sha(T/'results/e98w_a_20261114_04/ckpt_05000.pt'))
     cur=x[:,:24].reshape(len(fs),3,8,256).mean(2).flatten(1);hv=ov[:,24:56];hist=(x[:,24:56]*hv[:,:,None]).sum(1)/hv.sum(1,keepdim=True).clamp_min(1);c[start:start+len(fs)]=torch.cat([cur,hist],1).cpu().numpy().astype('f2');b[start:start+len(fs)]=np.asarray(raw[np.searchsorted(ids,fs)],dtype='f4').mean(2).reshape(len(fs),2304).astype('f2')
   b.flush();c.flush();put(f'{s["parent"]}_state_audit.json',dict(max_available_ahead_seconds=maxahead if np.isfinite(maxahead) else None,missing_sample_slots=missing,slots=len(frames)*3,brackets_present=br is not None))
   status('O_'+pool,parent=si+1,total=len(sources),last=s['parent'])
  put('O_contract_'+pool+'.json',audit);report('## 特征提取 '+pool+' '+now()+'\nSigLIP2 exact E86 preprocessing；源帧原位解码。O-only 子图与完整 A 前向差及未来 R/W 扰动检查：'+json.dumps(audit,ensure_ascii=False)+'。全部状态样本检查来源右端点≤当前截止时刻；缺失填零并附有效位。')
 finally:guard.finish()
if __name__=='__main__':extract(sys.argv[1])
