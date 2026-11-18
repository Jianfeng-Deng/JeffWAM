import os,sys,json,time,hashlib,threading,subprocess,traceback
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu'); R=T/'results/e107p_diagnosis_20261116_01'; REPORT=T/'notes/e107p_multimodal_diagnosis_20261116.md'
sys.path.insert(0,str(T/'code_e106c'))
import numpy as np,torch
from e102x_setup import datasets
from e103d_model import WorldModel
from e102a_contract import contract,onset_labels,states
from e86.data import move
from e96a.train import configure

def put(n,x):
 p=R/n;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def report(s):
 with REPORT.open('a') as f:f.write('\n'+s+'\n')
def gpurows():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 return [dict(index=int(a),uuid=b.strip(),free=float(c)) for a,b,c in (l.split(',') for l in out.splitlines())]
def main():
 configure();torch.backends.mha.set_fastpath_enabled(False)
 lock=json.loads((T/'results/e106c_control_20261115_02/PRETRAIN_LOCK.json').read_text()); entries=lock['entries'];c=contract()
 ck=T/'results/e106c_f2_s90_20261115_02/ckpt_20000.pt'; before=sha(ck)
 p,ready,tr,dev=datasets(); assert len(entries)==211 and len(set(e['parent'] for e in entries))==27
 assert len(dev.base.text_cache)==1 and '__human_default__' in dev.base.text_cache
 assert all(r['canonical_id']!='piper_4dc107335389f5711a0f65e0' for data in (tr,dev) for r in data.rows)
 labels={}; evcache={}
 def numeric(data,i):
  r=data.rows[i];cid=r['canonical_id'];a=r['anchor'];z,q,pose,good=data.states.get(data,cid)
  if q is None or a+50>=len(q) or a<6:return None
  if not good[a-6:a+51].all() or not (z['continuous_segment'][a-6:a+51]==z['continuous_segment'][a]).all():return None
  if (id(data),cid) not in labels:
   lab,ev=onset_labels(q,z['continuous_segment'],good,c);labels[id(data),cid]=lab;evcache[id(data),cid]=ev
  lab=labels[id(data),cid];ss=states(q[a],c).astype(int);vel=q[a,[6,13]]-q[a-6,[6,13]]
  direction=np.where(abs(vel)>np.asarray(c['noise_m'])*3,np.sign(vel),0).astype(int)
  x=np.r_[q[a,[0,1,2,3,4,5,7,8,9,10,11,12]],pose[a],(q[a]-q[a-6])[[0,1,2,3,4,5,7,8,9,10,11,12]]]
  phase=tuple(np.r_[ss,direction].tolist());ys=np.stack([lab[a+h] for h in (10,25,50)])
  branch=any(a<e['onset']<=a+50 for e in evcache[id(data),cid])
  return x,phase,ys,ss,branch
 # Train-only numeric reference: no language or model outputs used.
 refs=[];features=[];grips=[];trainmeta=[]
 grouped={}
 for i,r in enumerate(tr.rows):grouped.setdefault(r['parent_component'],[]).append(i)
 for parent,ids in sorted(grouped.items()):
  selected=set(ids[j] for j in np.linspace(0,len(ids)-1,min(16,len(ids)),dtype=int))
  for i in ids:
   v=numeric(tr,i)
   if v is None:continue
   x,phase,y,ss,br=v;refs.append(dict(index=i,parent=parent,x=x.tolist(),phase=phase,future=y.tolist()))
   if i in selected:
    row=tr.rows[i];raw=np.stack([tr.base.features(row['canonical_id'],k,[row['obs_rows'][4][k]])[0] for k in range(3)])
    assert np.isfinite(raw).all();features.append(raw.mean(1).reshape(-1));grips.append(ss);trainmeta.append(dict(parent=parent,index=i))
 print('train reference',len(refs),'probe',len(features),flush=True)
 np.savez_compressed(R/'train_probe.npz',X=np.array(features,'f4'),y=np.array(grips,'i1'),parent=np.array([v['parent'] for v in trainmeta]))
 put('train_reference.json',refs)
 numericdev=[]
 for e in entries:
  i=e['index'];r=dev.rows[i];assert (r['canonical_id'],r['anchor'],r['parent_component'])==(e['cid'],e['anchor'],e['parent'])
  v=numeric(dev,i)
  if v is None:numericdev.append(dict(index=i,available=False));continue
  x,phase,y,ss,branch=v;numericdev.append(dict(index=i,available=True,x=x.tolist(),phase=phase,future=y.tolist(),current=ss.tolist(),branch=branch))
 put('dev_numeric.json',numericdev);put('entries.json',entries)
 report('## 数据与预处理审计完成\n主集211窗/27parent，+100为208窗/26parent；附表48窗/6parent。训练数值近邻池'+str(len(refs))+'窗；线性夹爪探针训练候选'+str(len(features))+'窗。dev仅human default；夹爪合同SHA='+sha(T/'results/e106c_control_20261115_01/GRIPPER_CONTRACT.json')+'。所有图像特征原位只读，无编码器调用。\n')
 cp=torch.load(ck,map_location='cpu',weights_only=False);m=WorldModel(tr.norm);m.load_state_dict(cp['model']);del cp;m.eval().requires_grad_(False)
 g=max(gpurows(),key=lambda x:x['free']);peak=6144;assert g['free']>=1.5*peak
 os.environ['CUDA_VISIBLE_DEVICES']=g['uuid'];start=time.monotonic();stop=threading.Event();gpu=dict(status='active',initial=g,estimated_peak_mib=peak,seconds=0,pid=os.getpid());put('gpu.json',gpu)
 def check():
  assert time.monotonic()-start<2700,'budget'
  row=next(x for x in gpurows() if x['uuid']==g['uuid']);assert row['free']>=peak*1.5,'shared memory pressure'
  assert not torch.cuda.is_initialized() or torch.cuda.max_memory_allocated()/2**20<=peak,'peak'
  assert sum(p.stat().st_size for root in (R,T/'code_e107p') for p in root.rglob('*') if p.is_file())<4_900_000_000
 def watch():
  while not stop.wait(3):
   try:check()
   except BaseException as ex:
    gpu.update(status='stopped_guard',error=str(ex),seconds=time.monotonic()-start+30);put('gpu.json',gpu);os._exit(88)
 thread=threading.Thread(target=watch,daemon=True);thread.start();arrays={};meta=[]
 try:
  m=m.to('cuda')
  with torch.inference_mode():
   for j,e in enumerate(entries):
    check();i=e['index']
    # Exact evaluator per-horizon call preserves directional masks and the +100 pooled resolution.
    for h in e['horizons']:
     b=move(dev.batch([i],(h,),'short'),'cuda'); assert b['world_valid'].all()
     Y=b['world_raw'][0].cpu().numpy();C=b['world_current'][0].cpu().numpy();S=b['world_scale'][0].cpu().numpy()
     assert all(np.isfinite(x).all() for x in (Y,C,S))
     pred={}
     for cond in ('no_traj','gt_action'):
      bb=dict(b,robot=torch.zeros_like(b['robot']),robot_valid=torch.zeros_like(b['robot_valid'])) if cond=='no_traj' else b
      delta=m.forward_mean(bb)['mean'][0];assert torch.isfinite(delta).all();P=C+S*delta.cpu().numpy();assert np.isfinite(P).all();pred[cond]=P
     key=f'{i}_{h}'; arrays[key+'_Y']=Y;arrays[key+'_C']=C
     for cond,P in pred.items():arrays[key+'_'+cond]=P
     meta.append(dict(index=i,h=h,parent=e['parent'],original48=e['original48']))
    if (j+1)%25==0:print('inference',j+1,'/211',round(time.monotonic()-start,1),flush=True)
   torch.cuda.synchronize();gpu['peak_mib']=torch.cuda.max_memory_allocated()/2**20
 finally:
  del m;torch.cuda.empty_cache();stop.set();thread.join(5);gpu.update(status='closed',seconds=time.monotonic()-start+30);put('gpu.json',gpu)
 np.savez_compressed(R/'vectors.npz',**arrays);put('vector_meta.json',meta)
 assert sha(ck)==before
 put('COLLECT_COMPLETE.json',dict(status='complete',checkpoint_sha=before,gpu=gpu,records=len(meta),train_probe=len(features)))
 report('## 世界头前向完成\n固定20000终点SHA='+before+'；所有原始/预测token有限；MHA fastpath禁用；'+str(len(meta))+'个窗口×时域，两个条件；GPU保守'+str(gpu['seconds']/3600)+'h。预测原始token=当前token+world_scale×mean；另计算位移残差，避免把残差误当绝对未来。\n')
if __name__=='__main__':
 try:main()
 except BaseException:put('COLLECT_ERROR.json',dict(traceback=traceback.format_exc()));raise
