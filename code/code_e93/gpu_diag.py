import os,json,subprocess,time,traceback
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e93_dev_diag_20261114_01';started=time.monotonic();device=None;prior=0
from common import *
try:
 assert json.loads((R/'extract_status.json').read_text())['status']=='completed'
 prior=sum(json.loads((R/f).read_text())['seconds'] for f in ('gpu_ledger_extract.json','gpu_ledger_extract_supp.json'));assert prior<6500
 cards=[list(map(int,l.split(','))) for l in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()];device,free=max(cards,key=lambda z:z[1]);assert free>=12288;os.environ['CUDA_VISIBLE_DEVICES']=str(device);os.environ['CUDA_CACHE_PATH']=str(R/'cuda_cache');os.environ['CUDA_CACHE_MAXSIZE']='67108864'
 def guard():
  elapsed=time.monotonic()-started;peak=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0;free=int(subprocess.check_output(['nvidia-smi','-i',str(device),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  atomic(R/'gpu_ledger_diag.json',dict(status='active',gpu=device,seconds=elapsed,total_GPUh=(prior+elapsed)/3600,peak_mib=peak,free_mib=free,brt=brt()))
  assert elapsed+prior<7100 and peak<=8192 and free>=max(4096,peak*.5),'GPU shared headroom/budget guard'
 m=model_cpu().to('cuda');_,tr,dev=get_data();torch.set_num_threads(2)
 def repeat(b,n):
  return {k:(v.repeat((n,)+(1,)*(v.ndim-1)) if isinstance(v,torch.Tensor) and k!='world_positions' else v) for k,v in b.items()}
 def draw(b,i):return torch.cat([torch.randn(b['world'].shape,generator=torch.Generator().manual_seed(9001+i+s*100000)) for s in range(16)]).to('cuda')
 def run(b,w,steps):
  z=w.clone()
  for j in range(steps):
   tau=torch.full((len(z),),j/steps,device='cuda');z=z+m(b,w=z,tw=tau)['world']/steps
  return b['world_current']+z*b['world_scale']
 U,S,V=torch.linalg.svd(m.world_head.weight.float(),full_matrices=False); Q=U[:,S>S.max()*1e-6]; bias=m.world_head.bias; atomic(R/'world_head_subspace.json',dict(weight_shape=list(m.world_head.weight.shape),numerical_rank=Q.shape[1],singular_values=S.tolist(),bias_energy=float(bias.square().mean()),note='All final Euler outputs lie in the affine image of world_head; fixed-checkpoint projection is an oracle lower error bound, not a deployable predictor.'))
 fm=[]
 with torch.inference_mode():
  for name,data in [('dev',dev),('train64',tr)]:
   rs=[];joint=[];backend=[]
   for i,row in enumerate(data.rows):
    guard()
    for h in H:
     ids,why=data.index.resolve(row,h)
     if ids is None:continue
     cpu=data.batch([i],(h,),'short','P1');b=move(cpu,'cuda');bb=repeat(b,16);noise=draw(cpu,i)
     for steps in (4,8,16):
      preds=run(bb,noise,steps).cpu()
      for n in (1,8,16):rs.append(rec(row,h,preds[:n].mean(0,keepdim=True),cpu,index=i,Euler=steps,N=n,backend='CUDA fp32; CPU-generated identical seeds; mean of prefix samples'))
     if i==0:
      cp=m.to('cpu').rollout(cpu,steps=4,seed=9001+i);m.to('cuda');gp=run(b,noise[:1],4).cpu();backend.append(dict(h=h,max_abs_feature_difference=float((cp-gp).abs().max()),normalized_prediction_RMS=float(((cp-gp)/cpu['world_scale']).square().mean().sqrt())))
    # Train uses simultaneous horizon queries; compare this layout explicitly.
    cpu=data.batch([i],H,'short','P1');b=move(cpu,'cuda');bb=repeat(b,16);noise=draw(cpu,i)
    for steps,n in ((4,1),(16,16)):
     pp=run(bb if n==16 else b,noise if n==16 else noise[:1],steps).mean(0,keepdim=True).cpu();offset=0
     for h in H:
      nt=24 if h<=50 else 3;sl=slice(offset,offset+nt);offset+=nt
      if not cpu['world_valid'][0,sl].all():continue
      cb={k:(v[:,sl] if k.startswith('world') and isinstance(v,torch.Tensor) and v.ndim>=2 else v) for k,v in cpu.items()};joint.append(rec(row,h,pp[:,sl],cb,index=i,Euler=steps,N=n))
    # Fixed, parent-spread FM diagnostic; loss input includes all fixed horizons.
    if i%8==0:
     yy=bb['world'];eps=draw(cpu,50000+i);tau=torch.sigmoid(torch.randn(16,generator=torch.Generator().manual_seed(9393+i))).to('cuda');w=(1-tau[:,None,None])*eps+tau[:,None,None]*yy;velocity=m(bb,w=w,tw=tau)['world'];target=yy-eps
     for h in H:
      mask=bb['world_valid'] & (bb['world_horizons']==h)
      if not mask.any():continue
      avg=lambda z:float(z.square().mean(-1)[mask].mean())
      orth=(yy-bias)-((yy-bias)@Q)@Q.T
      fm.append(dict(oracle_subspace_mse=avg(orth),oracle_subspace_FM_floor=avg(orth/(1-tau[:,None,None]).clamp_min(.01)),pool=name,parent=row['parent_component'],h=h,model_fm=avg(velocity-target),zero_velocity=avg(target),zero_clean_head=avg(-w/(1-tau[:,None,None]).clamp_min(.01)-target),copy=avg(yy),samples=16))
    if i%4==0:atomic(R/'diag_status.json',dict(status='evaluating',pool=name,done=i+1,total=len(data.rows),total_GPUh=(prior+time.monotonic()-started)/3600))
    if i%32==0:atomic(R/(name+'_grid_records_partial.json'),rs)
   atomic(R/(name+'_grid_records.json'),rs);atomic(R/(name+'_joint_records.json'),joint);atomic(R/(name+'_backend_check.json'),backend)
   atomic(R/(name+'_grid_summary.json'),{f'E{e}_N{n}':summary([r for r in rs if r['Euler']==e and r['N']==n]) for e in (4,8,16) for n in (1,8,16)})
 atomic(R/'fm_diagnostic.json',fm);guard();atomic(R/'diag_status.json',dict(status='completed',GPUh=(prior+time.monotonic()-started)/3600))
except BaseException as e:atomic(R/'diag_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
finally:
 atomic(R/'gpu_ledger_diag.json',dict(status='finished',gpu=device,seconds=time.monotonic()-started,peak_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0,GPUh=(time.monotonic()-started)/3600,total_GPUh=(prior+time.monotonic()-started)/3600,brt=brt()))
