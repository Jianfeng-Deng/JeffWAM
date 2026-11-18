import os,sys,json,time,traceback,hashlib,collections
from pathlib import Path
import numpy as np,torch
from e98w_evaluate import T,R,ART,H,CAMS,METHODS,aggregate,WorldModel,get_data,augment
from e96a.train import configure
REC=ART/'recovery_02'
def put(n,z):(REC/n).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
 started=time.monotonic();assert not REC.exists();REC.mkdir();configure();torch.backends.mha.set_fastpath_enabled(False)
 old_paths=list(ART.glob('*_vectors_h*.npz'));hashes={p.name:sha(p) for p in old_paths}
 put('RECOVERY_LOCK.json',dict(status='locked_before_repair',method='CPU SDPA with MHA eval fastpath disabled, training-equivalent math; patch nonfinite rows only',model_checkpoints={v:sha(T/f'results/e98w_{v.lower()}_20261114_04/ckpt_05000.pt') for v in ('A','B')},old_vector_sha=hashes,steps=5000,scale='unchanged',bootstrap=2000,seed=90,no_model_selection=True,no_GPU=True))
 tr,dev=get_data();pools={'train64':augment(tr),'dev':augment(dev)};buf={(pool,h):dict(np.load(ART/f'{pool}_vectors_h{h}.npz')) for pool in pools for h in H};patches=[];contracts=[]
 for method in ('A','B'):
  cp=torch.load(T/f'results/e98w_{method.lower()}_20261114_04/ckpt_05000.pt',map_location='cpu',weights_only=False);assert cp['step']==5000
  model=WorldModel(tr.norm,method);model.load_state_dict(cp['model']);del cp;model.eval()
  # Train-only valid/invalid mask regression, with a nonzero trained readout.
  z=buf['train64',10];bad=~np.isfinite(z[method]).all((1,2));first_bad=int(z['indices'][np.where(bad)[0][0]]);first_good=int(z['indices'][np.where(~bad)[0][0]]);c={}
  with torch.inference_mode():
   outputs={}
   for fast in (True,False):
    torch.backends.mha.set_fastpath_enabled(fast)
    for label,i in [('good',first_good),('bad',first_bad)]:
     b=tr.batch([i],(10,),'short');pr=model.forward_mean(b)['mean'];outputs[fast,label]=pr.clone();c[f'{label}_fast{fast}_finite']=bool(torch.isfinite(pr).all())
   diff=float((outputs[True,'good']-outputs[False,'good']).abs().max());assert diff<2e-5;assert not c['bad_fastTrue_finite'] and c['bad_fastFalse_finite'];c.update(method=method,finite_backend_max_abs_diff=diff)
  contracts.append(c);torch.backends.mha.set_fastpath_enabled(False)
  with torch.inference_mode():
   for pool,data in pools.items():
    for h in H:
     z=buf[pool,h];a=z[method];bad=~np.isfinite(a).all((1,2));positions=np.where(bad)[0];finite_hash=digest(a[~bad]);ids=z['indices'][positions].tolist();preds=[]
     for start in range(0,len(ids),8):
      ii=ids[start:start+8];b=data.batch(ii,(h,),'short');pr=model.forward_mean(b)['mean']
      if method=='B':pr=pr+model.robot_prediction(b)
      assert torch.isfinite(pr).all();assert np.max(abs(b['world'].numpy()-z['target'][positions[start:start+8]]))<1e-6;preds.extend(pr.numpy())
     if ids:
      pred=np.stack(preds);a[positions]=pred;np.savez_compressed(REC/f'patch_{pool}_{method}_h{h}.npz',indices=ids,positions=positions,prediction=pred)
     assert np.isfinite(a).all() and finite_hash==digest(a[~bad]);patches.append(dict(pool=pool,h=h,method=method,repaired_windows=len(ids),kept_finite_windows=int((~bad).sum()),finite_prediction_sha=finite_hash))
     print('CPU_patch',pool,method,h,len(ids),flush=True)
  del model
 put('backend_contracts.json',contracts);put('patch_manifest.json',patches)
 assert all(np.isfinite(z[k]).all() for z in buf.values() for k in ('target',*METHODS));assert not torch.cuda.is_initialized()
 models={h:dict(np.load(R/f'R1_h{h}.npz')) for h in H};Qall={}
 for h in H:
  Q=[];m=models[h]
  for j in range(m['w1'].shape[0]):
   A=np.concatenate([m['w1'][j],m['ym'][j][None]],0).astype('f8');_,sv,V=np.linalg.svd(A,full_matrices=False);rank=int((sv>sv[0]*max(A.shape)*np.finfo('f4').eps).sum());Q.append(V[:rank].T)
  Qall[h]=Q
 records=[]
 for (pool,h),d in buf.items():
  Y=d['target'].astype('f8');S=d['R1'].astype('f8');k=Y.shape[1];nt=k//3;Q=Qall[h]
  def outside(a):return a-np.stack([(a[:,j]@Q[j])@Q[j].T for j in range(k)],1)
  YE=outside(Y)
  for method in METHODS:
   P=d[method].astype('f8')
   for section,target,pr in [('full',Y,P),('subtraction',Y-S,P-S),('outside_span',YE,outside(P))]:
    for ci,cam in [(None,'all'),*enumerate(CAMS)]:
     sl=slice(None) if ci is None else slice(ci*nt,(ci+1)*nt);y=target[:,sl];p=pr[:,sl];err=((p-y)**2).mean((1,2));cp=(y*y).mean((1,2));cos=(p*y).sum((1,2))/np.maximum(np.sqrt((p*p).sum((1,2))*(y*y).sum((1,2))),1e-12)
     assert np.isfinite(err).all() and np.isfinite(cp).all() and np.isfinite(cos).all()
     for i,index in enumerate(d['indices']):records.append(dict(pool=pool,h=h,camera=cam,section=section,method=method,parent=str(d['parents'][i]),index=int(index),model=float(err[i]),copy=float(cp[i]),cosine=float(cos[i])))
 summary,paired=aggregate(records)
 assert all(np.isfinite([v['mse'],v['copy_mse'],v['skill'],v['cosine'],*v['skill_ci95']]).all() for v in summary.values())
 assert all(np.isfinite([v['delta_skill'],*v['ci95'],v['mse_reduction'],*v['mse_reduction_ci95']]).all() for v in paired.values())
 old=json.loads((ART/'summary.json').read_text());max_unchanged=0.
 for key,v in summary.items():
  if v['h']==100 or v['method'] not in ('A','B'):
   max_unchanged=max(max_unchanged,max(abs(v[f]-old[key][f]) for f in ('mse','copy_mse','skill','cosine')),max(abs(a-b) for a,b in zip(v['skill_ci95'],old[key]['skill_ci95'])))
 assert max_unchanged<1e-12,max_unchanged
 put('evaluation_records.json',records);put('summary.json',summary);put('paired.json',paired)
 decisions={};effective=[];extra=[]
 for h in (25,50,100):
  a=paired[f'dev|{h}|all|full|A-E96a'];b=paired[f'dev|{h}|all|outside_span|B-A'];f=paired[f'dev|{h}|all|full|B-A'];effective.append(a['ci95'][0]>0);extra.append(b['ci95'][0]>0 and f['ci95'][0]>=0);decisions[str(h)]=dict(future_condition_effective=bool(effective[-1]),separation_extra_benefit=bool(extra[-1]),A_minus_E96a_full=a,B_minus_A_outside=b,B_minus_A_full=f)
 put('decision.json',dict(future_condition_effective=bool(all(effective)),separation_extra_benefit=bool(all(extra)),by_h=decisions,full_non_degradation_rule='paired full skill CI lower >=0; unchanged conservative sufficient evidence',recovery_exception='CPU replacement of original nonfinite outputs only; no finite dev output repeated for scoring; checkpoints/scale/windows/thresholds frozen'))
 assert all(sha(ART/name)==v for name,v in hashes.items())
 receipt=dict(status='completed_after_numerical_recovery',wall_seconds=time.monotonic()-started,GPUh=0,CPU_only=True,repaired_predictions=sum(x['repaired_windows'] for x in patches),repaired_dev_predictions=sum(x['repaired_windows'] for x in patches if x['pool']=='dev'),original_finite_predictions_preserved=True,h100_and_all_ridge_E96a_max_summary_diff=max_unchanged,old_vector_SHA_unchanged=True,finite_metric_guard=True,protocol_exception='Original numerical failure repeated forwards only for 18 invalid dev window×h×model outputs; strict one-forward implementation exception disclosed; no candidate selection')
 put('EVALUATION_COMPLETE.json',receipt);print(json.dumps(receipt),flush=True)
except_dummy=None
if __name__=='__main__':
 try:main()
 except BaseException:
  if REC.exists():put('failure.json',dict(status='stopped',traceback=traceback.format_exc()))
  raise
