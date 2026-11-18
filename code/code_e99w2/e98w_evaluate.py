import os,sys,json,time,traceback,collections
from pathlib import Path
import numpy as np,torch
from e98w_core import T,R,E93,H,States,x0,predict,sha
from e96a.common import ART,atomic,brt,report
from e96a.train import configure
from e96a.model import WorldModel as E96
from e98w_model import WorldModel
from e98w_data import augment
from eval_e94a import get_data
from e86.data import move
CAMS=('top','left_wrist','right_wrist');METHODS=('E96a','A','B','R0','R1','R2');SECTIONS=('full','subtraction','outside_span')
def put(name,x):atomic(ART/name,x)
def aggregate(records):
 assert all(np.isfinite([r["model"],r["copy"],r["cosine"]]).all() for r in records), "Nonfinite metrics rejected before aggregation"
 grouped=collections.defaultdict(list)
 for r in records:grouped[(r['pool'],r['h'],r['camera'],r['section'],r['method'])].append(r)
 summary={};arrays={}
 for key,rr in grouped.items():
  by=collections.defaultdict(list)
  for r in rr:by[r['parent']].append([r['model'],r['copy'],r['cosine']])
  ps=sorted(by);a=np.array([np.mean(by[p],0) for p in ps]);v=a.mean(0);draw=a[np.random.default_rng(90).integers(len(a),size=(2000,len(a)))].mean(1)
  summary['|'.join(map(str,key))]=dict(pool=key[0],h=key[1],camera=key[2],section=key[3],method=key[4],windows=len(rr),parents=len(ps),mse=float(v[0]),copy_mse=float(v[1]),skill=float(1-v[0]/v[1]),skill_ci95=np.quantile(1-draw[:,0]/draw[:,1],[.025,.975]).tolist(),cosine=float(v[2]))
  arrays[key]=(ps,a)
 paired={}
 for key in sorted({k[:4] for k in grouped}):
  for new,old in [('A','E96a'),('B','A'),('B','E96a')]:
   ps,a=arrays[(*key,new)];ps2,b=arrays[(*key,old)];assert ps==ps2 and np.max(abs(a[:,1]-b[:,1]))<1e-9
   rows=np.stack([a[:,0],b[:,0],a[:,1]],1);v=rows.mean(0);draw=rows[np.random.default_rng(90).integers(len(rows),size=(2000,len(rows)))].mean(1)
   paired['|'.join(map(str,(*key,new+'-'+old)))]=dict(pool=key[0],h=key[1],camera=key[2],section=key[3],comparison=new+'-'+old,parents=len(ps),delta_skill=float((v[1]-v[0])/v[2]),ci95=np.quantile((draw[:,1]-draw[:,0])/draw[:,2],[.025,.975]).tolist(),mse_reduction=float(v[1]-v[0]),mse_reduction_ci95=np.quantile(draw[:,1]-draw[:,0],[.025,.975]).tolist())
 return summary,paired

def main():
 assert (ART/'TRAIN_COMPLETE.json').exists()
 with (ART/'DEV_ONCE_LOCK.json').open('x') as f:json.dump(dict(brt=brt(),protocol='both fixed5000; single immutable inference; no checkpoint selection'),f)
 fitlock=json.loads((R/'FIT_LOCK.json').read_text())
 for name,digest in fitlock['hashes'].items():assert sha(R/name)==digest
 started=time.monotonic();configure();tr,dev=get_data();pools=[('train64',augment(tr,False)),('dev',augment(dev,False))];models={(h,m):dict(np.load(R/f'{m}_h{h}.npz')) for h in H for m in ('R0','R1','R2')};old={h:dict(np.load(E93/f'ridge_h{h}_coefficients.npz')) for h in H};buf={};guard=None
 try:
  for pool,data in pools:
   refs=json.loads((E93/f'{pool}_v1_records.json').read_text());refs={(r['index'],r['h']):r for r in refs}
   for h in H:
    d={'Y':[],'X':[],'Z':[],'idx':[],'parents':[]}
    for i,row in enumerate(data.rows):
     if (i,h) not in refs:continue
     b=data.get(i,(h,),'short','P1');assert b['world_valid'].all();d['Y'].append(b['world'].numpy());d['X'].append(x0(b['obs'].numpy(),b['history_valid'].numpy(),h,old[h]));d['Z'].append(b['robot'][h].numpy());d['idx'].append(i);d['parents'].append(row['parent_component']);assert row['parent_component']==refs[i,h]['parent'] and abs(float((b['world']**2).mean())-refs[i,h]['copy'])<1e-5
    for k in ('Y','X','Z'):d[k]=np.stack(d[k])
    d['pred']={}
    for method in ('R0','R1','R2'):
     m=models[h,method];d['pred'][method]=predict(m,(d['X']-m['xm'])/m['xs'],(d['Z']-m['zm'])/m['zs'],method).astype('f4')
    buf[pool,h]=d
  from e96a.resource import Guard
  guard=Guard('E98W_fixed_evaluation',1200)
  for method in ('E96a','A','B'):
   ck=T/'results'/({'E96a':'e96a_m1_p1_20261114_01','A':'e98w_a_20261114_04','B':'e98w_b_20261114_04'}[method])/'ckpt_05000.pt'
   cp=torch.load(ck,map_location='cpu',weights_only=False);assert cp['step']==5000
   if method=='E96a':assert sha(ck)=='e55c33857e59aaceef559a510e874c602d3827e232cf5817e5edeb7db6fc9a98'
   m=(E96() if method=='E96a' else WorldModel(tr.norm,method));m.load_state_dict(cp['model']);del cp;m=m.to('cuda');m.eval()
   with torch.inference_mode():
    for pool,data in pools:
     for h in H:
      d=buf[pool,h]
      if method=='E96a' and pool=='dev':
       z=dict(np.load(R/f'dev_forward_snapshot_h{h}.npz'));assert np.array_equal(z['indices'],d['idx']) and np.array_equal(z['parents'],d['parents']) and abs(z['target']-d['Y']).max()<1e-6;d['pred'][method]=z['E96a'];continue
      preds=[]
      for j in range(0,len(d['idx']),16):
       guard.check();b=move(data.batch(d['idx'][j:j+16],(h,),'short'),'cuda');out=m.forward_mean(b)['mean']
       if method=='B':out=out+m.robot_prediction(b)
       preds.extend(out.cpu().numpy())
      d['pred'][method]=np.stack(preds);print('forward',method,pool,h,len(preds),flush=True)
   del m;torch.cuda.empty_cache()
  torch.cuda.synchronize();guard.finish();guard=None
 except BaseException:
  if guard:guard.finish()
  raise
 assert all(np.isfinite(d["Y"]).all() and all(np.isfinite(p).all() for p in d["pred"].values()) for d in buf.values()), "Nonfinite predictions: no scoring or decision allowed"
 records=[];checks=[];baseline_checks=[]
 for (pool,h),d in buf.items():
  refs=json.loads((T/'results/e96a_control_20261114_01'/f'E96a_{pool}_records.json').read_text());refs={(r['index'],r['h']):r for r in refs}
  mse=((d['pred']['E96a']-d['Y'])**2).mean((1,2));diff=max(abs(float(v)-refs[int(i),h]['model']) for i,v in zip(d['idx'],mse));assert diff<2e-5,(pool,h,diff)
  row=dict(pool=pool,h=h,E96a_max_abs_old_window_mse=diff)
  if pool=='dev':
   oldvec=dict(np.load(R/f'dev_forward_snapshot_h{h}.npz'));row['frozen_ridge_max_abs_vector_diff']={m:float(abs(d['pred'][m]-oldvec[m]).max()) for m in ('R0','R1','R2')}
   assert max(row['frozen_ridge_max_abs_vector_diff'].values())<5e-5,row
  baseline_checks.append(row)
 put('baseline_replay_audit.json',baseline_checks)
 for (pool,h),d in buf.items():
  Y=d['Y'].astype('f8');S=d['pred']['R1'].astype('f8');k=Y.shape[1];nt=k//3;Q=[]
  for j in range(k):
   coeff=models[h,'R1'];A=np.concatenate([coeff['w1'][j],coeff['ym'][j][None]],0).astype('f8');_,sv,V=np.linalg.svd(A,full_matrices=False);rank=int((sv>sv[0]*max(A.shape)*np.finfo('f4').eps).sum());Q.append(V[:rank].T)
  def outside(a):return a-np.stack([(a[:,j]@Q[j])@Q[j].T for j in range(k)],1)
  YE=outside(Y);checks.append(dict(pool=pool,h=h,windows=len(Y),parents=len(set(d['parents'])),projection_ranks=[q.shape[1] for q in Q],R1_outside_max=float(abs(outside(S)).max())))
  for method in METHODS:
   P=d['pred'][method].astype('f8')
   for section,target,pr in [('full',Y,P),('subtraction',Y-S,P-S),('outside_span',YE,outside(P))]:
    for ci,cam in [(None,'all'),*enumerate(CAMS)]:
     sl=slice(None) if ci is None else slice(ci*nt,(ci+1)*nt);y=target[:,sl];p=pr[:,sl];err=((p-y)**2).mean((1,2));cp=(y*y).mean((1,2));cos=(p*y).sum((1,2))/np.maximum(np.sqrt((p*p).sum((1,2))*(y*y).sum((1,2))),1e-12)
     for i,index in enumerate(d['idx']):records.append(dict(pool=pool,h=h,camera=cam,section=section,method=method,parent=d['parents'][i],index=int(index),model=float(err[i]),copy=float(cp[i]),cosine=float(cos[i])))
  np.savez_compressed(ART/f'{pool}_vectors_h{h}.npz',target=d['Y'],**d['pred'],indices=d['idx'],parents=d['parents'])
 summary,paired=aggregate(records);put('evaluation_records.json',records);put('summary.json',summary);put('paired.json',paired);put('evaluation_audit.json',checks)
 # A conservative, predeclared interpretation of no degradation: full-skill paired lower CI >=0.
 decision={};effective=[];extra=[]
 for h in (25,50,100):
  a=paired[f'dev|{h}|all|full|A-E96a'];b=paired[f'dev|{h}|all|outside_span|B-A'];full=paired[f'dev|{h}|all|full|B-A'];effective.append(a['ci95'][0]>0);extra.append(b['ci95'][0]>0 and full['ci95'][0]>=0)
  decision[str(h)]=dict(future_condition_effective=bool(effective[-1]),separation_extra_benefit=bool(extra[-1]),A_minus_E96a_full=a,B_minus_A_outside=b,B_minus_A_full=full)
 put('decision.json',dict(future_condition_effective=bool(all(effective)),separation_extra_benefit=bool(all(extra)),by_h=decision,full_non_degradation_rule='paired full skill CI lower >=0; conservative sufficient evidence',brt=brt()))
 put('EVALUATION_COMPLETE.json',dict(status='completed',brt=brt(),wall_seconds=time.monotonic()-started,bootstrap=2000,seed=90,dev_passes=1,endpoint=5000,no_selection=True))
 put('STATUS.json',dict(status='evaluated_awaiting_final_audit',brt=brt()))
if __name__=='__main__':
 try:main()
 except BaseException:
  put('evaluation_failure.json',dict(brt=brt(),traceback=traceback.format_exc()));raise
