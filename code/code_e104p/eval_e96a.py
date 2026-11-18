"""E96a fixed-endpoint, single-h deployment; no fit, no selection."""
import json,time,traceback,collections
from pathlib import Path
import numpy as np
import torch
from e96a.common import T,ART,atomic,brt,sha,report
from e96a.model import WorldModel as SpatialModel
from e94a.model import WorldModel as BaseModel
from e96a.train import configure
from e86.data import move
from eval_e94a import get_data,rec,summary,decompose,H,E93

RUN=T/'results/e96a_m1_p1_20261114_01'
M1=T/'results/e94a_m1_p1_20261114_01';M2=T/'results/e94a_m2_p1_20261114_01'

def aggregate_score(rs,score='normalized'):
 if score=='normalized':return summary(rs)
 out={}
 for h in H:
  rr=[r for r in rs if r['h']==h]
  aa=[dict(parent=r['parent'],**r['scores'][score]) for r in rr]
  out[str(h)]=agg(aa)
 return out

def agg(rs):
 if not rs:return dict(windows=0,parents=0,values=None,ci95=None)
 keys=['model','copy','cosine'];ps=sorted({r['parent'] for r in rs})
 a=np.array([[np.mean([r[k] for r in rs if r['parent']==p]) for k in keys] for p in ps])
 draws=a[np.random.default_rng(90).integers(len(a),size=(2000,len(a)))].mean(1)
 v=dict(zip(keys,a.mean(0).tolist()));ci={k:np.quantile(draws[:,j],[.025,.975]).tolist() for j,k in enumerate(keys)}
 v['skill_copy']=1-v['model']/v['copy'];ci['skill_copy']=np.quantile(1-draws[:,0]/draws[:,1],[.025,.975]).tolist()
 return dict(windows=len(rs),parents=len(ps),values=v,ci95=ci)

def enrich(r,pred,b,h):
 nt=8 if h<=50 else 1
 d=(pred-b['world_current'])[0].reshape(3,nt,768)
 y=(b['world_raw']-b['world_current'])[0].reshape(3,nt,768)
 scores={}
 for name,x,z in [('fine_raw',d,y),('global_raw',d.mean(1,keepdim=True),y.mean(1,keepdim=True))]:
  e=(x-z).square();cp=z.square()
  scores[name]=dict(model=float(e.mean()),copy=float(cp.mean()),cosine=float(torch.nn.functional.cosine_similarity(x.flatten(),z.flatten(),dim=0)),model_camera=e.mean((1,2)).tolist(),copy_camera=cp.mean((1,2)).tolist())
 r['scores']=scores;return r

def ridge_predict(b,z,h):
 obs=b['obs'][0].numpy();hv=b['history_valid'][0].numpy();nt=8 if h<=50 else 1;n=3*nt
 zz=obs if h<=50 else obs.mean(2,keepdims=True);cur=zz[4].reshape(n,768);hist=zz[:4].reshape(4,n,768);delta=hist-cur[None];delta[~hv]=0
 X=np.concatenate([(cur-z['pca_mean'])@z['pca'],*[delta[j]@z['pca'] for j in range(4)],np.broadcast_to(hv.astype('f4'),(n,4)),np.eye(n,dtype='f4')],1)
 y=((X-z['x_mean'])/z['x_scale'])@z['coef']+z['intercept']
 return b['world_current']+torch.tensor(y,dtype=torch.float32)[None]*b['world_scale']

def paired_delta(new,old,score):
 out={}
 for h in H:
  a={(r['index'],r['h'],r['parent']):r for r in new if r['h']==h};b={(r['index'],r['h'],r['parent']):r for r in old if r['h']==h};assert a.keys()==b.keys()
  ps=sorted({k[2] for k in a});rows=[]
  for p in ps:
   ids=[k for k in a if k[2]==p];x=[]
   for k in ids:
    ar=a[k] if score=='normalized' else a[k]['scores'][score];br=b[k] if score=='normalized' else b[k]['scores'][score]
    assert abs(ar['copy']-br['copy'])<1e-6
    x.append([ar['model'],br['model'],ar['copy']])
   rows.append(np.mean(x,0))
  v=np.array(rows);draw=v[np.random.default_rng(90).integers(len(v),size=(2000,len(v)))].mean(1);mean=v.mean(0)
  improvement=(mean[1]-mean[0])/mean[2];ci=np.quantile((draw[:,1]-draw[:,0])/draw[:,2],[.025,.975]).tolist()
  out[str(h)]=dict(windows=len(a),parents=len(ps),delta_skill=improvement,ci95=ci,mse_reduction=float(mean[1]-mean[0]),mse_reduction_ci95=np.quantile(draw[:,1]-draw[:,0],[.025,.975]).tolist(),bootstrap=2000,seed=90)
 return out

def raw_decomposition(rs,score):
 out={}
 for h in H:
  rr=[r for r in rs if r['h']==h];q=np.quantile([r['change'] for r in rr],[.25,.5,.75]);groups={}
  for j,cam in enumerate(('top','left_wrist','right_wrist')):
   groups[cam]=agg([dict(parent=r['parent'],model=r['scores'][score]['model_camera'][j],copy=r['scores'][score]['copy_camera'][j],cosine=r['scores'][score]['cosine']) for r in rr])
  for j in range(4):groups['change_Q'+str(j+1)]=agg([dict(parent=r['parent'],**r['scores'][score]) for r in rr if np.searchsorted(q,r['change'],side='right')==j])
  out[str(h)]=dict(change_quartile_bounds=q.tolist(),groups=groups,camera_cosine_note='cosine remains full-window; camera MSE and copy are camera-specific')
 return out

def main():
 guard=None;started=time.monotonic()
 try:
  assert json.loads((RUN/'status.json').read_text())['status']=='completed'
  assert not (ART/'evaluation_status.json').exists()
  configure();tr,dev=get_data();pools=[('train64',tr),('dev',dev)]
  assert not ({r['parent_component'] for r in tr.rows}&{r['parent_component'] for r in dev.rows})
  assert all(r['canonical_id']!='piper_4dc107335389f5711a0f65e0' for _,d in pools for r in d.rows)
  fits={h:dict(np.load(E93/f'ridge_h{h}_coefficients.npz')) for h in H};allrs={};jointrs={};audit={}
  for method,path in [('E96a',RUN),('M1',M1),('M2',M2),('ridge_E93',None)]:
   # Repeated fixed-endpoint inference provides shared global scores; old native tables stay intact.
   if path is not None:
    if guard is None:
     from e96a.resource import Guard
     guard=Guard('E96a_evaluation',1800)
    cp=path/'ckpt_05000.pt';d=torch.load(cp,map_location='cpu',weights_only=False);assert d['step']==5000
    m=(SpatialModel() if method=='E96a' else BaseModel(recipe=method)).to('cuda');m.load_state_dict(d['model']);del d;m.eval()
   for pool,data in pools:
    refs=json.loads((E93/(pool+'_v1_records.json')).read_text());lookup={(r['index'],r['h']):r for r in refs};rs=[];jrs=[]
    if method in ('M1','M2'):
     old=[r for r in json.loads((path/(pool+'_records.json')).read_text()) if r['method']==method+'_mean']
    elif method=='ridge_E93':old=json.loads(((T/'results/e94a_control_20261114_01/ridge_train64_records.json') if pool=='train64' else E93/'linear_records.json').read_text())
    else:old=[]
    ol={(r['index'],r['h']):r for r in old};maxerr=0.
    with torch.inference_mode():
     for i,row in enumerate(data.rows):
      if guard:guard.check()
      if path is not None:
       joint_cpu=data.batch([i],H,'short','P1');joint=m.mean_rollout(move(joint_cpu,'cuda')).cpu()
      for h in H:
       if (i,h) not in lookup:continue
       cpu=data.batch([i],(h,),'short','P1')
       pred=ridge_predict(cpu,fits[h],h) if path is None else m.mean_rollout(move(cpu,'cuda')).cpu()
       r=enrich(dict(rec(row,h,pred,cpu,lookup[i,h],i),method=method,layout='single_h'),pred,cpu,h);rs.append(r)
       if old:
        e=abs(r['model']-ol[i,h]['model']);maxerr=max(e,maxerr);assert e<2e-5,(method,pool,i,h,e)
       if path is not None:
        start=H.index(h)*24 if h<=50 else 72;n=24 if h<=50 else 3;pr=joint[:,start:start+n]
        jrs.append(enrich(dict(rec(row,h,pr,cpu,lookup[i,h],i),method=method,layout='joint_four_h_supplement'),pr,cpu,h))
      if i%32==0:
       atomic(ART/'evaluation_status.json',dict(status='evaluating',method=method,pool=pool,done=i+1,total=len(data.rows),elapsed_s=time.monotonic()-started,brt=brt()));print(method,pool,i+1,flush=True)
    allrs[method,pool]=rs;jointrs[method,pool]=jrs;audit[method+'_'+pool]=dict(records=len(rs),old_native_max_abs_mse_difference=maxerr)
    atomic(ART/(method+'_'+pool+'_records.json'),rs)
    atomic(ART/(method+'_'+pool+'_summaries.json'),{s:aggregate_score(rs,s) for s in ('normalized','fine_raw','global_raw')})
    atomic(ART/(method+'_'+pool+'_decomposition.json'),dict(normalized=decompose(rs),**{s:raw_decomposition(rs,s) for s in ('fine_raw','global_raw')}))
    if jrs:
     atomic(ART/(method+'_'+pool+'_joint_records.json'),jrs);atomic(ART/(method+'_'+pool+'_joint_summaries.json'),{s:aggregate_score(jrs,s) for s in ('normalized','fine_raw','global_raw')})
   if path is not None:del m;torch.cuda.empty_cache()
  paired={p:{s:paired_delta(allrs['E96a',p],allrs['M1',p],s) for s in ('normalized','fine_raw','global_raw')} for p,_ in pools}
  atomic(ART/'paired_E96a_minus_M1.json',paired)
  checks=[]
  for p,_ in pools:
   x=summary(allrs['E96a',p]);ridge=summary(allrs['ridge_E93',p])
   for h in (10,25,50):
    a=x[str(h)];b=ridge[str(h)]['values']['skill_copy'];skill=a['values']['skill_copy'];lo=a['ci95']['skill_copy'][0]
    checks.append(dict(pool=p,h=h,skill=skill,ci_lower=lo,ridge_skill=b,ridge_fraction=skill/b,positive_point=skill>0,positive_ci=lo>0,at_least_half_ridge=skill>=.5*b,passed=lo>0 and skill>=.5*b))
  improvement=bool(paired['dev']['normalized']['10']['ci95'][0]>0)
  # Raw fine/global table explicitly checks a score-granularity-only gain.
  local=bool(paired['dev']['fine_raw']['10']['delta_skill']>0)
  verdict=dict(pre_registered_before_results=True,dev10_paired_improvement_ci_lower_positive=improvement,local_fine_raw_improves=local,six_short_h_checks=checks,six_short_h_all_pass=all(r['passed'] for r in checks),passed=improvement and local and all(r['passed'] for r in checks),no_extension_or_checkpoint_selection=True)
  atomic(ART/'preregistered_verdict.json',verdict);atomic(ART/'evaluation_audit.json',audit)
  atomic(ART/'evaluation_status.json',dict(status='completed',seconds=time.monotonic()-started,bootstrap=2000,seed=90,main_layout='single_h',joint='supplement only',raw_scores='raw residuals; global per-camera average before square; fine_raw same units',peak_mib=torch.cuda.max_memory_allocated()/2**20,brt=brt()))
  report('\n## 统一评估完成\n'+brt()+'：E96a/M1/M2均值与冻结E93岭在全部固定窗口完成同h细格及按camera平均global评分；单h为主、四h为补充；paired_E96a_minus_M1.json及preregistered_verdict.json已落盘。无选点或延训。')
 except BaseException as e:
  atomic(ART/'evaluation_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc(),brt=brt()));raise
 finally:
  if guard:guard.finish()

if __name__=='__main__':main()
