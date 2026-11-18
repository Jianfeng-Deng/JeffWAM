"""Pinned V3 engineering smoke/profile only. No formal launcher or checkpoint selection."""
import os,json,time,random,argparse,traceback,hashlib
from pathlib import Path
from collections import defaultdict,Counter
from .common import *
DEFAULT_INPUT=C/'artifacts/E86-V3-20261113T144520'
def configure():
 import torch,numpy as np
 torch.set_num_threads(2);torch.manual_seed(86);random.seed(86);np.random.seed(86);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False

def datasets(inputs):
 from .data import Package,verify_receipt,verify_file,CachedDataset
 from .language import load
 p=Package();ready=verify_receipt(inputs/'READY_V3_cache_windows.json')
 if ready['status']!='ready' or ready['refs']!=p.refs:raise ValueError('V3 windows not ready/pinned')
 stats=json.loads((inputs/'stats.json').read_text());cache=Path(ready['cache']);verify_file(cache/'contract.json',ready['cache_contract_sha256'])
 ws={s:json.loads(Path(ready['windows'][s]).read_text()) for s in ('P1','P3')}
 # Per-entry smoke subset retains all 78 added cycles, including all 75 demo6 cycles.
 sub={}
 for s,rs in ws.items():
  groups=defaultdict(list)
  for r in rs:groups[r.get('entry_id',r['canonical_id'])].append(r)
  sub[s]=[r for vs in groups.values() for r in vs[::max(1,len(vs)//4)][:4]]
 lc=load(ws['P1']+ws['P3'],C/'artifacts/language_v1')
 full={s:CachedDataset(rs,p.captures,cache,stats,lc) for s,rs in ws.items()};small={s:CachedDataset(rs,p.captures,cache,stats,lc) for s,rs in sub.items()}
 return p,stats,ready,full,small,lc

def model_digest(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def update(m,opt,sampler,ds,stage,step,steps,lr,contract,guard,batch_size=8):
 import torch,numpy as np
 from .data import collate,move
 from .objective import loss
 guard.check();start=time.perf_counter();ix=sampler.sample(batch_size);b=move(collate([ds[i] for i in ix]),'cuda:0')
 warm=min(1.,(step+1)/max(1,int(.05*steps)));cos=.1+.9*(1+np.cos(np.pi*step/steps))/2
 for group in opt.param_groups:group['lr']=lr*warm*cos
 if stage=='P3_no_language':b=dict(b,text_valid=torch.zeros_like(b['text_valid']),align_valid=torch.zeros_like(b['align_valid']))
 opt.zero_grad(set_to_none=True);value,metrics=loss(m,b,'P1' if stage=='P1' else 'P3',step,seed=86,clean_aux_probability=contract['clean_aux_probability'],mode_probabilities=contract['main_mode_probabilities'])
 if not torch.isfinite(value):raise RuntimeError('NaN/Inf loss')
 value.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
 if not torch.isfinite(norm):raise RuntimeError('NaN/Inf gradient')
 opt.step();torch.cuda.synchronize();return float(value.detach()),metrics,time.perf_counter()-start,ix

def stratified_diagnostic(m,ds,stage,contract,guard):
 import torch,numpy as np
 from .data import collate,move
 from .objective import loss
 groups=defaultdict(list)
 for i,r in enumerate(ds.rows):groups[r['stratum']].append(i)
 out={};m.eval()
 with torch.no_grad():
  for st,ids in sorted(groups.items()):
   guard.check();selected=[ids[i] for i in np.linspace(0,len(ids)-1,min(8,len(ids))).astype(int)];b=move(collate([ds[i] for i in selected]),'cuda:0')
   if stage=='P3_no_language':b=dict(b,text_valid=torch.zeros_like(b['text_valid']),align_valid=torch.zeros_like(b['align_valid']))
   v,d=loss(m,b,'P1' if stage=='P1' else 'P3',777,seed=86,mode='world_only' if stage=='P1' else 'independent',clean_aux_probability=0,mode_probabilities=contract['main_mode_probabilities'])
   out[st]=dict(windows=len(selected),fixed_noise_step=777,loss=float(v),**d)
 m.train();return out

def dev_evaluate(m,stats,lc,guard):
 from .data import CachedDataset,stratum
 from .evaluation import evaluate
 caps=json.loads((C/'artifacts/dev_v1/captures.json').read_text());lookup={c['canonical_id']:stratum(c) for c in caps};ws=json.loads((C/'artifacts/dev_v1/windows_v2.json').read_text());ws=[dict(r,stratum=lookup[r['canonical_id']]) for r in ws]
 return evaluate(m,CachedDataset(ws,caps,C/'artifacts/dev_features_v1',stats,lc),'cuda:0',guard)

def smoke(run_id,inputs=DEFAULT_INPUT,aux=.25,probabilities=(.3,.3,.3,.1),strategy='parent_equal',cap=.45,stratum_weights=None):
 from .resource_v3 import Guard
 run=C/'artifacts'/run_id;run.mkdir(exist_ok=False,parents=True);guard=None
 try:
  # All source/hash checks and dataset setup precede CUDA admission.
  p,stats,ready,full,small,lc=datasets(Path(inputs))
  from .checkpoint import save,load,code_hash
  from .model import WAM
  from .data import CappedParentSampler
  from .objective import mode_for
  import torch,numpy as np
  mode_for(0,probabilities=probabilities)
  if not 0<=aux<=1:raise ValueError('Invalid aux probability')
  contract=dict(schema='e86-v3-smoke-contract-v1',package=p.refs,inputs=str(inputs),input_ready_sha256=sha(Path(inputs)/'READY_V3_cache_windows.json'),stats=sha(Path(inputs)/'stats.json'),cache=ready['cache'],cache_contract=ready['cache_contract_sha256'],language=sha(C/'artifacts/language_v1/READY_language.json'),code=code_hash(),P1_subset=small['P1'].rows,P3_subset=small['P3'].rows,clean_aux_probability=aux,main_mode_probabilities=list(probabilities),sampler_strategy=strategy,stratum_weights=stratum_weights or {},demo3_max_window_share=cap,steps=dict(P1=192,P3=384,P3_no_language=384),effective_batch=8,clean_interpretation=dict(main_modes=['sync','independent','action_only','clean'],main_probabilities=list(probabilities),auxiliary_applies_to=['sync','independent'],auxiliary_probability=aux,expected_clean_world_exposure_per_update=probabilities[3]+(probabilities[0]+probabilities[1])*aux,meaning='Additional auxiliary W losses averaged with the main W losses; this is not a 25% mutually-exclusive clean mode'),evaluation='fixed stage-end engineering reference only; no checkpoint choice; no model language labels on dev',formal_training_started=False)
  atomic(run/'contract.json',contract);refs=dict(contract=sha(run/'contract.json'));guard=Guard('V3_P1_P3_paired_smoke',2200);configure();m=WAM().to('cuda:0');results={}
  atomic(run/'language_coverage.json',{s:{st:dict(windows=sum(r['stratum']==st for r in ds.rows),instruction_windows=sum(r['stratum']==st and (r['canonical_id'],r['anchor']) in lc for r in ds.rows),alignment_windows=sum(r['stratum']==st and ('align',r['canonical_id'],r['anchor']) in lc for r in ds.rows)) for st in sorted({r['stratum'] for r in ds.rows})} for s,ds in full.items()})
  with (run/'updates.jsonl').open('x') as log:
   for stage in ('P1','P3','P3_no_language'):
    ds=small['P1' if stage=='P1' else 'P3'];steps=contract['steps'][stage];lr=3e-4 if stage=='P1' else 1e-4;opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);sampler=CappedParentSampler(ds.rows,seed=86 if stage=='P1' else 87,cap=cap,strategy=strategy,stratum_weights=stratum_weights);m.train()
    if stage=='P3':save(run/'P3_start.pt',m,opt,0,sampler,refs)
    if stage=='P3_no_language':load(run/'P3_start.pt',m,opt,sampler,refs)
    before=stratified_diagnostic(m,ds,stage,contract,guard);losses=[];times=[];modes=Counter();drawn=Counter();auxn=0
    for step in range(steps):
     val,metrics,dt,ix=update(m,opt,sampler,ds,stage,step,steps,lr,contract,guard);losses.append(val);times.append(dt);modes[metrics['mode']]+=1;auxn+=int(metrics['clean_aux']);drawn.update(ds.rows[i]['stratum'] for i in ix)
     log.write(json.dumps(dict(stage=stage,step=step+1,loss=val,seconds=dt,**metrics))+'\n')
     if step==steps//2:
      cp=run/(stage+'_resume.pt');save(cp,m,opt,step+1,sampler,refs)
      v1,_,_,_=update(m,opt,sampler,ds,stage,step+1,steps,lr,contract,guard);digest1=model_digest(m);load(cp,m,opt,sampler,refs);v2,_,_,_=update(m,opt,sampler,ds,stage,step+1,steps,lr,contract,guard);digest2=model_digest(m)
      check=dict(step=step+1,loss1=v1,loss2=v2,model_digest1=digest1,model_digest2=digest2,identical_loss=v1==v2,identical_parameters=digest1==digest2);atomic(run/(stage+'_resume_check.json'),check)
      if v1!=v2 or digest1!=digest2:raise RuntimeError('Resume mismatch')
      load(cp,m,opt,sampler,refs)
     if step%16==0:
      log.flush();atomic(run/'status.json',dict(status='smoke_running',stage=stage,step=step+1,elapsed_s=guard.check(),loss=val));print(stage,step+1,round(val,5),flush=True)
    save(run/(stage+'_end.pt'),m,opt,steps,sampler,refs)
    results[stage]=dict(steps=steps,effective_batch=8,windows=len(ds),parents=len({r['parent_component'] for r in ds.rows}),loss_first16=float(np.mean(losses[:16])),loss_last16=float(np.mean(losses[-16:])),all_finite=bool(np.isfinite(losses).all()),mean_step_s=float(np.mean(times[8:])),mode_counts=dict(modes),clean_aux_updates=auxn,clean_world_exposure=(modes.get('clean',0)+auxn)/steps,sampler=sampler.state_dict(),drawn_windows_by_stratum=dict(drawn),stratified_fixed_train_before=before,stratified_fixed_train_after=stratified_diagnostic(m,ds,stage,contract,guard))
    atomic(run/'stage_results.json',results)
    if stage in ('P3','P3_no_language'):atomic(run/('dev_'+stage+'.json'),dev_evaluate(m,stats,lc if stage=='P3' else {},guard))
  atomic(run/'status.json',dict(status='smoke_completed',elapsed_s=guard.check(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,formal_training_started=False))
 except BaseException as e:atomic(run/'status.json',dict(status='stopped',reason=str(e),traceback=traceback.format_exc(),updated_brt=brt()));raise
 finally:
  if guard:guard.finish()

def profile(runpath):
 from .resource_v3 import Guard
 run=Path(runpath);dest=run/'profile_batch32';dest.mkdir(exist_ok=False);guard=None
 try:
  contract=json.loads((run/'contract.json').read_text());p,stats,ready,full,small,lc=datasets(Path(contract['inputs']))
  if p.refs!=contract['package']:raise ValueError('Changed package')
  from .data import CappedParentSampler
  from .model import WAM
  import torch,numpy as np
  guard=Guard('V3_fullpool_batch32_profile',650);configure();results={}
  for stage in ('P1','P3'):
   ds=full[stage];sampler=CappedParentSampler(ds.rows,seed=861,cap=contract['demo3_max_window_share'],strategy=contract['sampler_strategy'],stratum_weights=contract['stratum_weights']);m=WAM().to('cuda:0');m.load_state_dict(torch.load(run/(stage+'_end.pt'),map_location='cpu',weights_only=False)['model']);m.train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4);times=[];modes=defaultdict(list)
   for step in range(108):
    v,d,dt,_=update(m,opt,sampler,ds,stage,step,108,1e-4,contract,guard,32)
    if step>=8:times.append(dt);modes[d['mode']].append(dt)
   results[stage]=dict(pool_windows=len(ds),pool_parents=len({r['parent_component'] for r in ds.rows}),steps_measured=len(times),effective_batch=32,mean_step_s=float(np.mean(times)),p50_s=float(np.median(times)),p99_s=float(np.quantile(times,.99)),mode_mean_s={k:float(np.mean(v)) for k,v in modes.items()},peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,sampler=sampler.state_dict());atomic(dest/'results.json',results)
  t1=results['P1']['mean_step_s'];t3=results['P3']['mean_step_s'];raw=(5000*t1+10000*t3)/3600;atomic(dest/'READY_profile.json',dict(status='ready',results=results,formal_estimate=dict(P1_5000_GPUh=5000*t1/3600,P3_10000_GPUh=10000*t3/3600,single_P1_P3_GPUh=raw,single_P1_P3_with_25pct_margin_GPUh=raw*1.25,paired_R_P_with_25pct_margin_GPUh=(5000*t1+20000*t3)/3600*1.25,scope='Full immutable V3 P3 / permanently excluded readable P1; cached features batch32; 100 measured updates each; excludes cache rebuild/queue/storage tails; no formal launch'),elapsed_s=guard.check()))
 except BaseException as e:atomic(dest/'status.json',dict(status='stopped',reason=str(e),traceback=traceback.format_exc()));raise
 finally:
  if guard:guard.finish()

def replay(runpath,stage):
 from .resource_v3 import Guard
 from .checkpoint import load
 from .model import WAM
 from .data import CappedParentSampler
 run=Path(runpath);contract=json.loads((run/'contract.json').read_text());p,stats,ready,full,small,lc=datasets(Path(contract['inputs']));guard=Guard('V3_external_resume_'+stage,120)
 try:
  import torch
  configure();ds=small['P1' if stage=='P1' else 'P3'];m=WAM().train().to('cuda:0');opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);sampler=CappedParentSampler(ds.rows,seed=86 if stage=='P1' else 87,cap=contract['demo3_max_window_share'],strategy=contract['sampler_strategy'],stratum_weights=contract['stratum_weights']);step=load(run/(stage+'_resume.pt'),m,opt,sampler,dict(contract=sha(run/'contract.json')))
  value,_,_,_=update(m,opt,sampler,ds,stage,step,contract['steps'][stage],3e-4 if stage=='P1' else 1e-4,contract,guard);digest=model_digest(m);expected=json.loads((run/(stage+'_resume_check.json')).read_text());out=dict(status='passed' if value==expected['loss1'] and digest==expected['model_digest1'] else 'failed',stage=stage,actual_loss=value,expected_loss=expected['loss1'],model_digest=digest,expected_model_digest=expected['model_digest1'],elapsed_s=guard.check());atomic(run/(stage+'_external_resume.json'),out)
  if out['status']!='passed':raise RuntimeError('External resume differs')
 finally:guard.finish()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['smoke','profile','replay']);a.add_argument('run');a.add_argument('--stage',default='P3',choices=['P1','P3','P3_no_language']);a.add_argument('--inputs',default=str(DEFAULT_INPUT));a.add_argument('--clean-aux',type=float,default=.25);a.add_argument('--mode-probabilities',type=float,nargs=4,default=[.3,.3,.3,.1]);a.add_argument('--sampler',choices=['parent_equal','stratified'],default='parent_equal');a.add_argument('--demo3-cap',type=float,default=.45);a.add_argument('--stratum-weights',default='{}');x=a.parse_args()
 if x.action=='smoke':smoke(x.run,Path(x.inputs),x.clean_aux,tuple(x.mode_probabilities),x.sampler,x.demo3_cap,json.loads(x.stratum_weights))
 elif x.action=='profile':profile(x.run)
 else:replay(x.run,x.stage)
