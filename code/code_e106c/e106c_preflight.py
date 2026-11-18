import os,sys,json,hashlib,collections,torch,numpy as np
sys.dont_write_bytecode=True
from e103d_runtime import *
from e106c_train import RESUME,MultiSourceSampler,sample_horizons,reset,WorldModel
from e99w2_data import load
from e90.data import load_data
from e96a.train import configure

def main():
 configure();torch.backends.mha.set_fastpath_enabled(False);old=T/'results/e104p_control_20261115_03';lock=json.loads((old/'PRETRAIN_LOCK.json').read_text());expected=json.loads(RESUME.with_name(RESUME.name+'.json').read_text())['sha256'];assert sha(RESUME)==expected
 for p,h in lock['input_hashes'].items():assert sha(Path(p))==h,p
 _,_,tr,dev=load();_,_,p1,_,_=load_data();assert len(tr.rows)==34777 and len(p1.rows)==107057
 _,_,_,expanded=__import__('e102x_setup').datasets()
 for d in (tr,p1,dev,expanded):assert all(r['canonical_id']!='piper_4dc107335389f5711a0f65e0' for r in d.rows)
 assert not {r['parent_component'] for r in p1.rows}&{r['parent_component'] for r in expanded.rows}
 cp=torch.load(RESUME,map_location='cpu',weights_only=False);assert cp['step']==10000 and cp['phase']=='piper' and cp['group']=='F2' and cp['seed']==90
 m=WorldModel(tr.norm);m.load_state_dict(cp['model'],strict=True);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(cp['optimizer'])
 for x in list(cp['model'].values())+[v for state in opt.state.values() for v in state.values() if torch.is_tensor(v)]:assert torch.isfinite(x).all()
 assert all(pg['lr']==3e-4 and pg['weight_decay']==1e-4 for pg in opt.param_groups)
 steps={n:int(opt.state[p]['step']) for n,p in m.named_parameters() if p in opt.state};assert steps['delta_head.weight']==12000
 reset();sampler=MultiSourceSampler(tr.rows,seed=90,weights={'piper':1.},caps={},demo3_cap=.45);p1sampler=MultiSourceSampler(p1.rows,seed=10390,weights={'piper':1.},caps={},demo3_cap=.45)
 h3=hashlib.sha256();h1=hashlib.sha256();hf=hashlib.sha256()
 for step in range(1,10001):
  ids=sampler.sample(32);pids=p1sampler.sample(32);hs=sample_horizons(2000+step,90);h3.update(np.asarray(ids,dtype='<i8').tobytes());h1.update(np.asarray(pids,dtype='<i8').tobytes());hf.update(json.dumps([ids,list(hs)],default=int).encode())
  if step%1000==0:atomic(CONTROL/'preflight_status.json',dict(step=step,status='CPU_sampler_replay',brt=brt()));print('replay',step,flush=True)
 got=dict(P3_samples_sha256=h3.hexdigest(),P1_samples_sha256=h1.hexdigest(),P3_flow_sha256=hf.hexdigest());summary=json.loads((RESUME.parent/'summary_10000.json').read_text());assert all(summary[k]==v for k,v in got.items()),got;assert summary['adapter_initial'] is True
 torch.save(dict(p3=sampler.state_dict(),p1=p1sampler.state_dict()),CONTROL/'SAMPLER_10000.pt')
 checkpoints={k:lock['checkpoints'][k] for k in ('A2','F0','F2')};checkpoints.update(S15000=str(T/'results/e106c_f2_s90_20261115_01/ckpt_15000.pt'),S20000=str(T/'results/e106c_f2_s90_20261115_01/ckpt_20000.pt'))
 lock.update(checkpoints=checkpoints,resume_sha=expected,initial_sha=lock['initial_sha'],brt=brt(),GPU_seconds_cap=7200,criterion='Description only; fixed checkpoints, no dev selection; A2 remains default',code_hashes={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')})
 atomic(CONTROL/'PRETRAIN_LOCK.json',lock);out=dict(status='passed',resume_sha=expected,optimizer_steps=steps,replay_hashes=got,sampler_draws=[sampler.count,p1sampler.count],CUDA_initialized=torch.cuda.is_initialized(),sampler_method='CPU replay steps1..10000 then restore exact nested state/counts, retain per-step F2 seed namespace and globalstep2000+local',evaluator_source_sha=sha(ROOT/'e106c_eval.py'),brt=brt());atomic(CONTROL/'CPU_AUDIT.json',out);report('## CPU核验与训练前锁定\n'+json.dumps(out,ensure_ascii=False));assert not torch.cuda.is_initialized();print('CPU AUDIT PASSED',expected,flush=True)
if __name__=='__main__':main()
