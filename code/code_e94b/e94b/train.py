from .common import *
import torch,numpy as np,traceback
from .data import load
from .resource import Guard
from .model import WorldModel
import e90.train as original
from e90.model import adapter_pass
from e90.checkpoint import save
from e90.common import object_hash,source_hash

def calibrate(model,raws,guard,run,label):
 opt=torch.optim.Adam(model.reentry.parameters(),lr=.001);history=[];was=model.training;model.eval();raw,sig=raws
 for step in range(1200):
  guard.check();v,_=model.adapter_loss((raw[:64],sig[:64]));opt.zero_grad();v.backward();opt.step()
  if step%25==0:
   with torch.no_grad():hv,hc=model.adapter_loss((raw[64:],sig[64:]))
   history.append(dict(step=step+1,nmse=float(hv),cosine=float(hc)))
   if adapter_pass(hv,hc):break
 with torch.no_grad():hv,hc=model.adapter_loss((raw[64:],sig[64:]))
 passed=adapter_pass(hv,hc);model.adapter_ready.fill_(passed);model.train(was)
 result=dict(enabled=passed,normalized_mse=float(hv),cosine=float(hc),steps=step+1,history=history,threshold=dict(nmse_max=.1,cosine_min=.95),fit_train_parents=64,heldout_train_parents=16,no_dev=True,input='DINOv3 2x4',target='current SigLIP2 observation pool stopgrad')
 atomic(run/('adapter_'+label+'.json'),result);return passed

def run(branch):
 runid='e94b_'+branch+'_20261114_01';dest=R/runid;dest.mkdir(exist_ok=False);g=None
 try:
  data,_,_=load();original.configure();cfg=original.config_for();cfg.update(schema='e94b-p1-v1',target='DINOv3 ViT-B/16 normalized residual',readout=branch,checkpoint_selection='fixed5000')
  original.WorldModel=WorldModel
  if branch=='b2':
   from .readout import configure_readout
   configure_readout(original,cfg)
  g=Guard(runid);m,opt,scheduler,sampler=original.build(cfg,data)
  manifest=dict(original_E90_manifest_sha256=sha(T/'code_e90/artifacts/preflight_01/manifest.json'),target_contract_sha256=sha(CACHE/'contract.json'),normalization_contract_sha256=sha(CACHE/'normalization_contract.json'),fixed_E93_dev_sha256=sha(E93/'dev/windows_full.json'));mh=object_hash(manifest)
  atomic(dest/'config.json',cfg);atomic(dest/'provenance.json',dict(code_hash=source_hash(),manifest_hash=mh,config_hash=object_hash(cfg),inputs=manifest,brt=brt()))
  ids=[];seen=set()
  for i,r in enumerate(data.rows):
   if r['parent_component'] not in seen:ids.append(i);seen.add(r['parent_component'])
   if len(ids)==80:break
  raws=(torch.stack([torch.tensor(np.stack([data.target.get(data.rows[i]['canonical_id'],k,[data.rows[i]['obs_rows'][4][k]])[0] for k in range(3)])) for i in ids]).cuda(),torch.stack([data.base[i]['obs'][4] for i in ids]).cuda())
  atomic(dest/'adapter_parent_selection.json',dict(fit=[data.rows[i]['parent_component'] for i in ids[:64]],heldout=[data.rows[i]['parent_component'] for i in ids[64:]],selection='same original E90 first80 distinct parents'))
  extra=dict(initial_adapter_pass=calibrate(m,raws,g,dest,'initial'));disabled=not extra['initial_adapter_pass'];values=[];chains=0;m.train()
  with (dest/'updates.jsonl').open('x') as log:
   for step in range(1,5001):
    if step==3000:
     passed=calibrate(m,raws,g,dest,'chain_boundary');disabled|=not passed;extra['boundary_adapter_pass']=passed
    if step%16==0:
     with torch.no_grad():av,ac=m.adapter_loss((raws[0][64:],raws[1][64:]))
     passed=adapter_pass(av,ac);m.adapter_ready.fill_(passed and not disabled);extra['last_adapter_check']=dict(step=step,nmse=float(av),cosine=float(ac),passed=passed)
    metrics=original.update(m,opt,scheduler,sampler,data,step,cfg,g);values.append(metrics['loss']);chains+=int(metrics['chain_weight']>0);log.write(json.dumps(metrics)+'\n')
    if step%10==0 or step==1:
     log.flush();atomic(dest/'status.json',dict(status='training',step=step,brt=brt(),elapsed_s=g.check(),loss=metrics['loss']));status(branch+'_training',step=step)
    if step%500==0:
     save(dest/f'ckpt_{step:05d}.pt',m,opt,scheduler,sampler,step,cfg,mh,extra);report(f'## 5 {branch} step {step}\nloss={metrics["loss"]:.6f}，GPU段秒={g.check():.1f}，链有效累计={chains}。')
  final=dict(steps=5000,checkpoint=str(dest/'ckpt_05000.pt'),checkpoint_sha256=sha(dest/'ckpt_05000.pt'),chain_updates=chains,adapter_gate_disabled=disabled,adapter_last=extra.get('last_adapter_check'),first32=float(np.mean(values[:32])),last32=float(np.mean(values[-32:])),sampler=sampler.state_dict(),peak_mib=torch.cuda.max_memory_allocated()/2**20,code_hash=source_hash())
  atomic(dest/'train_summary.json',final);atomic(dest/'status.json',dict(status='completed',step=5000,brt=brt()));report('## 5 '+branch+' 固定终点完成\n'+json.dumps(final,ensure_ascii=False));status(branch+'_trained')
 except BaseException as e:
  atomic(dest/'status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));status(branch+'_stopped',error=str(e));raise
 finally:
  if g:g.finish()
if __name__=='__main__':run(sys.argv[1])
