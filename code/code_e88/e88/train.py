"""Bounded engineering driver only (no formal training entry point)."""
import os,json,time,random,traceback,hashlib,copy,collections
from pathlib import Path
import numpy as np,torch
from e86.data import CappedParentSampler,CachedDataset,collate,move,stratum
from e86.language import load as load_short
from .common import *
from .audit import read_inputs,labels_index,HORIZONS
from .data import Data,sample_horizons
from .model import WorldModel
from .objective import loss,chain_loss
from .evaluation import world_evaluate,action_evaluate

def configure():
    torch.set_num_threads(2);torch.manual_seed(88);np.random.seed(88);random.seed(88);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False

def source_hash():
    h=hashlib.sha256()
    for folder in ('e86','e88'):
        for p in sorted((OUT/folder).glob('*.py')):h.update(str(p.relative_to(OUT)).encode());h.update(p.read_bytes())
    return h.hexdigest()
def digest(m):return hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes() for k,v in sorted(m.state_dict().items()))).hexdigest()
def save(path,m,opt,sampler,step,refs):
    if path.exists():raise FileExistsError(path)
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state_dict(),step=step,refs=refs,code=source_hash(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),path)
def restore(path,m,opt,sampler,refs):
    d=torch.load(path,map_location='cpu',weights_only=False)
    if d['refs']!=refs or d['code']!=source_hash():raise ValueError('Resume contract changed')
    m.load_state_dict(d['model']);opt.load_state_dict(d['optimizer']);sampler.load_state_dict(d['sampler']);torch.set_rng_state(d['rng']);torch.cuda.set_rng_state(d['cuda_rng']);np.random.set_state(d['numpy']);random.setstate(d['python']);return d['step']
def make_data():
    from .language import load
    p,ready,ws,stats=read_inputs();labels,labelsource=labels_index(p);lc=load_short(ws['P1']+ws['P3'],C/'artifacts/language_v1');nv=load();rs=json.loads((ART/'cpu_audit_01/residual_stats.json').read_text());full={s:Data([dict(row,_stage=s) for row in rows],p.captures,ready['cache'],stats,rs,labels,lc,nv) for s,rows in ws.items()};small={}
    for s,ds in full.items():
        groups=collections.defaultdict(list)
        for row in ds.rows:groups[row.get('entry_id',row['canonical_id'])].append(row)
        selected=[r for vs in groups.values() for r in vs[::max(1,len(vs)//4)][:4]]
        small[s]=Data(selected,p.captures,ready['cache'],stats,rs,labels,lc,nv)
    return p,ready,stats,full,small,lc

def make_chain(data,ids,k,language,stage,step,device):
    selected=[];shift=[]
    for i in ids:
        row=dict(data.rows[i],_stage=stage)
        if data.index.resolve(row,k)[0] is None or data.index.resolve(row,2*k)[0] is None:continue
        b=data.shifted(i,k,language,stage)
        if b is not None and b['world_valid'].all():selected.append(i);shift.append(b)
        if len(selected)>=2:break
    if not selected:return None
    first=move(data.batch(selected,(k,),language,stage),device);direct=move(data.batch(selected,(2*k,),language,stage),device);shifted=collate(shift);shifted['world_positions']=shifted['world_positions'][0];shifted=move(shifted,device)
    target=shifted['world_raw']
    if k==50:target=target.reshape(len(selected),3,8,-1).mean(2)
    if not torch.allclose(target,direct['world_raw'],atol=1e-6):raise ValueError('Chain endpoint frame mismatch')
    probability=min(.8,.2+.6*step/192);predicted=bool(torch.rand((),generator=generator(88,stage,step,'reentry_mix')).item()<probability)
    return lambda model:chain_loss(model,first,shifted,direct,k,predicted,steps=2,seed=88+step)

def update(m,opt,sampler,data,stage,step,language,guard,batch=8,force_chain=False,recipe=None):
    guard.check();t=time.perf_counter();ids=sampler.sample(batch);hs=sample_horizons(step);b=move(data.batch(ids,hs,language,stage),'cuda:0');chain=None
    # Subsample expensive chain batches, explicit 1/4 update schedule. Main weight remains 0.1 max.
    if step%4==0 or force_chain:
        k=(25,50,100)[(step//4)%3];chain=make_chain(data,ids,k,language,stage,step,'cuda:0')
    opt.zero_grad(set_to_none=True);v,metrics=loss(m,b,stage,step,chain=chain,chain_warmup=128,chain_ramp=128,engineering_chain_probe=True,mode_probabilities=tuple((recipe or {}).get('mode_probabilities',[.3,.3,.3,.1])),clean_aux_probability=(recipe or {}).get('clean_aux',.25))
    if not torch.isfinite(v):raise RuntimeError('nonfinite loss')
    v.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    if not torch.isfinite(gn):raise RuntimeError('nonfinite gradient')
    opt.step();torch.cuda.synchronize();metrics.update(seconds=time.perf_counter()-t,horizons=list(hs),chain_candidates=chain is not None);return float(v.detach()),metrics

def calibrate(m,data,guard,run):
    ids=[];seen=set()
    for i,r in enumerate(data.rows):
        if r['parent_component'] not in seen:ids.append(i);seen.add(r['parent_component'])
        if len(ids)==80:break
    raws=torch.stack([data.base[i]['obs'][4] for i in ids]).to('cuda:0');fit=raws[:64];held=raws[64:];opt=torch.optim.Adam(m.reentry.parameters(),lr=.001);before=None;history=[];m.eval()
    for step in range(1200):
        guard.check();v,co=m.adapter_loss(fit)
        if step==0:before=dict(normalized_mse=float(v.detach()),cosine=float(co.detach()))
        opt.zero_grad();v.backward();opt.step()
        if step%50==0:
            with torch.no_grad():hv,hc=m.adapter_loss(held)
            history.append(dict(step=step+1,train_normalized_mse=float(v.detach()),holdout_normalized_mse=float(hv),holdout_cosine=float(hc)))
            atomic(run/'adapter_calibration.json',dict(before=before,history=history,threshold=dict(normalized_mse_max=.1,cosine_min=.95),fitted_windows=64,train_holdout_windows=16,no_eval_parents=True))
            if hv<=.1 and hc>=.95:break
    with torch.no_grad():hv,hc=m.adapter_loss(held);m.adapter_ready.fill_(bool(hv<=.1 and hc>=.95))
    atomic(run/'adapter_gate.json',dict(enabled=bool(m.adapter_ready),normalized_mse=float(hv),cosine=float(hc),steps=step+1,threshold=dict(normalized_mse_max=.1,cosine_min=.95),scope='train-only heldout real current frames; threshold independent of model labels'))
    return held

def eval_ids(data):
    out=[]
    for st in ('demo3','demo4','demo6early','demo6native'):
        ids=[i for i,r in enumerate(data.rows) if r['stratum']==st];valid=[i for i in ids if data.index.resolve(data.rows[i],300)[0] is not None]
        out.extend(valid[::max(1,len(valid)//4)][:4] if valid else ids[::max(1,len(ids)//4)][:4])
    return out

def main(runid='smoke_01',recipe='A'):
    if recipe not in ('A','B'):raise ValueError('recipe must be A or B')
    recipe_config=dict(selected=recipe,mode_probabilities=[.3,.3,.3,.1] if recipe=='A' else [.25]*4,clean_aux=.25 if recipe=='A' else 0.)
    run=ART/runid;run.mkdir(exist_ok=False);guard=None
    try:
        p,ready,stats,full,small,lc=make_data();configure()
        contract=dict(schema='e88-v31-smoke-v1',created_brt=brt(),code_sha256=source_hash(),refs=p.refs,input_ready_sha256=sha(INPUT/'READY_V3_cache_windows.json'),residual_stats_sha256=sha(ART/'cpu_audit_01/residual_stats.json'),narration_ready_sha256=sha(ART/'narration_01/READY.json'),P1_steps=160,P3_steps=192,batch=8,language_conditions=['none','short','long'],frozen_SigLIP2=True,no_LoRA=True,formal_training=False,physical_actions='12 joint residuals anchored to measured q; signed absolute grippers m; unchanged next-frame feedback',fine_horizons=[10,25,50],coarse_horizons=[100,200,300],default_formal_max_horizon=100,smoke_horizons='all fixed horizons plus one integer frame offset drawn within a randomly chosen adjacent interval; invalid endpoints masked',attention='O only O; long W only O/long W; short W reads O/A/all W; A reads O/A/all W',cosine='short only, deterministic O-conditioned clean W queries',flow_parameterization='v=(predicted_clean_normalized_residual-w)/(1-tau); analytic full-768 noise skip; zero clean-head initialization; denominator floor .01',adapter_gate=dict(normalized_mse_max=.1,cosine_min=.95),chain=dict(max_weight=.1,p1_converged=False,engineering_chain_probe=True,engineering_P1_boundary=128,ramp=128,update_frequency=.25,steps=2,scheduled_prediction_probability=[.2,.8],first_hop_stopgrad=True,direct_stopgrad=True,formal='requires explicit converged-P1 declaration; smoke boundary is engineering exercise only'),phase=dict(max_stages=32,weight=.01,source='E83 Tier A strict cutoff <= anchor, current segment only, mask otherwise',clean_world_queries_only=True),recipe=recipe_config,evaluation='train fixed windows for world/adapter/phase only; existing dev real feedback for action only, no model language labels, no checkpoint selection',cache='read-only original overlay; no frame extraction or downloads')
        atomic(run/'contract.json',contract);refs=dict(contract=sha(run/'contract.json'))
        from .resource import Guard
        guard=Guard('E88_v31_bounded_smoke',3400);m=WorldModel().to('cuda:0');held=calibrate(m,small['P1'],guard,run);results={};initial=None;devcaps=json.loads((C/'artifacts/dev_v1/captures.json').read_text());devrows=json.loads((C/'artifacts/dev_v1/windows_v2.json').read_text());devrows=[dict(r,stratum=stratum(next(c for c in devcaps if c['canonical_id']==r['canonical_id']))) for r in devrows];devds=CachedDataset(devrows,devcaps,C/'artifacts/dev_features_v1',stats,lc)
        with (run/'updates.jsonl').open('x') as log:
            for name,stage,language,steps in [('P1','P1','short',160),('P3_none','P3','none',192),('P3_short','P3','short',192),('P3_long','P3','long',192)]:
                if stage=='P3':m.load_state_dict(initial);configure()
                ds=small[stage];sampler=CappedParentSampler(ds.rows,seed=88 if stage=='P1' else 89);opt=torch.optim.AdamW(m.parameters(),lr=3e-4 if stage=='P1' else 1e-4,weight_decay=1e-4);values=[];times=[];chaincalls=0
                m.train()
                for step in range(steps):
                    if step%16==0:
                        with torch.no_grad():av,ac=m.adapter_loss(held);m.adapter_ready.fill_(bool(av<=.1 and ac>=.95))
                    v,d=update(m,opt,sampler,ds,stage,step,language,guard,recipe=recipe_config);values.append(v);times.append(d['seconds']);chaincalls+=int(d['chain_weight']>0)
                    log.write(json.dumps(dict(name=name,step=step+1,loss=v,**d))+'\n')
                    if step==steps//2:
                        cp=run/(name+'_resume.pt');save(cp,m,opt,sampler,step+1,refs);v1,_=update(m,opt,sampler,ds,stage,step+1,language,guard,recipe=recipe_config);d1=digest(m);restore(cp,m,opt,sampler,refs);v2,_=update(m,opt,sampler,ds,stage,step+1,language,guard,recipe=recipe_config);d2=digest(m);atomic(run/(name+'_resume_check.json'),dict(loss_equal=v1==v2,parameters_equal=d1==d2,loss=v1,digest=d1,step=step+1));assert v1==v2 and d1==d2;restore(cp,m,opt,sampler,refs)
                    if step%16==0:
                        log.flush();atomic(run/'status.json',dict(status='running',name=name,step=step+1,loss=v,elapsed_s=guard.check()));print(name,step+1,round(v,5),flush=True)
                save(run/(name+'_end.pt'),m,opt,sampler,steps,refs)
                with torch.no_grad():av,ac=m.adapter_loss(held)
                results[name]=dict(steps=steps,first16=float(np.mean(values[:16])),last16=float(np.mean(values[-16:])),mean_step_s=float(np.mean(times[8:])),chain_updates=chaincalls,adapter_mse=float(av),adapter_cosine=float(ac),adapter_ready=bool(m.adapter_ready),sampler=sampler.state_dict())
                atomic(run/'stage_results.json',results)
                atomic(run/('world_'+name+'.json'),world_evaluate(m,full[stage],eval_ids(full[stage]),'cuda:0',guard,language))
                if stage=='P1':initial=copy.deepcopy(m.state_dict())
                else:atomic(run/('dev_'+name+'.json'),action_evaluate(m,devds,'cuda:0',guard))
        # Legacy fixed stage checkpoint comparison, read-only state, no resume claim.
        from e86.model import WAM
        legacy=WAM().to('cuda:0');oldpath=INPUT/'smoke_v3_02/P3_end.pt';legacy.load_state_dict(torch.load(oldpath,map_location='cpu',weights_only=False)['model']);atomic(run/'legacy_dev_gripper.json',action_evaluate(legacy,devds,'cuda:0',guard));byparent=collections.defaultdict(list)
        for row in full['P3'].rows:byparent[row['parent_component']].append(row)
        selected64=[vs[len(vs)//2] for parent,vs in sorted(byparent.items())][:64]
        train64=CachedDataset(selected64,p.captures,ready['cache'],stats,lc)
        atomic(run/'v31_train64_gripper.json',action_evaluate(m,train64,'cuda:0',guard));atomic(run/'legacy_train64_gripper.json',action_evaluate(legacy,train64,'cuda:0',guard))
        # Chain diagnostics on actual real shifted histories, whether or not train gate passed.
        diagnostics={};m.eval()
        for k in (25,50,100):
            candidates=[]
            for i,row in enumerate(small['P3'].rows):
                if len(candidates)>=32:break
                if small['P3'].index.resolve(row,2*k)[0] is not None:candidates.append(i)
            fn=make_chain(small['P3'],candidates,k,'long','P3',191,'cuda:0')
            if fn:
                with torch.no_grad():v,d=fn(m)
                diagnostics[str(k)]=dict(loss=float(v),**d)
            else:diagnostics[str(k)]=dict(status='no_valid_reentry_history')
        atomic(run/'chain_diagnostics.json',diagnostics);atomic(run/'status.json',dict(status='completed',elapsed_s=guard.check(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,formal_training=False))
    except BaseException as e:
        atomic(run/'status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc(),brt=brt()));raise
    finally:
        if guard:guard.finish()
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('run_id');parser.add_argument('--recipe',choices=['A','B'],default='A');args=parser.parse_args();main(args.run_id,args.recipe)
