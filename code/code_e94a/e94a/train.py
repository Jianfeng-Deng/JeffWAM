import argparse,copy,json,random,time,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from e86.data import move,collate
from e86.common import generator
from .common import *
from e90.data import load_data,sample_horizons
from .model import WorldModel,adapter_pass
from e90.sampling import MultiSourceSampler
from .objective import loss,chain_loss
from .checkpoint import save,restore,initialize

def configure():
    torch.set_num_threads(2);torch.manual_seed(90);np.random.seed(90);random.seed(90)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False

def config_for(smoke=False,recipe="M1"):
    return dict(schema='e94a-p1-config-v1',recipe=recipe,readout_hidden=1024,mean_weight=1.,fm_weight=1. if recipe=='M2' else 0.,seed=90,steps=32 if smoke else 5000,batch=32,lr=3e-4,
                weight_decay=1e-4,gradient_clip=1.,scheduler='constant LambdaLR(1), E88 fixed LR',
                max_horizon=100,enable_200_300=False,continuous_horizon=True,
                phase_weight=0.,alignment_weight=0.,adapter_weight=.1,cosine_weight=.1,
                history_dropout=.15,state_dropout=.1,state_noise=.02,
                mode_probabilities=[.25]*4,clean_aux=0.,deployment='action_only',
                chain_warmup=16 if smoke else 3000,chain_ramp=8 if smoke else 1000,chain_maximum=.1,
                chain_frequency=1,chain_batch=2,chain_flow_steps=2,eval_flow_steps=4,
                chain_schedule_sampling=[.2,.8],chain_pairs=[[25,25,50],[50,50,100]],
                source_weights={'piper':1.},source_caps={},demo3_cap=.45,
                text='existing E83 task instruction in its original range; fixed short fallback; no narration; dev short only',
                frozen_encoder='SigLIP2 base-patch16-224 vision+text, cached fixed 2x4 features, no LoRA',
                no_action_training=True,checkpoint_interval=16 if smoke else 500,smoke=smoke)

def digest(model):
    return hashlib.sha256(b''.join(x.detach().cpu().contiguous().numpy().tobytes() for _,x in sorted(model.state_dict().items()))).hexdigest()

def calibration_raws(data):
    ids=[];seen=set()
    for i,r in enumerate(data.rows):
        if r['parent_component'] not in seen:ids.append(i);seen.add(r['parent_component'])
        if len(ids)==80:break
    return torch.stack([data.base[i]['obs'][4] for i in ids]).to('cuda:0'),ids

def calibrate(model,raws,guard,run,label):
    opt=torch.optim.Adam(model.reentry.parameters(),lr=.001);history=[];was=model.training;model.eval()
    for step in range(1200):
        guard.check();v,_=model.adapter_loss(raws[:64]);opt.zero_grad();v.backward();opt.step()
        if step%25==0:
            with torch.no_grad():hv,hc=model.adapter_loss(raws[64:])
            history.append(dict(step=step+1,nmse=float(hv),cosine=float(hc)))
            if adapter_pass(hv,hc):break
    with torch.no_grad():hv,hc=model.adapter_loss(raws[64:])
    passed=adapter_pass(hv,hc);model.adapter_ready.fill_(passed);model.train(was)
    result=dict(enabled=passed,normalized_mse=float(hv),cosine=float(hc),steps=step+1,
                threshold=dict(nmse_max=.1,cosine_min=.95),history=history,fit_train_parents=64,heldout_train_parents=16,no_dev=True)
    atomic(run/('adapter_'+label+'.json'),result);return passed

def make_chain(data,ids,k,step,cfg):
    chosen=[];shifted=[]
    for i in ids:
        if data.index.resolve(data.rows[i],2*k)[0] is None:continue
        b=data.shifted(i,k,'instruction','P1')
        if b is not None and b['world_valid'].all():chosen.append(i);shifted.append(b)
        if len(chosen)==cfg['chain_batch']:break
    if not chosen:return None
    first=move(data.batch(chosen,(k,),'instruction'),'cuda:0');direct=move(data.batch(chosen,(2*k,),'instruction'),'cuda:0')
    second=collate(shifted);second['world_positions']=second['world_positions'][0];second=move(second,'cuda:0')
    check=second['world_raw']
    if k==50:check=check.reshape(len(chosen),3,8,-1).mean(2)
    if not torch.allclose(check,direct['world_raw'],atol=1e-6):raise ValueError('Chain exact endpoint mismatch')
    fraction=max(0.,min(1.,(step-cfg['chain_warmup'])/max(1,cfg['steps']-cfg['chain_warmup'])))
    p=.2+.6*fraction;predicted=bool(torch.rand((),generator=generator(90,step,'reentry')).item()<p)
    return lambda model:chain_loss(model,first,second,direct,k,predicted,steps=cfg['chain_flow_steps'],seed=90+step)

def update(model,opt,scheduler,sampler,data,step,cfg,guard):
    guard.check();started=time.perf_counter();ids=sampler.sample(cfg['batch'])
    hs=sample_horizons(step,maximum=cfg['max_horizon'],continuous=cfg['continuous_horizon'])
    b=move(data.batch(ids,hs,'instruction'),'cuda:0');chain=None
    if step>cfg['chain_warmup'] and bool(model.adapter_ready):chain=make_chain(data,ids,(25,50)[step%2],step,cfg)
    opt.zero_grad(set_to_none=True)
    value,metrics=loss(model,b,'P1',step,seed=90,chain=chain,chain_warmup=cfg['chain_warmup'],
                       chain_ramp=cfg['chain_ramp'],chain_maximum=cfg['chain_maximum'],p1_converged=True,
                       mode_probabilities=tuple(cfg['mode_probabilities']),clean_aux_probability=cfg['clean_aux'],
                       phase_weight=cfg['phase_weight'],alignment_weight=cfg['alignment_weight'],
                       adapter_weight=cfg['adapter_weight'],cosine_weight=cfg['cosine_weight'])
    if not torch.isfinite(value):raise RuntimeError('Nonfinite loss')
    value.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
    if not torch.isfinite(gn):raise RuntimeError('Nonfinite gradient')
    opt.step();scheduler.step();torch.cuda.synchronize()
    metrics.update(loss=float(value.detach()),step=step,seconds=time.perf_counter()-started,horizons=list(hs),
                   learning_rate=opt.param_groups[0]['lr'],chain_candidates=chain is not None,chain_k=(25,50)[step%2] if chain else None)
    return metrics

def build(cfg,data):
    m=WorldModel(recipe=cfg['recipe'],readout_hidden=cfg['readout_hidden']).to('cuda:0');opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    sampler=MultiSourceSampler(data.rows,seed=cfg['seed'],weights=cfg['source_weights'],caps=cfg['source_caps'],demo3_cap=cfg['demo3_cap'])
    return m,opt,scheduler,sampler

def main(runid,smoke=False,resume=None,init_ckpt=None,recipe="M1"):
    run=T/"results"/runid;run.mkdir(exist_ok=False);guard=None
    try:
        cfg=config_for(smoke,recipe);p,ready,data,dev,coverage=load_data();configure()
        manifest=json.loads((T/'code_e90/artifacts/preflight_01/manifest.json').read_text());mh=manifest['manifest_hash']
        from .resource import Guard
        guard=Guard('E94a_'+recipe,1000 if smoke else 3500)
        m,opt,scheduler,sampler=build(cfg,data);start=0;extra={}
        if init_ckpt:extra['initialization']=initialize(init_ckpt,m)
        if resume:start,extra=restore(resume,m,opt,scheduler,sampler,cfg,mh)
        atomic(run/'config.json',cfg);atomic(run/'provenance.json',dict(code_hash=source_hash(),manifest_hash=mh,config_hash=object_hash(cfg),git='T is not a git repository',created_brt=brt(),initialization=extra.get('initialization'),resume=str(resume)))
        raws,raw_ids=calibration_raws(data)
        if not resume:extra['initial_adapter_pass']=calibrate(m,raws,guard,run,'initial')
        values=[];chain_updates=0;gate_disabled=not extra['initial_adapter_pass'] or not extra.get('boundary_adapter_pass',True);m.train()
        with (run/'updates.jsonl').open('x') as log:
            for step in range(start+1,cfg['steps']+1):
                if step==cfg['chain_warmup']:
                    passed=calibrate(m,raws,guard,run,'chain_boundary')
                    gate_disabled=gate_disabled or not passed;extra['boundary_adapter_pass']=passed
                if step%16==0:
                    with torch.no_grad():av,ac=m.adapter_loss(raws[64:])
                    passed=adapter_pass(av,ac)
                    # A failed initial/boundary gate disables this entire run's chain.
                    m.adapter_ready.fill_(passed and not gate_disabled)
                    extra['last_adapter_check']=dict(step=step,nmse=float(av),cosine=float(ac),passed=passed)
                metrics=update(m,opt,scheduler,sampler,data,step,cfg,guard);values.append(metrics['loss']);chain_updates+=int(metrics['chain_weight']>0)
                log.write(json.dumps(metrics)+'\n')
                if step%10==0 or step==1:
                    log.flush();atomic(run/'status.json',dict(status='training',step=step,total=cfg['steps'],loss=metrics['loss'],elapsed_s=guard.check(),brt=brt()))
                    print(json.dumps(dict(step=step,loss=round(metrics['loss'],6),seconds=round(metrics['seconds'],3),chain_weight=metrics['chain_weight'])),flush=True)
                if step%cfg['checkpoint_interval']==0:
                    cp=run/f'ckpt_{step:05d}.pt';save(cp,m,opt,scheduler,sampler,step,cfg,mh,extra)
                    if smoke and step==16:
                        a=update(m,opt,scheduler,sampler,data,step+1,cfg,guard);d=digest(m)
                        restore(cp,m,opt,scheduler,sampler,cfg,mh)
                        b=update(m,opt,scheduler,sampler,data,step+1,cfg,guard);d2=digest(m)
                        result=dict(loss_equal=a['loss']==b['loss'],parameters_equal=d==d2,loss=a['loss'],digest=d,checkpoint=str(cp))
                        atomic(run/'resume_check.json',result)
                        if not result['loss_equal'] or not result['parameters_equal']:raise RuntimeError('Resume mismatch')
                        restore(cp,m,opt,scheduler,sampler,cfg,mh)
                if step%500==0:report(f'训练进度：{runid} step={step}，最近loss={metrics["loss"]:.6f}，累计GPU秒={guard.check():.1f}。')
        end=run/f'ckpt_{cfg["steps"]:05d}.pt'
        summary=dict(steps=cfg['steps'],first32=float(np.mean(values[:32])),last32=float(np.mean(values[-32:])),
                     chain_updates=chain_updates,adapter_gate_disabled=gate_disabled,adapter_last=extra.get('last_adapter_check'),
                     checkpoint=str(end),checkpoint_sha256=sha(end),code_hash=source_hash(),manifest_hash=mh,config_hash=object_hash(cfg),
                     sampler=sampler.state_dict(),trainable_parameters=sum(x.numel() for x in m.parameters()),shared_core_parameters=sum(x.numel() for x in m.blocks.parameters()))
        atomic(run/'train_summary.json',summary)
        atomic(run/'status.json',dict(status='completed',step=cfg['steps'],elapsed_s=guard.check(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,brt=brt()))
        report(('烟雾' if smoke else '正式P1')+'完成：'+json.dumps(summary,ensure_ascii=False))
    except BaseException as e:
        atomic(run/'status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc(),brt=brt()));raise
    finally:
        if guard:guard.finish()

def replay(smoke_run):
    run=ART/smoke_run;cfg=json.loads((run/'config.json').read_text());expected=json.loads((run/'resume_check.json').read_text())
    _,_,data,_,_=load_data();configure();manifest=json.loads((T/'code_e90/artifacts/preflight_01/manifest.json').read_text());guard=None
    try:
        from .resource import Guard
        guard=Guard('E90_independent_resume_check',300);m,opt,scheduler,sampler=build(cfg,data)
        step,_=restore(expected['checkpoint'],m,opt,scheduler,sampler,cfg,manifest['manifest_hash']);m.train()
        actual=update(m,opt,scheduler,sampler,data,step+1,cfg,guard)
        result=dict(loss_equal=actual['loss']==expected['loss'],parameters_equal=digest(m)==expected['digest'],loss=actual['loss'])
        atomic(run/'independent_resume_check.json',result)
        if not all(result[k] for k in ('loss_equal','parameters_equal')):raise RuntimeError('Independent resume mismatch')
    finally:
        if guard:guard.finish()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--smoke',action='store_true');ap.add_argument('--replay',action='store_true')
    ap.add_argument('--recipe',choices=['M1','M2'],default='M1');ap.add_argument('--resume');ap.add_argument('--init-ckpt');ap.add_argument('--mode-probabilities',nargs=4,type=float,default=[.25]*4);ap.add_argument('--clean-aux',type=float,default=0.)
    a=ap.parse_args()
    if a.mode_probabilities!=[.25]*4 or a.clean_aux!=0:ap.error('E90 locks recipe B: 0.25 each, clean-aux=0')
    if a.resume and a.init_ckpt:ap.error('resume and init-ckpt are mutually exclusive')
    replay(a.run_id) if a.replay else main(a.run_id,a.smoke,a.resume,a.init_ckpt,a.recipe)
