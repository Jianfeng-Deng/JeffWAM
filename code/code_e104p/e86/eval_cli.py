import argparse,json,time,numpy as np
from pathlib import Path
from .common import *
def main(run_id):
    from .resource import Guard
    guard=Guard('GPU_latency_'+run_id,1200)
    run=C/'artifacts'/run_id;dest=run/'latency_v1';dest.mkdir(exist_ok=False)
    try:
        import torch,cv2
        from .model import WAM,Config
        from .data import CachedDataset,collate,move,CAMS,Package
        from .language import load as load_language
        from .vision import FrozenEncoder
        from .deploy import infer_images
        from .inference import predict
        from .evaluation import evaluate
        torch.set_num_threads(2);cv2.setNumThreads(1);m=WAM().eval().to('cuda:0');ckpt=torch.load(run/'P3_end.pt',map_location='cpu',weights_only=False);m.load_state_dict(ckpt['model'])
        stats=json.loads((C/'artifacts/stats_v1.json').read_text());dev=C/'artifacts/dev_v1';ws=json.loads((dev/'windows_v2.json').read_text());cs=json.loads((dev/'captures.json').read_text());lc=load_language([],C/'artifacts/language_v1');ds=CachedDataset(ws,cs,C/'artifacts/dev_features_v1',stats,lc)
        # Use eight fixed dev windows, original raw frame numbers. No final-test source is opened.
        p=Package();contract=json.loads((run/'contract.json').read_text());assert contract['package']==p.refs;assert contract['stats']==sha(C/'artifacts/stats_v1.json');records=[];selected=list(range(min(8,len(ds))))
        def rgb_for(row):
            images=[];r=p.routes[row['canonical_id']]
            for k,cam in enumerate(CAMS):
                path=next(x['path'] for x in r['cameras'] if x['camera']==cam);v=cv2.VideoCapture(path);rr=row['obs_rows'][4][k];v.set(cv2.CAP_PROP_POS_FRAMES,rr);actual=v.get(cv2.CAP_PROP_POS_FRAMES);ok,frame=v.read();v.release()
                if not ok or abs(actual-rr)>.1:raise IOError('Untraceable latency source frame')
                images.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            return images
        encoder=FrozenEncoder('cuda:0')
        for mode in ('action_only','joint'):
            values=[];kernel=[];waits=[];completion_ages=[];request_ages=[]
            for j in range(208):
                guard.check();i=selected[j%len(selected)];b=move(collate([ds[i]]),'cuda:0');row=ds.rows[i]
                torch.cuda.synchronize();t=time.perf_counter();pred=predict(m,b,mode);torch.cuda.synchronize();cached_ms=(time.perf_counter()-t)*1000
                # Replay measured state delay; real wait is included, as are source decode/CPU preprocessing/current encoder/four-noise integration.
                torch.cuda.synchronize();t=time.perf_counter();available_at=time.monotonic()+row['decision_wait_s'];rgb=rgb_for(row);pred,meta=infer_images(m,encoder,b,rgb,available_at,mode);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
                assert torch.isfinite(pred).all()
                if j>=8:
                    values.append(ms);kernel.append(cached_ms);waits.append(meta['state_wait_s']*1000);age=max(x[1] for x in row['obs_meta'][4])*1000;request_ages.append(age);completion_ages.append(max(0.,ms-row['decision_wait_s']*1000+age))
                if j%16==0:atomic(dest/'status.json',dict(status='measuring',mode=mode,calls=max(0,j-7),elapsed_s=guard.check(),updated_brt=brt()))
            quant=lambda x:dict(calls=len(x),p50_ms=float(np.quantile(x,.5)),p95_ms=float(np.quantile(x,.95)),p99_ms=float(np.quantile(x,.99)),max_ms=float(max(x)),miss_333ms=float(np.mean(np.asarray(x)>333)))
            record=dict(mode=mode,full_pipeline=quant(values),cached_features_policy=quant(kernel),completion_image_age=quant(completion_ages),current_image_age_at_decision_ms_mean=float(np.mean(request_ages)),state_wait_replayed_ms=float(np.mean([ds.rows[i]['decision_wait_s']*1000 for i in selected])),state_wait_remaining_after_decode_ms=float(np.mean(waits)),noise_samples=4,euler_steps=10,history='cached causal images only',text='cached original human instruction',images='three real current RGB decoded in place each call',scope='shared GPU offline replay; not online scheduler/robot timing',peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            records.append(record);atomic(dest/(mode+'.json'),record);print(json.dumps(record),flush=True)
        atomic(dest/'READY_latency.json',dict(status='ready',results=records,elapsed_s=guard.check(),test_sources_read=0))
    except BaseException as e:
        import traceback
        atomic(dest/'status.json',dict(status='stopped',reason=str(e),traceback=traceback.format_exc()));raise
    finally:guard.finish()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('run_id');main(a.parse_args().run_id)
