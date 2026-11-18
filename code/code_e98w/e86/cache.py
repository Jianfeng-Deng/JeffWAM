"""Bounded, resumable per-capture frozen features; no source mutation or copying videos."""
import os,json,time,traceback,sys
from pathlib import Path
from .common import *
def stat(p):
    s=Path(p).stat();return [s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino]
def main(planpath,cachepath,phase_seconds=7200):
    # Guard chooses and pins physical GPU before importing torch/transformers.
    from .resource import Guard
    guard=Guard('feature_cache',phase_seconds)
    try:
        import torch,numpy as np,cv2
        from .vision import FrozenEncoder,preprocess,PREPROCESS,MODEL
        torch.set_num_threads(2);cv2.setNumThreads(1)
        plan=json.loads(Path(planpath).read_text());dest=Path(cachepath)
        if not dest.exists():dest.mkdir(parents=True,exist_ok=False)
        elif not (dest/'contract.json').exists():raise ValueError('Existing cache without own contract')
        contract=dict(plan_sha256=sha(planpath),preprocess=PREPROCESS,encoder_files={str(p):sha(p) for p in sorted(Path(MODEL).glob('*')) if p.suffix in ('.safetensors','.json','.model')},target_shape=[8,768],dtype='fp16')
        if (dest/'contract.json').exists():assert json.loads((dest/'contract.json').read_text())==contract
        else:atomic(dest/'contract.json',contract)
        encoder=FrozenEncoder('cuda:0');completed=[];frames=0;newframes=0;started=time.monotonic();maxerr=0.;timings=[];bytes_written=0
        statuspath=dest/'status.json';atomic(statuspath,dict(status='extracting',started_brt=brt(),plan_sha256=sha(planpath)))
        for capidx,c in enumerate(plan['captures']):
            guard.check();cid=c['canonical_id'];out=dest/cid
            if (out/'receipt.json').exists():
                receipt=json.loads((out/'receipt.json').read_text());assert receipt['contract_sha256']==sha(dest/'contract.json')
                for rec in receipt['cameras']:
                    for e in rec['files']:assert sha(e['path'])==e['sha256']
                completed.append(cid);frames+=sum(len(cam['rows']) for cam in c['cameras']);continue
            if out.exists():raise ValueError('Partial capture requires new run ID, preserve failed evidence: '+cid)
            out.mkdir(exist_ok=False);recs=[]
            for ci,cam in enumerate(c['cameras']):
                guard.check();path=cam['path'];before=stat(path);ids=cam['rows'];n=len(ids);frames+=n
                assert bytes_written+n*8*768*2<min(100_000_000_000,150_000_000_000)
                disk=os.statvfs(dest);assert disk.f_bavail*disk.f_frsize>plan['estimated_bytes']-bytes_written+5_000_000_000,'Insufficient shared filesystem headroom'
                arr=np.lib.format.open_memmap(out/(cam['camera']+'.npy'),mode='w+',dtype='float16',shape=(n,8,768));np.save(out/(cam['camera']+'_rows.npy'),np.array(ids,dtype='i8'))
                video=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);assert video.isOpened(),path
                pixels=[];valid=[];meta=[];affine=None;cursor=-1
                def flush():
                    nonlocal pixels,valid,meta,newframes,maxerr
                    if not pixels:return
                    guard.check();x=torch.stack(pixels);v=torch.stack(valid);t=time.monotonic();features=encoder.images(x,v);torch.cuda.synchronize();timings.append((len(meta),time.monotonic()-t))
                    z=features.cpu().numpy().astype('f2');arr[meta]=z
                    if newframes==0:
                        # Actual frozen encoder, identical batch replay then fp16 cache roundtrip.
                        repeat=encoder.images(x,v).cpu().numpy().astype('f2');maxerr=float(np.max(np.abs(z.astype('f4')-repeat.astype('f4'))));assert maxerr<=.002
                    newframes+=len(meta);pixels=[];valid=[];meta=[]
                for ix,rr in enumerate(ids):
                    while cursor<rr:
                        ok=video.grab();cursor+=1
                        if not ok:raise IOError(f'video decode {cid}/{cam["camera"]}:{rr}')
                    ok,bgr=video.retrieve();assert ok
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);x,v,af=preprocess(rgb)
                    if affine is None:affine=af.tolist();hw=list(rgb.shape[:2])
                    pixels.append(x);valid.append(v);meta.append(ix)
                    if len(meta)>=64:flush()
                flush();arr.flush();del arr;video.release();assert stat(path)==before,'Source changed'
                recs.append(dict(camera=cam['camera'],source_path=path,source_stat=before,source_before_after_equal=True,encoded_frames=n,affine=affine,original_hw=hw,files=[dict(path=str(out/(cam['camera']+suffix)),sha256=sha(out/(cam['camera']+suffix))) for suffix in ('.npy','_rows.npy')]))
                bytes_written+=n*8*768*2
            receipt=dict(canonical_id=cid,parent_component=c['parent_component'],split=c['split'],contract_sha256=sha(dest/'contract.json'),cameras=recs,created_brt=brt());atomic(out/'receipt.json',receipt);completed.append(cid)
            elapsed=guard.check();status=dict(status='extracting',updated_brt=brt(),completed_captures=len(completed),total_captures=len(plan['captures']),encoded_camera_frames=frames,new_frames=newframes,elapsed_s=elapsed,expected_total_s=elapsed*plan['unique_camera_frames']/max(frames,1),max_repeat_error=maxerr,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,bytes_written=bytes_written)
            atomic(statuspath,status)
            if capidx%5==0:print(json.dumps(status),flush=True)
            remaining=(time.monotonic()-started)*(plan['unique_camera_frames']-frames)/max(newframes,1)
            available=min(guard.limit,10800-guard.ledger['elapsed_s'])-elapsed
            if capidx>=2 and remaining>available-600:raise RuntimeError(f'Full feature cache projected {remaining:.0f}s exceeds remaining extraction budget {available:.0f}s; stopped')
        ready=dict(status='ready',plan_sha256=sha(planpath),contract_sha256=sha(dest/'contract.json'),captures=len(completed),camera_frames=frames,new_frames=newframes,bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),elapsed_s=guard.check(),max_repeat_error=maxerr,gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,inference_images_per_s=sum(n for n,t in timings)/sum(t for n,t in timings));atomic(dest/'READY_features.json',ready);atomic(statuspath,ready);print(json.dumps(ready),flush=True)
    except Exception as e:
        if 'statuspath' in locals():atomic(statuspath,dict(status='stopped',reason=str(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-guard.started,updated_brt=brt()))
        raise
    finally:guard.finish()
if __name__=='__main__':main(*sys.argv[1:])
