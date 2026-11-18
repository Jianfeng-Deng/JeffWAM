"""Three CPU camera decoders feeding one GPU; exact E84 preprocessing and cache v1."""
import os,json,time,traceback,sys,queue,threading
from pathlib import Path
from .common import *
from .cache import stat

def main(planpath,cachepath,reuse=None):
    from .resource import Guard
    guard=Guard('feature_cache_pipelined',6500)
    try:
        import torch,numpy as np,cv2
        from concurrent.futures import ThreadPoolExecutor
        from .vision import FrozenEncoder,preprocess,PREPROCESS,MODEL
        torch.set_num_threads(1);cv2.setNumThreads(1);dest=Path(cachepath);dest.mkdir(parents=True,exist_ok=False)
        plan=json.loads(Path(planpath).read_text());contract=dict(plan_sha256=sha(planpath),preprocess=PREPROCESS,encoder_files={str(p):sha(p) for p in sorted(Path(MODEL).glob('*')) if p.suffix in ('.safetensors','.json','.model')},target_shape=[8,768],dtype='fp16');atomic(dest/'contract.json',contract)
        if reuse:
            old=Path(reuse);old_contract=json.loads((old/'contract.json').read_text());assert {k:v for k,v in old_contract.items() if k!='plan_sha256'}=={k:v for k,v in contract.items() if k!='plan_sha256'}
            reuse_records=[]
            for cap in plan['captures']:
                cid=cap['canonical_id'];p=old/cid
                if (p/'receipt.json').exists():
                    rec=json.loads((p/'receipt.json').read_text());assert rec['contract_sha256']==sha(old/'contract.json') or rec['contract_sha256'] in [sha(x) for x in [old.parent/'features_v1/contract.json'] if x.exists()]
                    assert rec['parent_component']==cap['parent_component'] and rec['split']==cap['split']
                    for ci,cam in enumerate(rec['cameras']):
                        assert cam['source_path']==cap['cameras'][ci]['path'];assert np.array_equal(np.load(p/(cam['camera']+'_rows.npy')),cap['cameras'][ci]['rows'])
                    reuse_records.append(dict(canonical_id=cid,receipt_sha256=sha(p/'receipt.json'),source=str(p.resolve())))
                    for cam in rec['cameras']:
                        for e in cam['files']:assert sha(e['path'])==e['sha256']
                    (dest/cid).symlink_to(p.resolve(),target_is_directory=True)
            atomic(dest/'reuse_verified.json',dict(source_contract=str(old/'contract.json'),source_contract_sha256=sha(old/'contract.json'),captures=reuse_records))
        encoder=FrozenEncoder('cuda:0');statuspath=dest/'status.json';completed=[];frames=0;fresh=0;beforefresh=time.monotonic();compute=0.;writes=0.;guardtime=0.;maxerr=0.;lastcheck=0.;newbytes=0
        def check():
            nonlocal lastcheck,guardtime
            if time.monotonic()-lastcheck>1:
                t=time.monotonic();guard.check();guardtime+=time.monotonic()-t;lastcheck=time.monotonic()
        for ci,capinfo in enumerate(plan['captures']):
            check();cid=capinfo['canonical_id'];out=dest/cid
            if out.exists():
                completed.append(cid);frames+=sum(len(c['rows']) for c in capinfo['cameras']);continue
            out.mkdir(exist_ok=False);recs=[None]*3;arrs=[]
            for c in capinfo['cameras']:
                n=len(c['rows']);arrs.append(np.lib.format.open_memmap(out/(c['camera']+'.npy'),mode='w+',dtype='float16',shape=(n,8,768)));np.save(out/(c['camera']+'_rows.npy'),np.asarray(c['rows'],dtype='i8'))
            q=queue.Queue(maxsize=6);stop=threading.Event()
            def put(item):
                while not stop.is_set():
                    try:q.put(item,timeout=.2);return
                    except queue.Full:continue
            def produce(k,c):
                path=c['path'];video=None
                try:
                    bs=stat(path);video=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);assert video.isOpened(),path;cursor=-1;pixels=[];valid=[];ixs=[];af=None;hw=None
                    for ix,rr in enumerate(c['rows']):
                        if stop.is_set():return
                        while cursor<rr:
                            ok=video.grab();cursor+=1
                            if not ok:raise IOError(f'decode {path}:{rr}')
                        ok,bgr=video.retrieve();assert ok
                        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);x,v,a=preprocess(rgb)
                        if af is None:af=a.tolist();hw=list(rgb.shape[:2])
                        pixels.append(x);valid.append(v);ixs.append(ix)
                        if len(ixs)==128:put(('batch',k,torch.stack(pixels),torch.stack(valid),ixs));pixels=[];valid=[];ixs=[]
                    if pixels:put(('batch',k,torch.stack(pixels),torch.stack(valid),ixs))
                    assert stat(path)==bs
                    put(('done',k,dict(camera=c['camera'],source_path=path,source_stat=bs,source_before_after_equal=True,encoded_frames=len(c['rows']),affine=af,original_hw=hw)))
                except BaseException as e:put(('error',k,e))
                finally:
                    if video is not None:video.release()
            pool=ThreadPoolExecutor(max_workers=3);tasks=[pool.submit(produce,k,c) for k,c in enumerate(capinfo['cameras'])];done=0
            try:
                while done<3:
                    check()
                    try:item=q.get(timeout=1)
                    except queue.Empty:continue
                    if item[0]=='error':raise item[2]
                    if item[0]=='done':recs[item[1]]=item[2];done+=1;continue
                    _,k,x,v,ixs=item;t=time.monotonic();z=encoder.images(x,v).cpu().numpy().astype('f2');compute+=time.monotonic()-t
                    if fresh==0:
                        replay=encoder.images(x,v).cpu().numpy().astype('f2');maxerr=float(np.max(np.abs(z.astype('f4')-replay.astype('f4'))));assert maxerr<=.002
                    t=time.monotonic();arrs[k][ixs]=z;writes+=time.monotonic()-t;fresh+=len(ixs);newbytes+=z.nbytes
                    if newbytes>100_000_000_000:raise RuntimeError('100GB total incremental cache hard stop')
            finally:stop.set();pool.shutdown(wait=True,cancel_futures=True)
            for k,c in enumerate(capinfo['cameras']):
                arrs[k].flush();recs[k]['files']=[dict(path=str(out/(c['camera']+suffix)),sha256=sha(out/(c['camera']+suffix))) for suffix in ('.npy','_rows.npy')]
            del arrs
            atomic(out/'receipt.json',dict(canonical_id=cid,parent_component=capinfo['parent_component'],split=capinfo['split'],contract_sha256=sha(dest/'contract.json'),cameras=recs,created_brt=brt()))
            completed.append(cid);frames+=sum(len(c['rows']) for c in capinfo['cameras']);elapsed=guard.check();speed=fresh/max(time.monotonic()-beforefresh,1e-6);remaining=(plan['unique_camera_frames']-frames)/speed
            status=dict(status='extracting',updated_brt=brt(),completed_captures=len(completed),total_captures=len(plan['captures']),encoded_camera_frames=frames,new_camera_frames=fresh,elapsed_s=elapsed,remaining_projected_s=remaining,throughput_frames_s=speed,compute_s=compute,write_s=writes,guard_s=guardtime,max_repeat_error=maxerr,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,new_bytes=newbytes)
            atomic(statuspath,status)
            if ci%5==0:print(json.dumps(status),flush=True)
            if fresh>12000 and remaining>min(guard.limit,10800-guard.ledger['elapsed_s'])-elapsed-500:raise RuntimeError('Pipelined full cache still projected beyond budget; stopped at complete capture')
        status.update(status='ready',plan_sha256=sha(planpath),contract_sha256=sha(dest/'contract.json'),camera_frames=frames,captures=len(completed),files_include_reused_symlinks=bool(reuse));atomic(dest/'READY_features.json',status);atomic(statuspath,status)
    except BaseException as e:
        if 'statuspath' in locals():atomic(statuspath,dict(status='stopped',reason=str(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-guard.started,updated_brt=brt()))
        raise
    finally:guard.finish()
if __name__=='__main__':main(*sys.argv[1:])
