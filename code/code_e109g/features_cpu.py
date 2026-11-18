import os,sys,json,time,traceback,collections
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';sys.path.insert(0,str(T/'code_e106c'))
import numpy as np,torch,cv2
from e86.vision import FrozenEncoder,preprocess,PREPROCESS
from e86.data import Package
from e96a.train import configure
from e86.common import sha

def main():
 start=time.monotonic();configure();torch.backends.mha.set_fastpath_enabled(False);audit=json.loads((O/'CACHE_AUDIT.json').read_text());p=Package();enc=FrozenEncoder('cpu');data={};receipts=[];referrs=[]
 for root in audit['roots']:
  ct=json.loads((Path(root)/'contract.json').read_text());assert ct['preprocess']==PREPROCESS
  for path,h in ct['encoder_files'].items():assert sha(Path(path))==h
 for j,x in enumerate(audit['missing']):
  cid=x['cid'];cam=x['cam'];route=next(c for c in p.routes[cid]['cameras'] if c['camera']==cam);path=route['path'];stat=Path(path).stat();wanted=x['rows'];refs=[];reference={}
  for root in audit['roots']:
   f=Path(root)/cid/(cam+'_rows.npy')
   if f.exists():
    ix=np.load(f);feat=np.load(f.with_name(cam+'.npy'),mmap_mode='r');near=int(ix[np.argmin(abs(ix-np.median(wanted)))]);refs.append(near);reference[near]=feat[np.where(ix==near)[0][0]].astype('f4');break
  ids=sorted(set(wanted+refs));video=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert video.isOpened();cursor=-1;pixels=[];valid=[]
  for rr in ids:
   while cursor<rr:ok=video.grab();cursor+=1;assert ok
   ok,bgr=video.retrieve();assert ok;xx,v,_=preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));pixels.append(xx);valid.append(v)
  video.release();features=[]
  for k in range(0,len(ids),8):features.extend(enc.images(torch.stack(pixels[k:k+8]),torch.stack(valid[k:k+8])).cpu().numpy())
  out={r:f for r,f in zip(ids,features)}
  for rr,ref in reference.items():referrs.append(dict(cid=cid,cam=cam,row=rr,max_abs=float(abs(out[rr]-ref).max()),mean_abs=float(abs(out[rr]-ref).mean())))
  key=cid+'__'+cam;data[key+'__rows']=np.array(wanted,dtype='i8');data[key+'__features']=np.array([out[r] for r in wanted],dtype='f2');after=Path(path).stat();assert (stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns);receipts.append(dict(cid=cid,cam=cam,frames=len(wanted),source=path,source_unchanged=True));print(j+1,len(audit['missing']),round(time.monotonic()-start,1),flush=True)
 assert not torch.cuda.is_initialized();target=O/'missing_features_cpu.npz';assert not target.exists();np.savez_compressed(target,**data);(O/'FEATURES_COMPLETE.json').write_text(json.dumps(dict(status='complete',frames=sum(x['frames'] for x in receipts),reference_roundtrip=referrs,receipts=receipts,elapsed_s=time.monotonic()-start,GPUh=0),indent=2))
if __name__=='__main__':
 try:main()
 except BaseException:(O/'FEATURES_FAILED.json').write_text(traceback.format_exc());raise
