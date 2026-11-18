import numpy as np,torch,cv2,json,os
from common import *
from encoder import preprocess
from contract import sha

def extract_pixels(encoder,dest,guard,budget):
    out=dest/'pixel_150.npz'
    if out.exists():return
    rows=[json.loads(x) for x in (R/'pixel_gt/pixel_gt.jsonl').read_text().splitlines()]
    split=json.loads((R/'split/probe_split.json').read_text())
    assert len(rows)==150
    batch=[];meta=[];patches=[];glob=[];aff=[];rect=[];shapes=[];ids=[]
    def flush():
        if not batch:return
        guard.check();g,p=encoder.images(torch.stack(batch));patches.extend(p);glob.extend(g)
        batch.clear()
    sources={}
    for row in rows:
        assert split['parents'][row['parent_id']]==row['probe_split']
        f=pathlib.Path(row['decoded_frame']);sources[str(f)]=sha(f)
        bgr=cv2.imread(str(f));assert bgr is not None
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert list(rgb.shape[1::-1])==row['resolution']
        pix,a,r=preprocess(rgb);batch.append(pix);aff.append(a);rect.append(r);shapes.append(rgb.shape[:2]);ids.append(row['id'])
        if len(batch)==8:flush()
    flush()
    tmp=str(out)+'.tmp'
    with open(tmp,'wb') as f:np.savez(f,ids=np.array(ids),patch_features=np.array(patches,dtype='f2'),global_features=np.array(glob,dtype='f2'),affine=np.array(aff),valid_rect=np.array(rect),original_hw=np.array(shapes))
    os.replace(tmp,out)
    atomic(dest/'READY_pixel_features.json',{'timestamp_brt':brt(),'path':str(out),'sha256':sha(out),'frames':150,'source_frame_sha256':sources,'label_coordinates_used_for_extraction':False})
    update(pixel_features_ready=True,model_frames=150,status='pixel_features_ready')
    print('PIXEL_FEATURES',len(rows),flush=True)
