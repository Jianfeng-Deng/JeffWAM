import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys
sys.dont_write_bytecode=True
import numpy as np,cv2,json
from common import *
from metrics import *
rng=np.random.default_rng(20261112)
for _ in range(20):
    score=rng.integers(0,5,size=(12,12));mask=rng.random((12,12))>.7;a=score[mask][:,None];b=score[~mask][None,:]
    expected=float(np.mean((a>b)+.5*(a==b)));actual=localization(score,mask)['auroc'];assert abs(expected-actual)<1e-12
dirs=np.stack([np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)],-1);assert planar_follow(dirs,dirs).all();assert not planar_follow(dirs,-dirs).any();assert not planar_follow(dirs*0,dirs).any()
assert planar_follow(dirs,np.roll(dirs,1,axis=0)).sum()==0
path=P/'synthetic_decoder.avi';w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48));assert w.isOpened()
for i in range(6):w.write(np.full((48,64,3),i*30,dtype='u1'))
w.release();cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened();means=[]
for i in range(6):assert cap.grab();ok,im=cap.retrieve();assert ok;means.append(float(im.mean()))
cap.release();assert np.max(np.abs(np.array(means)-np.arange(6)*30))<3
obj={'timestamp_brt':brt(),'status':'passed','synthetic_only':True,'rank_auroc_random_tie_checks':20,'direction_boundary_rule':'same floor(theta+pi/8) bins as F; zero rejected','decoder_threads':1,'cv2_version':cv2.__version__,'synthetic_frame_means':means}
atomic(P/'utility_checks.json',obj);print(obj)
