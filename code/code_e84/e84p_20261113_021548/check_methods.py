import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,cv2,json
from metrics import localization,parent_ci,planar_follow,paired_ci
from common import *
# Nontrivial metric edge cases: ties get expected credit; balanced ranking is exact.
a=localization(np.array([[1.,1.],[0.,0.]]),np.array([[True,False],[False,True]]));assert a['pointing']==.5 and a['auroc']==.5,a
b=localization(np.array([[3.,2.],[1.,0.]]),np.array([[True,True],[False,False]]));assert b['auroc']==1 and b['pointing']==1,b
# Parent weighting must differ from micro weighting under unequal cluster sizes.
c=parent_ci([1,1,1,0],['a','a','a','b']);assert c['estimate']==.5 and c['micro_estimate']==.75,c
# Counterfactual success excludes stationary outputs even if their angle is numerically0.
assert not planar_follow([[0,0]],[[1,0]])[0]
assert planar_follow([[.01,0]],[[1,0]])[0]
# Inverse affine transport preserves the known center location of an image-space peak.
x=np.zeros((224,224),np.float32);x[100,140]=1;a=np.array([[.35,0,0],[0,.35,28]],np.float32)
y=cv2.warpAffine(x,a,(640,480),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
q=np.unravel_index(y.argmax(),y.shape);assert np.linalg.norm(np.array(q[::-1])-np.array([400,205.714]))<2,q
d=paired_ci([True,False,True],[False,False,True],['a','a','b']);assert d['estimate']==.25 and abs(d['micro_estimate']-1/3)<1e-12,d
assert paired_ci([False],[True],['a'])['estimate']==-1
atomic(P/'method_checks.json',{'timestamp_brt':brt(),'status':'passed','checks':['tie credit and AUROC','perfect rank AUROC','parent weighting versus micro','no stationary counterfactual successes','inverse-affine map transport','boolean paired contrasts preserve signed differences']})
print('METHOD_CHECKS passed6')
