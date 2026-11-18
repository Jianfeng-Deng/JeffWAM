import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np,torch,json
from common import *
from encoder import Encoder,preprocess,PROCESSOR
from metrics import *
from heads import *
torch.set_num_threads(2)
np.random.seed(20261112);torch.manual_seed(20261112)
enc=Encoder('cpu')
x=torch.randn(2,196,768)
with torch.inference_mode():
    expected=F.normalize(enc.model.vision_model.head(x.reshape(-1,1,768)).reshape(2,196,768),dim=-1).numpy()
actual=enc.patch_vectors(x.numpy());err=float(np.abs(actual-expected).max());assert err<1e-5,err
pixel,af,rect=preprocess(np.zeros((720,1280,3),dtype=np.uint8));assert pixel.shape==(3,224,224)
assert np.allclose(af@np.array([1280,720,1]),[224,175,1]),(af,rect)
g=np.random.randn(5,5,3,768).astype('f4');v=np.ones((5,5,3),dtype=bool);features=observations(g,v);assert features.shape==(5,6912)
mask=np.zeros((14,14),bool);mask[3:5,4:6]=True
assert localization(mask.astype(float),mask)['auroc']==1
assert localization(np.zeros((14,14)),mask)['auroc']==.5
assert abs(localization(np.zeros((14,14)),mask)['pointing']-4/196)<1e-8
ci=parent_ci([1,1,0],['a','a','b']);assert ci['estimate']==.5
head=AlignmentHead();assert head(torch.from_numpy(features)).shape==(5,768)
result={'timestamp_brt':brt(),'synthetic_only':True,'singleton_optimized_vs_literal_max_abs':err,'alignment_head_params':sum(p.numel() for p in head.parameters()),'affine':af.tolist(),'valid_rect':rect.tolist(),'metrics_checked':['tie-safe pointing','tie-safe AUROC','equal-parent bootstrap','history shape','inverse spatial transform dimensions'],'status':'passed','processor':PROCESSOR}
atomic(P/'synthetic_checks.json',result);append_report('## 合成输入与指标验证\n\n'+json.dumps({k:result[k] for k in ['timestamp_brt','status','singleton_optimized_vs_literal_max_abs','alignment_head_params','metrics_checked']},ensure_ascii=False)+'。仅合成输入；不作为真实实验结果。官方processor元数据缺失已通过官方公开394B配置文档核实：bilinear/224/.5归一化，未下载权重/数据，未改模型目录。空间padding为显式实验选择，并非NaFlex。')
print(json.dumps(result,ensure_ascii=False))
