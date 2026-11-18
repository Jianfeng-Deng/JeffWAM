import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import sys
sys.dont_write_bytecode=True
import torch,numpy as np
from common import *
from train_heads import *
from lora_module import LoRALinear
from metrics import *
torch.set_num_threads(2);torch.manual_seed(20261112);rng=np.random.default_rng(20261112)
y=np.tile(np.arange(24),8);x=np.eye(24,dtype='f4')[y];text=normalize(rng.normal(size=(24,768)).astype('f4'))
linear=train_classifier(x,y,None,epochs=200);pl=classify(linear,x).argmax(-1);assert (pl==y).mean()>.8
align=train_classifier(x,y,text,epochs=40);pa=classify(align,x,text).argmax(-1);assert (pa==y).mean()>.8
base=torch.nn.Linear(12,7);lora=LoRALinear(base,rank=8);xx=torch.randn(3,12);assert torch.equal(base(xx),lora(xx))
with torch.no_grad():lora.B.normal_(0,.02)
before=lora(xx).detach();after=lora.merge()(xx).detach();err=float((before-after).abs().max());assert err<1e-5
state=np.zeros((len(y),14),dtype='f4');obs=np.zeros((len(y),24),dtype='f4');directions=np.stack([np.cos((y//3)*np.pi/4),np.sin((y//3)*np.pi/4),np.zeros(len(y))],-1).astype('f4')
model=fit_displacement(obs,state,text[y],directions,True,epochs=60);pred=displacement(model,obs,state,text[y]);cosine=float((normalize(pred[:,:2])*normalize(directions[:,:2])).sum(-1).mean());assert cosine>.8
result={'timestamp_brt':brt(),'synthetic_only':True,'linear_train_accuracy':float((pl==y).mean()),'contrastive_train_accuracy':float((pa==y).mean()),'language_displacement_cosine':cosine,'lora_merge_max_abs':err,'status':'passed'}
atomic(P/'synthetic_learning_checks.json',result);append_report('## 合成学习路径验证\n\n'+str(result)+'。仅确认线性/对比/语言位移头与LoRA合并实现能学习人为已知信号；不计入真实对齐成绩。')
print(result)
