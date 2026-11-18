import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import sys
sys.dont_write_bytecode=True
import torch
from common import *
from encoder import Encoder
from lora_module import inject,merge
torch.set_num_threads(2);torch.manual_seed(20261112)
encoder=Encoder('cpu');model=encoder.model
pixels=torch.linspace(-1,1,3*224*224).reshape(1,3,224,224)
tokens=encoder.tok(['the right arm moves left.','the left arm moves right.'],padding='max_length',max_length=64,truncation=True,return_tensors='pt')
config=inject(model);parameters=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=1e-5)
v=model.vision_model(pixel_values=pixels).pooler_output;t=model.get_text_features(**tokens)
if not isinstance(t,torch.Tensor):t=t.pooler_output
loss=(v.float().square().mean()+t.float().square().mean());loss.backward()
grads={tower:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(tower) and p.grad is not None) for tower in ['vision_model','text_model']}
assert all(x>0 for x in grads.values())
assert not any(p.grad is not None for n,p in model.named_parameters() if not (n.endswith('.A') or n.endswith('.B')))
opt.step()
with torch.no_grad():before=model.vision_model(pixel_values=pixels).pooler_output
merge(model)
with torch.no_grad():after=model.vision_model(pixel_values=pixels).pooler_output
err=float((before-after).abs().max());assert err<1e-4,err
obj={'timestamp_brt':brt(),'status':'passed','synthetic_only':True,'config':config,'both_tower_gradient_sums':grads,'full_forward_merge_max_abs':err,'frozen_after_merge':not any(p.requires_grad for p in model.parameters())}
atomic(P/'synthetic_lora_check.json',obj);append_report('## SigLIP2双塔LoRA梯度/合并合成验证\n\n'+str(obj)+'。未用真实图像或标签。')
print(obj)
