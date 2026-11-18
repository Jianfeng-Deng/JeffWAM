import torch,math
from torch import nn
import torch.nn.functional as F
class LoRALinear(nn.Module):
    def __init__(self,base,rank=8):
        super().__init__();self.base=base;self.rank=rank
        self.A=nn.Parameter(torch.empty(rank,base.in_features));self.B=nn.Parameter(torch.zeros(base.out_features,rank));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        self.base.requires_grad_(False)
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.A),self.B)
    def merge(self):
        with torch.no_grad():self.base.weight.add_(self.B@self.A)
        return self.base
def inject(model,rank=8,layers=2):
    model.requires_grad_(False);selected=[]
    for name,tower in [('vision',model.vision_model),('text',model.text_model)]:
        for i in range(len(tower.encoder.layers)-layers,len(tower.encoder.layers)):
            attn=tower.encoder.layers[i].self_attn
            for proj in ['q_proj','v_proj']:
                setattr(attn,proj,LoRALinear(getattr(attn,proj),rank));selected.append(f'{name}.{i}.{proj}')
    return {'modules':selected,'rank':rank,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
def adapter_state(model):return {k:v.detach().cpu() for k,v in model.state_dict().items() if k.endswith('.A') or k.endswith('.B')}
def merge(model):
    for tower in [model.vision_model,model.text_model]:
        for layer in tower.encoder.layers:
            for proj in ['q_proj','v_proj']:
                module=getattr(layer.self_attn,proj)
                if isinstance(module,LoRALinear):setattr(layer.self_attn,proj,module.merge())
    model.requires_grad_(False);return model
