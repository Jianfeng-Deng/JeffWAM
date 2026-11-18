from pathlib import Path
root=Path(__file__).parent
p=root/'e94a'; (p/'__init__.py').write_text('')
for name in ('checkpoint','resource','common','train'):
 s=(root/'e90'/f'{name}.py').read_text()
 if name=='common':
  s=s.replace("T / 'code_e90'","T / 'code_e94a'").replace("ART = OUT / 'artifacts'","ART = T / 'results' / 'e94a_control_20261114_01'").replace("e90_full_v31_p1_20261114.md","e94a_readout_20261114.md").replace("('e86', 'e88', 'e90')","('e86', 'e88', 'e90', 'e94a')")
 if name=='resource':
  s=s.replace('21600.','7200.').replace('21500.','7190.').replace("'e90-gpu-walltime-v1'","'e94a-gpu-walltime-v1'")
 if name=='train':
  s=s.replace('from .data import load_data,sample_horizons','from e90.data import load_data,sample_horizons').replace('from .sampling import MultiSourceSampler','from e90.sampling import MultiSourceSampler')
  s=s.replace('def config_for(smoke=False):','def config_for(smoke=False,recipe="M1"):').replace("schema='e90-p1-config-v1'","schema='e94a-p1-config-v1',recipe=recipe,readout_hidden=1024,mean_weight=1.,fm_weight=1. if recipe=='M2' else 0.")
  s=s.replace("m=WorldModel().to('cuda:0')","m=WorldModel(recipe=cfg['recipe'],readout_hidden=cfg['readout_hidden']).to('cuda:0')")
  s=s.replace('def main(runid,smoke=False,resume=None,init_ckpt=None):','def main(runid,smoke=False,resume=None,init_ckpt=None,recipe="M1"):')
  s=s.replace('run=ART/runid','run=T/"results"/runid').replace('cfg=config_for(smoke);','cfg=config_for(smoke,recipe);')
  s=s.replace("(ART/'preflight_01/manifest.json')","(T/'code_e90/artifacts/preflight_01/manifest.json')")
  s=s.replace("'E90_smoke' if smoke else 'E90_P1_5000',1050 if smoke else 21000","'E94a_'+recipe,1000 if smoke else 3500")
  # Training-only driver: fixed E93 evaluator is a separate accounted segment.
  start=s.index('        if smoke:\n            from .evaluate import evaluate',s.index('        end=run/'))
  end=s.index("        atomic(run/'status.json'",start)
  s=s[:start]+s[end:]
  s=s.replace("ap.add_argument('--resume');", "ap.add_argument('--recipe',choices=['M1','M2'],default='M1');ap.add_argument('--resume');")
  s=s.replace('main(a.run_id,a.smoke,a.resume,a.init_ckpt)','main(a.run_id,a.smoke,a.resume,a.init_ckpt,a.recipe)')
 (p/f'{name}.py').write_text(s)
# The shared forward implementation is the E88 forward, with only query and readout additions.
s=(root/'e88/model.py').read_text(); begin=s.index('    def forward('); end=s.index('    def infer(',begin); forward=s[begin:end]
forward=forward.replace('def forward(self,b,a=None,w=None,ta=None,tw=None):','def forward(self,b,a=None,w=None,ta=None,tw=None,branch="fm"):')
forward=forward.replace('        # Legacy action-only API keeps physical W deletion.','        if branch not in ("mean","fm"): raise ValueError(branch)\n        if branch=="fm" and self.recipe=="M1": raise RuntimeError("M1 never calls FM")')
forward=forward.replace("            clock=self.tw(torch.stack((tw,torch.sin(tw*math.pi)),1))[:,None]\n            parts.append(self.world_proj(w)+he+pos[None]+clock+self.kind[2]);valids.append(b['world_valid'])", "            if branch=='mean':\n                query=self.mean_query.expand(len(o),nw,-1)+he+pos[None]+self.kind[2]\n            else:\n                clock=self.tw(torch.stack((tw,torch.sin(tw*math.pi)),1))[:,None]\n                query=self.world_proj(w)+he+pos[None]+clock+self.kind[2]\n            parts.append(query);valids.append(b['world_valid'])")
forward=forward.replace("        velocity=(self.world_head(wh)-w)/(1-tw[:,None,None]).clamp_min(.01) if wh is not None else None", "        mean=self.mean_head(wh) if wh is not None and branch=='mean' else None\n        velocity=(self.world_head(wh)-w)/(1-tw[:,None,None]).clamp_min(.01) if wh is not None and branch=='fm' else None")
forward=forward.replace('return dict(action=', 'return dict(mean=mean,action=')
header='''"""Frozen E94 unified readout v1. E90 inputs/attention; deterministic and FM MLP heads."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from e90.model import WorldModel as E90WorldModel, adapter_pass, SOURCES
from e86.model import Config, mask_oa
from e88.model import directional_mask, horizon_embedding

class WorldModel(E90WorldModel):
    def __init__(self,c=None,recipe='M1',readout_hidden=1024):
        super().__init__(c)
        if recipe not in ('M1','M2'): raise ValueError(recipe)
        self.recipe=recipe
        self.readout_hidden=readout_hidden
        self.mean_query=nn.Parameter(torch.randn(1,1,self.c.width)*.02)
        self.mean_head=nn.Sequential(nn.Linear(self.c.width,readout_hidden),nn.GELU(),nn.Linear(readout_hidden,self.c.latent))
        self.world_head=nn.Sequential(nn.Linear(self.c.width,readout_hidden),nn.GELU(),nn.Linear(readout_hidden,self.c.latent))
        for head in (self.mean_head,self.world_head):
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def forward_mean(self,b,a=None,ta=None):
        # Only target SHAPE is used; no target values, random noise, or FM clock.
        return self(b,a=a,w=torch.empty_like(b['world']),ta=ta,branch='mean')

    def mean_rollout(self,b,steps=4,seed=90):
        return b['world_current']+self.forward_mean(b)['mean']*b['world_scale']

'''
(p/'model.py').write_text(header+forward)
s=(root/'e90/objective.py').read_text(); chain=s[s.index('def chain_loss'):].replace('model.rollout','model.mean_rollout')
head='''"""P1 direct conditional mean MSE; M2 adds original FM with weight 1:1."""
import torch
import torch.nn.functional as F
from e86.common import generator
from e90.objective import weighted_mean,chain_weight

def loss(model,b,stage,step,seed=90,chain=None,chain_warmup=3000,chain_ramp=1000,chain_maximum=.1,p1_converged=False,engineering_chain_probe=False,phase_weight=0.,adapter_weight=.1,alignment_weight=0.,cosine_weight=.1,**kwargs):
    if stage!='P1' or phase_weight or alignment_weight:raise ValueError('E94 readout objective is P1 only, phase/alignment weights zero')
    y=b['world'].detach(); out=model.forward_mean(b)
    mean=weighted_mean((out['mean']-y).square().mean(-1),b['world_valid'],b['world_horizons'])
    cosine=weighted_mean(1-F.cosine_similarity(out['world_cos'],y,dim=-1),b['world_valid'],b['world_horizons'],True)
    adapter,_=model.adapter_loss(b['obs'][:,4]);zero=mean*0;fm=zero
    if model.recipe=='M2':
        device=y.device;n=len(y);g=lambda k:generator(seed,stage,step,k,device=device)
        tw=torch.sigmoid(torch.randn(n,device=device,generator=g('tw')))
        ew=torch.randn(y.shape,device=device,generator=g('ew'))
        w=(1-tw[:,None,None])*ew+tw[:,None,None]*y
        velocity=model(b,w=w,tw=tw)['world']
        fm=weighted_mean((velocity-(y-ew)).square().mean(-1),b['world_valid'],b['world_horizons'])
    cw=chain_weight(stage,step,bool(model.adapter_ready) and chain is not None and (p1_converged or engineering_chain_probe),chain_warmup,chain_ramp,chain_maximum)
    cl=zero;cc={}
    if cw:cl,cc=chain(model)
    total=mean+fm+cosine_weight*cosine+adapter_weight*adapter+cw*cl
    err=(out['mean']-y).square().mean(-1)
    per={str(int(h)):float(err[b['world_valid'] & (b['world_horizons']==h)].mean().detach()) for h in b['world_horizons'][0].unique() if (b['world_valid'] & (b['world_horizons']==h)).any()}
    return total,dict(per_horizon=per,mean=float(mean.detach()),fm=float(fm.detach()),world=float((mean+fm).detach()),cosine_short=float(cosine.detach()),adapter=float(adapter.detach()),phase=0.,action=0.,chain=float(cl.detach()),chain_weight=cw,mode='world_only',clean_aux=False,**cc)

'''
(p/'objective.py').write_text(head+chain)
