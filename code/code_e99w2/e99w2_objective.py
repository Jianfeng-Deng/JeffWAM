import torch,torch.nn.functional as F
from e86.common import generator
from e86.objective import mode_for,contrastive
from e90.objective import weighted_mean,chain_weight
from e94a.objective import chain_loss

def loss(model,b,step,chain=None):
 device=b['obs'].device;n=len(b['obs']);g=lambda k:generator(90,'P3',step,k,device=device)
 ta=torch.sigmoid(torch.randn(n,device=device,generator=g('ta')));mode=mode_for(step,90,(.25,)*4);tw=torch.sigmoid(torch.randn(n,device=device,generator=g('tw')))
 if mode=='sync':tw=ta
 if mode=='clean':ta=torch.ones_like(ta)
 ya=(b['action']-b['q_anchor'][:,None]-b['action_mean'])/b['action_std'];ea=torch.randn(ya.shape,device=device,generator=g('ea'));a=(1-ta[:,None,None])*ea+ta[:,None,None]*ya
 w=None if mode=='action_only' else torch.empty_like(b['world']);out=model(b,a,w,ta,tw);zero=out['observation'].sum()*0;world=zero;cos=zero;ad=zero;cl=zero;cc={};cw=0.
 if w is not None:
  world=weighted_mean((out['mean']-b['world']).square().mean(-1),b['world_valid'],b['world_horizons']);cos=weighted_mean(1-F.cosine_similarity(out['world_cos'],b['world'],dim=-1),b['world_valid'],b['world_horizons'],True);ad,_=model.adapter_loss(b['obs'][:,4])
  # Original P1 warmup and gate retained explicitly in this 5000-update new stage.
  cw=chain_weight('P1',step,bool(model.adapter_ready) and chain is not None,3000,1000,.1)
  if cw:cl,cc=chain(model)
 if mode=='clean':ap=model(b,.5*ea+.5*ya,None,ta*.0+.5,tw)['action']
 else:ap=out['action']
 wt=torch.tensor([4.]*10+[2.]*15+[1.]*25,device=device)[None,:,None];action=((ap-(ya-ea)).square()*wt).sum()/(wt.sum()*14*n)
 align=contrastive(model,out,b);value=action+.1*(world+.1*cos+.1*ad+cw*cl)+.01*align
 return value,dict(mode=mode,action=float(action.detach()),world=float(world.detach()),cosine=float(cos.detach()),adapter=float(ad.detach()),alignment=float(align.detach()),chain=float(cl.detach()),chain_weight=cw,**cc)
