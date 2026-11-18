import torch,torch.nn.functional as F
from e86.common import generator
from e86.objective import mode_for,contrastive
from e90.objective import weighted_mean,chain_weight
from e94a.objective import chain_loss
from e99w2_common import SEED,GROUP
from e99w2_model import JOINTS

def loss(model,b,step,chain=None):
 device=b['obs'].device;n=len(b['obs']);g=lambda k:generator(SEED,'P3',step,k,device=device)
 ta=torch.sigmoid(torch.randn(n,device=device,generator=g('ta')));mode=mode_for(step,SEED,(.25,)*4);tw=torch.sigmoid(torch.randn(n,device=device,generator=g('tw')))
 if mode=='sync':tw=ta
 if mode=='clean':ta=torch.ones_like(ta)
 ya=(b['action']-b['q_anchor'][:,None]-b['action_mean'])/b['action_std'];ea=torch.randn(ya.shape,device=device,generator=g('ea'));a=(1-ta[:,None,None])*ea+ta[:,None,None]*ya
 w=None if mode=='action_only' else torch.empty_like(b['world']);out=model(b,a,w,ta,tw);zero=out['observation'].sum()*0;world=zero;cos=zero;ad=zero;cl=zero;cc={};cw=0.
 if w is not None:
  world=weighted_mean((out['mean']-b['world']).square().mean(-1),b['world_valid'],b['world_horizons']);cos=weighted_mean(1-F.cosine_similarity(out['world_cos'],b['world'],dim=-1),b['world_valid'],b['world_horizons'],True);ad,_=model.adapter_loss(b['obs'][:,4])
  # Continue C2 chain schedule at global step5000+local; same for all three groups.
  cw=chain_weight('P1',5000+step,bool(model.adapter_ready) and chain is not None,3000,1000,.1)
  if cw:cl,cc=chain(model)
 if mode=='clean':ap=model(b,.5*ea+.5*ya,None,ta*.0+.5,tw)['action']
 else:ap=out['action']
 wt=torch.tensor([4.]*10+[2.]*15+[1.]*25,device=device)[None,:,None];err=(ap-(ya-ea)).square()
 if GROUP in ('A1','A2'):err=err[:,:,JOINTS]
 action=(err*wt).sum()/(wt.sum()*err.shape[-1]*n)
 grip=zero;delta=zero
 if GROUP in ('A1','A2'):
  heads=model(b,torch.zeros_like(a),None,torch.zeros_like(ta),torch.zeros_like(tw))
  grip=F.binary_cross_entropy_with_logits(heads['grip_logits'],b['grip_label'])
  if GROUP=='A2':
   target=(b['action'][:,:,JOINTS]-b['q_current'][:,None,JOINTS])/b['action_std'][:,:,JOINTS]
   delta=((heads['delta']-target).square()*wt).sum()/(wt.sum()*12*n)
 action=action+grip+delta
 align=contrastive(model,out,b);value=action+.1*(world+.1*cos+.1*ad+cw*cl)+.01*align
 return value,dict(grip_bce=float(grip.detach()),delta_mse=float(delta.detach()),mode=mode,action=float(action.detach()),world=float(world.detach()),cosine=float(cos.detach()),adapter=float(ad.detach()),alignment=float(align.detach()),chain=float(cl.detach()),chain_weight=cw,**cc)
