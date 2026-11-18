import torch,math
from torch import nn
from e103d_model import WorldModel
from e90.objective import weighted_mean
from e86.common import generator
from e86.objective import mode_for
class StochasticWorld(WorldModel):
 def __init__(self,norm):
  super().__init__(norm)
  self.requires_grad_(False)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(90)
   self.noise_branch=nn.Sequential(nn.Linear(self.c.width+32,256),nn.SiLU(),nn.Linear(256,768))
   nn.init.normal_(self.noise_branch[-1].weight,std=.002);nn.init.zeros_(self.noise_branch[-1].bias)
  self.mean_head.register_forward_hook(self.capture)
  self.reading='zero';self.eval_seed=90
 def capture(self,module,args,result):self.hidden=args[0].detach();return result
 def samples(self,mean,k=4,seed=90):
  x=self.hidden;g=torch.Generator(device=x.device).manual_seed(seed)
  z=torch.randn((len(x),k,32),device=x.device,dtype=x.dtype,generator=g)
  xx=x[:,None].expand(-1,k,-1,-1);zz=z[:,:,None].expand(-1,-1,x.shape[1],-1)
  center=self.noise_branch(torch.cat([x,torch.zeros_like(z[:,0,None]).expand(-1,x.shape[1],-1)],-1))
  residual=self.noise_branch(torch.cat([xx,zz],-1))-center[:,None]
  return mean[:,None]+residual
 def forward_mean(self,b):
  out=super().forward_mean(b)
  if self.reading=='mean':out['mean']=self.samples(out['mean'],seed=self.eval_seed).mean(1)
  return out
 def with_z(self,mean,z):
  if not torch.count_nonzero(z):return mean
  x=self.hidden;zz=z[:,None].expand(-1,x.shape[1],-1)
  return mean+self.noise_branch(torch.cat([x,zz],-1))-self.noise_branch(torch.cat([x,torch.zeros_like(zz)],-1))
def scores(s,y):
 # Euclidean RMS over feature dimensions, jointly over tokens at each horizon below.
 a=(s-y[:,None]).square().mean(-1).clamp_min(1e-12).sqrt().mean(1)
 pair=(s[:,:,None]-s[:,None,:]).square().mean(-1).clamp_min(1e-12).sqrt().mean((1,2))
 return a-.5*pair

def energy(m,b,mean,seed):
 s=m.samples(mean,4,seed);terms=[]
 # Joint flattened feature-token vector at each horizon, finite ensemble V-statistic.
 for h in b['world_horizons'][0].unique():
  ix=b['world_horizons'][0]==h;ss=s[:,:,ix].flatten(2);yy=b['world'][:,ix].flatten(1)
  valid=b['world_valid'][:,ix].all(1);es=scores(ss,yy)
  mse=(ss.mean(1)-yy).square().mean(1)
  terms.append(((es+.001*mse)*valid).sum()/valid.sum().clamp_min(1))
 return torch.stack(terms).mean()
def p3_energy(m,b,step):
 n=len(b['obs']);device=b['obs'].device;g=lambda k:generator(90,'P3',step,k,device=device)
 ta=torch.sigmoid(torch.randn(n,device=device,generator=g('ta')));mode=mode_for(step,90,(.25,)*4);tw=torch.sigmoid(torch.randn(n,device=device,generator=g('tw')))
 if mode=='sync':tw=ta
 if mode=='clean':ta=torch.ones_like(ta)
 ya=(b['action']-b['q_anchor'][:,None]-b['action_mean'])/b['action_std'];ea=torch.randn(ya.shape,device=device,generator=g('ea'));a=(1-ta[:,None,None])*ea+ta[:,None,None]*ya
 with torch.no_grad():out=m(b,a,None if mode=='action_only' else torch.empty_like(b['world']),ta,tw)
 v=sum(p.sum()*0 for p in m.noise_branch.parameters()) if mode=='action_only' else .1*energy(m,b,out['mean'],step+900000)
 return v,dict(mode=mode,world_energy=float(v.detach()))
