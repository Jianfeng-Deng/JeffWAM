"""E101-S frozen S2 CPU online interface. Research candidate; not real-robot qualified.
Input: 21 current-frame E99-B3 method-b predicate probabilities in spec predicate order.
No ground-truth stage, future frames, cycle labels or mechanical-lock claim.
"""
import json,pickle
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp

class OnlineStageController:
 def __init__(self,result_dir):
  r=Path(result_dir);lock=json.loads((r/'CONTROLLER_LOCK.json').read_text());cfg=lock['selected']['S2'];assert cfg['method']=='b'
  with (r/'final_aux.pkl').open('rb') as f:self.aux=pickle.load(f)['b']
  self.graph=np.array(lock['graph'],bool);self.states=lock['states'];self.temp=cfg['config']['temp'];self.minimum=cfg['config']['m'];self.loga=np.full((10,10),-np.inf);m=self.aux['transition']>0;self.loga[m]=np.log(self.aux['transition'][m]);self.reset()
 def reset(self):
  self.forward=np.full(10,-np.inf);self.forward[0]=0;self.state=0;self.age=0
 def step(self,prob):
  p=np.asarray(prob,float);assert p.shape==(21,) and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
  x=expit(np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1))*self.aux['cal'][:,0]+self.aux['cal'][:,1]);x[13]=.5 # P073 index verified below on artifact predicate order.
  z=np.r_[x,1.]@self.aux['w'];e=(z-logsumexp(z)-np.log(self.aux['prior']))*self.temp
  self.forward=logsumexp(self.forward[:,None]+self.loga,axis=0)+e;self.forward-=logsumexp(self.forward);self.age+=1
  if self.age>=self.minimum:
   nxt=int(np.argmax(np.where(self.graph[self.state],self.forward,-np.inf)))
   if nxt!=self.state:self.state=nxt;self.age=0
  return dict(stage=self.states[self.state],posterior=np.exp(self.forward).tolist(),frames_since_switch=self.age)
