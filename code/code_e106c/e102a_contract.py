"""Training-only gripper contract and causal proprioception; no model labels."""
import numpy as np,torch,json
from e99w2_common import CONTROL
GR=[6,13]
from functools import lru_cache
@lru_cache(None)
def contract():return json.loads((CONTROL/'GRIPPER_CONTRACT.json').read_text())
def states(q,c):return (q[...,GR]>np.asarray(c['threshold_m'])).astype('f4')
def fixed_np(q,c):
 x=q.copy();s=states(x,c);x[...,GR]=np.asarray(c['closed_m'])+s*(np.asarray(c['open_m'])-np.asarray(c['closed_m']));return x

def onset_labels(q,segments,good,c):
 s=states(q,c).astype('i1');lab=s.copy();events=[]
 for g in range(2):
  idx=np.flatnonzero((s[1:,g]!=s[:-1,g])&good[1:]&good[:-1]&(segments[1:]==segments[:-1]))+1
  prev=0
  for cross in idx:
   sign=1 if s[cross,g] else -1;noise=c['noise_m'][g];start=cross;quiet=0
   for j in range(cross,max(prev,cross-120),-1):
    if not good[j-1] or segments[j]!=segments[j-1]:break
    delta=sign*(q[j,GR[g]]-q[j-1,GR[g]])
    if delta < -noise:break
    if delta>noise:start=j;quiet=0
    else:
     quiet+=1
     if quiet>=3:break
   lab[start:cross,g]=s[cross,g];events.append(dict(gripper=g,onset=int(start),crossing=int(cross),lag=int(cross-start),direction=int(s[cross,g])))
   prev=cross
 return lab,events

def causal_features(q,pose,a,valid,segments,good,c):
 # Input state is observed threshold state, never retrospective onset labels.
 cur=fixed_np(q[a],c);p0=pose[a];dq=[];dp=[];flags=[]
 for off,v in zip((-24,-18,-12,-6),valid):
  j=a+off;ok=bool(v and j>=0 and good[j] and segments[j]==segments[a]);flags.append(float(ok));dq.append(fixed_np(q[j],c)-cur if ok else np.zeros(14));dp.append(pose[j]-p0 if ok else np.zeros(18))
 # Explicit binary gripper channels instead of fixed physical width in the 70 q channels.
 raw=np.concatenate([cur,*dq]).reshape(5,14)
 raw[0,GR]=states(q[a],c)
 for k,(off,ok) in enumerate(zip((-24,-18,-12,-6),flags)):raw[k+1,GR]=states(q[a+off],c)-states(q[a],c) if ok else 0
 return np.concatenate([raw.ravel(),p0,*dp,flags]).astype('f4')

def fixed_torch(q,c=None):
 c=contract() if c is None else c;x=q.clone();th=x.new_tensor(c['threshold_m']);lo=x.new_tensor(c['closed_m']);hi=x.new_tensor(c['open_m']);x[...,GR]=torch.where(x[...,GR]>th,hi,lo);return x
