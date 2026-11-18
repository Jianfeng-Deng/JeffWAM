import copy,collections,json,numpy as np,torch
from pathlib import Path
from e99w2_common import *
from e90.data import Data,ImageDataset,instruction_cache
from e86.data import Package,verify_receipt,collate,GRIPS,JOINTS
from e90.common import INPUT,E88
from e98w_core import States,fk,CHAIN

def pose_np(q):
 shape=q.shape[:-1];q=q.reshape(-1,14);p=[]
 for sl in (slice(0,6),slice(7,13)):
  m=fk.fk_link6(q[:,sl],CHAIN);p.append(np.concatenate([m[:,:3,3]+m[:,:3,:3]@np.array([0,0,.1358]),m[:,:3,0],m[:,:3,1]],1))
 return np.concatenate(p,1).reshape(*shape,18).astype('f4')

def planned_np(context,action):
 # q0, 4 causal history deltas, qfuture-q0, pose0, posefuture-pose0, flags.
 q0=context[:14];p0=context[84:102];seq=np.concatenate([q0[None],action, np.repeat(action[-1:],50,0)],0)
 return np.concatenate([np.broadcast_to(context[:70],(101,70)),seq-q0,np.broadcast_to(p0,(101,18)),pose_np(seq)-p0,np.broadcast_to(context[120:124],(101,4))],1).astype('f4')

class PlannedData(Data):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);self.states=States()
  # ImageDataset uses permanent read-only mmap handles, including on P3.
  old=self.base;self.base=ImageDataset(self.rows,list(old.captures.values()),old.cache,old.stats,old.text_cache)
 def get(self,i,horizons=(10,25,50,100),language='instruction',stage='P3'):
  b=super().get(i,horizons,language,stage);r=self.rows[i];a=r['anchor'];z,q,pose,good=self.states.get(self,r['canonical_id']);assert q is not None and good[a]
  past=[];flags=[]
  for off,v in zip((-24,-18,-12,-6),r['history_valid']):
   j=a+off;ok=bool(v and j>=0 and good[j] and z['continuous_segment'][j]==z['continuous_segment'][a] and z['state_available_time'][j]<=z['state_available_time'][a]+1e-7);assert not v or ok
   past.append(q[j]-q[a] if ok else np.zeros(14));flags.append(float(ok))
  context=np.concatenate([q[a],*past,np.zeros(14),pose[a],np.zeros(18),flags]).astype('f4')
  action=getattr(self,'shift_plan',None)
  if action is None:
   action=z['action'][a:a+50];assert np.array_equal(z['original_action_index'][a:a+50],np.arange(a+1,a+51))
  assert action.shape==(50,14)
  assert np.isfinite(action).all();anchor=z['q'][a].copy();anchor[GRIPS]=0
  b.update(q_normalized=torch.tensor((z['q'][a]-self.base.stats['q_mean'])/np.asarray(self.base.stats['q_std']),dtype=torch.float32),q_anchor=torch.tensor(anchor),q_current=torch.tensor(q[a],dtype=torch.float32),action=torch.tensor(action),action_mean=torch.tensor(self.base.stats['action_mean']),action_std=torch.tensor(self.base.stats['action_std']),state_valid=torch.tensor(True),context=torch.tensor(context),robot=torch.tensor(planned_np(context,action)),robot_valid=torch.ones(101,dtype=torch.bool))
  align=self.base.text_cache.get(('align',r['canonical_id'],a));b['align_valid']=torch.tensor(align is not None and r['split']=='train');b['align_text']=torch.tensor(align if align is not None else np.zeros(768),dtype=torch.float32)
  return b
 def oracle(self,i):
  r=self.rows[i];a=r['anchor'];z,q,pose,good=self.states.get(self,r['canonical_id']);base=self.get(i,(10,),'short')['context'].numpy();idx=np.minimum(a+np.arange(101),len(q)-1);v=(a+np.arange(101)<len(q))&good[idx]&(z['continuous_segment'][idx]==z['continuous_segment'][a])
  v=np.logical_and.accumulate(v);seq=q[idx];feat=np.concatenate([np.broadcast_to(base[:70],(101,70)),seq-q[a],np.broadcast_to(base[84:102],(101,18)),pose[idx]-pose[a],np.broadcast_to(base[120:],(101,4))],1).astype('f4');feat[~v]=0
  return torch.tensor(feat),torch.tensor(v)
 def shifted(self,i,k,language='instruction',stage='P3'):
  # Original exact image endpoint/history contract, but shifted plan is ONLY original plan suffix + terminal hold.
  orig=self.get(i,(k,),language);act=orig['action'].numpy();suffix=np.concatenate([act[k:],np.repeat(act[-1:],k,0)],0)
  temp=copy.copy(self);temp.shift_plan=suffix
  b=Data.shifted(temp,i,k,language,'P1')
  if b is None:return None
  ctx=b['context'].numpy();b['action']=torch.tensor(suffix);b['robot']=torch.tensor(planned_np(ctx,suffix));b['robot_valid']=torch.ones(101,dtype=torch.bool)
  return b

def load():
 p=Package();ready=verify_receipt(INPUT/'READY_V3_cache_windows.json');assert p.refs==ready['refs'];rs=json.loads(Path(ready['windows']['P3']).read_text());assert len(rs)==34777 and len({r['parent_component'] for r in rs})==145
 rs=[dict(r,_stage='P3',source='piper') for r in rs];assert all(p.split[r['parent_component']]=='train' and r['canonical_id'] not in p.excluded for r in rs)
 stats=json.loads((INPUT/'stats.json').read_text());norm=json.loads((E88/'artifacts/cpu_audit_01/residual_stats.json').read_text());assert norm['fitted_split']=='train'
 from e86.language import load as textload
 text=textload(rs,C/'artifacts/language_v1');tr=PlannedData(rs,p.captures,ready['cache'],stats,norm,text,stage='P3')
 caps=json.loads((C/'artifacts/dev_v1/captures.json').read_text());rows=json.loads((C/'artifacts/dev_v1/windows_v2.json').read_text());rows=[dict(r,_stage='P3',source='piper') for r in rows];assert len(rows)==48 and len({r['parent_component'] for r in rows})==6
 assert all(p.split[r['parent_component']]=='dev' and r['canonical_id'] not in p.excluded for r in rows)
 dev=PlannedData(rows,caps,C/'artifacts/dev_features_v1',stats,norm,{'__human_default__':text['__human_default__']},stage='P3')
 return p,ready,tr,dev

def audit(tr):
 sums={off:dict(abs=np.zeros(14),sq=np.zeros(14),max=np.zeros(14),n=0) for off in (-1,0,1)};time=[];parents=collections.defaultdict(list);exact=0
 for r in tr.rows:
  z,q,_,g=tr.states.get(tr,r['canonical_id']);a=r['anchor'];act=z['action'][a:a+50].astype('f8');assert np.array_equal(z['original_action_index'][a:a+50],np.arange(a+1,a+51))
  for off,s in sums.items():
   target=q[a+1+off:a+51+off]
   if len(target)!=50:continue
   e=act-target
   s['abs']+=np.abs(e).sum(0);s['sq']+=(e*e).sum(0);s['max']=np.maximum(s['max'],abs(e).max(0));s['n']+=50
  e=act-q[a+1:a+51];exact+=int(np.array_equal(act,q[a+1:a+51]));parents[r['parent_component']].append(np.abs(e).mean(0));time.extend(np.diff(z['timestamp'][a:a+51]).tolist())
 out=dict(windows=len(tr.rows),parents=len(parents),exact_raw_q_blocks=exact,truth='next-frame follower feedback, not sent commands',order='L6,Lgrip,R6,Rgrip; rad and signed m',by_q_offset={str(1+off):dict(mae=(s['abs']/s['n']).tolist(),rmse=np.sqrt(s['sq']/s['n']).tolist(),max=s['max'].tolist(),pairs=s['n']) for off,s in sums.items()},parent_equal_mae=np.mean([np.mean(v,0) for v in parents.values()],0).tolist(),frame_dt_s_quantiles=np.quantile(time,[0,.5,.95,1]).tolist(),extension='row0=current measured q; rows1..50=action0..49; rows51..100=action49 hold; no interpolation or measured future extension')
 atomic(ART/'action_contract.json',out);report('## 动作合同全池核查\n'+json.dumps(out,ensure_ascii=False));return out
