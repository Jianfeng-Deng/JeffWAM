from e98w_core import States,T
from e90.data import Data
import numpy as np,torch,json
OUT=T/'results/e98w_control_20261114_01'
class RobotData(Data):
 def get(self,i,horizons=(10,25,50,100),language='instruction',stage='P1'):
  b=super().get(i,horizons,language,stage);row=self.rows[i];a=row['anchor'];z,q,pose,good=self.robot_states.get(self,row['canonical_id']);assert q is not None
  past=[];flags=[]
  for off,valid in zip((-24,-18,-12,-6),row['history_valid']):
   j=a+off;ok=bool(valid and j>=0 and good[j] and z['continuous_segment'][j]==z['continuous_segment'][a] and z['state_available_time'][j]<=z['state_available_time'][a]+1e-7)
   assert not valid or ok
   past.append(q[j]-q[a] if ok else np.zeros(14));flags.append(float(ok))
  offsets=np.arange(101);idx=np.minimum(a+offsets,len(q)-1);valid=(a+offsets<len(q))&good[idx]&(z['continuous_segment'][idx]==z['continuous_segment'][a])&z['p1_source_mask'][idx]
  dt=np.diff(z['timestamp'][a:min(a+101,len(q))]);dtgood=np.ones(101,bool);dtgood[1:len(dt)+1]=(dt>0)&(dt<=.050001);valid=np.logical_and.accumulate(valid&dtgood)
  cur=np.broadcast_to(np.concatenate([q[a],*past]),(101,70));po=np.broadcast_to(pose[a],(101,18));fl=np.broadcast_to(flags,(101,4))
  feat=np.concatenate([cur,q[idx]-q[a],po,pose[idx]-pose[a],fl],1).astype('f4');feat[~valid]=0
  for h in horizons:
   if bool(b['world_valid'][b['world_horizons']==h].any()):assert valid[h],(row['canonical_id'],a,h)
  b['robot']=torch.tensor(feat);b['robot_valid']=torch.tensor(valid);b['robot_fold']=torch.tensor(self.robot_folds[row['parent_component']] if self.robot_crossfit else 5)
  return b

def augment(data,crossfit=False):
 data.__class__=RobotData;data.robot_states=States();data.robot_folds=json.loads((OUT/'crossfit_folds.json').read_text());data.robot_crossfit=crossfit;return data
