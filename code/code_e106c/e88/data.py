import numpy as np,torch,json,collections
from e86.data import CachedDataset,collate
from e86.common import generator
from .audit import HorizonIndex,HORIZONS,stage_target,causal_segments

def sample_horizons(step,seed=88):
    # Fixed probes always present, plus a uniformly drawn integer frame offset.
    g=generator(seed,step,'continuous_horizon');j=int(torch.randint(0,5,(),generator=g));lo,hi=HORIZONS[j:j+2]
    random_h=int(torch.randint(lo,hi+1,(),generator=g));return tuple(dict.fromkeys((*HORIZONS,random_h)))
def coarse(z,h):return z if h<=50 else z.mean(-2,keepdims=True)
class Data:
    def __init__(self,rows,captures,cache,stats,residual_stats,labels=None,text_cache=None,narration_vectors=None):
        self.base=CachedDataset(rows,captures,cache,stats,text_cache);self.rows=rows;self.index=HorizonIndex(cache,captures);self.norm=residual_stats['scale'];self.labels=labels or {};self.narration_vectors=narration_vectors or {}
    def scale(self,h):
        if str(h) in self.norm:return np.asarray(self.norm[str(h)],np.float32)
        # Interpolate only within a common resolution regime.
        keys=[x for x in HORIZONS if (x<=50)==(h<=50)]
        if h<min(keys):return np.asarray(self.norm[str(min(keys))],np.float32)
        a=max(x for x in keys if x<=h);b=min((x for x in keys if x>=h),default=max(keys));t=(h-a)/max(1,b-a)
        return np.asarray(self.norm[str(a)],np.float32)*(1-t)+np.asarray(self.norm[str(b)],np.float32)*t
    def endpoint(self,row,h):
        ids,why=self.index.resolve(row,h)
        if ids is None:return np.zeros((3,8,768),np.float32),False
        return np.stack([self.base.features(row['canonical_id'],k,[ids[k]])[0] for k in range(3)]),True
    def get(self,i,horizons=HORIZONS,language='short',stage='P3'):
        r=dict(self.rows[i]);stage=r.get('_stage',stage);r['_stage']=stage;b=self.base[i];current=b['obs'][4].numpy();future=[];currents=[];scales=[];valid=[];hs=[];positions=[];segments=self.labels.get(r['canonical_id'],[]);a=r['anchor']
        if language=='none':b['text_valid']=torch.zeros_like(b['text_valid']);b['align_valid']=torch.tensor(False)
        elif language=='long':
            vectors=[self.narration_vectors[(r['canonical_id'],j)] for j,x in causal_segments(segments,a) if (r['canonical_id'],j) in self.narration_vectors]
            if vectors:b['text']=torch.cat((b['text'],torch.tensor(np.concatenate(vectors))),0);b['text_valid']=torch.ones(len(b['text']),dtype=torch.bool)
        elif language=='short':
            b['text']=torch.tensor(self.base.text_cache.get('__human_default__',np.empty((0,768),np.float32)));b['text_valid']=torch.ones(len(b['text']),dtype=torch.bool)
        else:raise ValueError(language)
        for h in horizons:
            z,ok=self.endpoint(r,h);z=coarse(z,h);cur=coarse(current,h);n=z.shape[1];future.append(z.reshape(-1,768));currents.append(cur.reshape(-1,768));scales.append(np.broadcast_to(self.scale(h)[:,:,None],z.shape).reshape(-1,768));valid.extend([ok]*(3*n));hs.extend([h]*(3*n));positions.extend([k*8+j for k in range(3) for j in (range(8) if n==8 else [0])])
        future=np.concatenate(future);cur=np.concatenate(currents);scale=np.concatenate(scales)
        b.update(world=torch.tensor((future-cur)/scale),world_raw=torch.tensor(future),world_current=torch.tensor(cur),world_scale=torch.tensor(scale),world_valid=torch.tensor(valid),world_horizons=torch.tensor(hs),world_positions=torch.tensor(positions))
        phase,progress,ok=stage_target(segments,a);b.update(phase_target=torch.tensor(phase),progress_target=torch.tensor(progress),phase_valid=torch.tensor(ok and r['split']=='train'))
        return b
    def batch(self,ids,horizons=HORIZONS,language='short',stage='P3'):
        b=collate([self.get(i,horizons,language,stage) for i in ids]);b['world_positions']=b['world_positions'][0];return b
    def shifted(self,i,k,language='short',stage='P3'):
        from e86.data import build_window
        r=dict(self.rows[i],_stage=stage);z,_=self.index.get(r['canonical_id']);a=r['anchor']+k;ids,why=self.index.resolve(r,k)
        if ids is None:return None
        # Rebuild causally available camera history at the shifted time; no action/state is used.
        entry={'subinterval_raw_rows_exclusive':r.get('subinterval_raw_rows_exclusive')};c=self.base.captures[r['canonical_id']];ct=z['camera_exposure_timestamp'];rc=z['camera_received_timestamp'];avail=z['state_available_time'];oi=[]
        for off in (-6,0):oi.append([int(np.searchsorted(np.maximum.accumulate(np.maximum(ct[:,cam],rc[:,cam])),avail[a+off],side='right')-1) for cam in range(3)])
        # build_window requires +50; chain k=25,50,100 already implies this except k25, handled by original masks.
        if a+50>=len(z['timestamp']):return None
        try:row=build_window(c,z,a,np.array(oi),np.array([ids]*3),stage,entry if entry['subinterval_raw_rows_exclusive'] else None)
        except ValueError:return None
        row.update({key:r[key] for key in ('stratum','subinterval_raw_rows_exclusive') if key in r});row['_stage']=stage
        # Only actually cached causal history frames are kept; missing history masked as a complete time slot.
        for j in range(4):
            if row['history_valid'][j]:
                try:
                    for cam in range(3):self.base.features(r['canonical_id'],cam,[row['obs_rows'][j][cam]])
                except KeyError:row['history_valid'][j]=False;row['obs_rows'][j]=[-1]*3
        # Use a temporary one-row dataset; never mutate original authoritative windows.
        temp=Data([row],list(self.base.captures.values()),self.base.cache,self.base.stats,{'scale':self.norm},self.labels,self.base.text_cache,self.narration_vectors)
        try:b=temp.get(0,(k,),language,stage)
        except KeyError:return None
        return b
