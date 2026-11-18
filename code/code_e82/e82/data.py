from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch.utils.data import Dataset
from .randomness import generator,derive

JOINTS=list(range(6))+list(range(7,13)); GRIPPERS=[6,13]
MANIFEST_FIELDS=('canonical_id','owner','source_path','source_collection','raw_capture_id','parent_component','session_id','split','task_id','robot_schema','camera','timestamps','command_semantics_evidence','duplicate_group','continuous_segment','original_action_index','outcome_source','reviewer','allow_p1','allow_p2','allow_p3','exclude_reason','transform_id','code_hash','data_hash')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_ready(root,require_data=True):
    root=Path(root); names=['READY_parent_split.json']+(['READY_data.json'] if require_data else [])
    result={}
    for name in names:
        p=root/name
        if not p.is_file(): raise FileNotFoundError(f'Data gate missing: {p}')
        d=json.loads(p.read_text()); result[name]=d
        status=str(d.get('status','')).lower()
        valid_status=status=='parent_split_locked' if name=='READY_parent_split.json' else status.startswith('ready')
        if not d.get('files') or not valid_status:
            raise ValueError(f'READY not valid/published: {name}: {d.get("status")}')
        entries=d.get('files',[])
        if not isinstance(entries,list): entries=[dict(path=k,sha256=v) for k,v in entries.items()]
        for e in entries:
            path=Path(e['path']); path=path if path.is_absolute() else root/path
            if not path.is_file() or sha256(path)!=e['sha256']: raise ValueError(f'READY hash mismatch: {path}')
    if require_data:
        parent=result['READY_data.json'].get('parent_split')
        if parent and sha256(parent['path'])!=parent['sha256']:raise ValueError('Data READY refers to different parent/split revision')
    return result

def causal_dq(q,previous_q,time,previous_time,valid=True):
    dt=float(time)-float(previous_time)
    if not valid or not np.isfinite(dt) or dt<=0: return np.zeros(14,np.float32),False
    v=(np.asarray(q)-np.asarray(previous_q))/dt
    if not np.isfinite(v).all(): return np.zeros(14,np.float32),False
    return v.astype(np.float32),True

class StatContract:
    """Train-only, frozen throughout P1 -> P3; E* group statistics stay 768D."""
    def __init__(self,d):
        self.d=d
        if d.get('fitted_split')!='train': raise ValueError('Statistics must be training-only')
        if d.get('world_fitted_stage')!='P1': raise ValueError('World stats must use allowed P1 train parents')
        self.a_mean=torch.tensor(d['action_mean']); self.a_std=torch.tensor(d['action_std'])
        self.q_mean=torch.tensor(d['q_mean']); self.q_std=torch.tensor(d['q_std'])
        self.dq_mean=torch.tensor(d['dq_mean']); self.dq_std=torch.tensor(d['dq_std'])
        self.h_mean=torch.tensor(d['history_mean']); self.h_std=torch.tensor(d['history_std'])
        self.w_mean=torch.tensor(d['world_mean']); self.w_std=torch.tensor(d['world_std'])
        for x in (self.a_std,self.q_std,self.dq_std,self.h_std,self.w_std):
            if not torch.isfinite(x).all() or not (x>0).all(): raise ValueError('Invalid stat scale')
        assert self.a_mean.shape==self.a_std.shape==(50,14)
        assert self.w_mean.shape==self.w_std.shape==(91,768)
    @classmethod
    def fixture(cls):
        return cls(dict(fitted_split='train',world_fitted_stage='P1',fixture=True,action_mean=np.zeros((50,14)).tolist(),action_std=np.ones((50,14)).tolist(),q_mean=[0.]*14,q_std=[1.]*14,dq_mean=[0.]*14,dq_std=[1.]*14,history_mean=[0.]*14,history_std=[1.]*14,world_mean=np.zeros((91,768)).tolist(),world_std=np.ones((91,768)).tolist()))
    @classmethod
    def load(cls,path): return cls(json.loads(Path(path).read_text()))
    def norm_action(self,a,q):
        a=a.clone(); a[...,JOINTS]-=q[...,None,JOINTS]
        return (a-self.a_mean.to(a))/self.a_std.to(a)
    def denorm_action(self,a,q):
        a=a*self.a_std.to(a)+self.a_mean.to(a); a[...,JOINTS]+=q[...,None,JOINTS]; return a
    def norm_world(self,w): return (w-self.w_mean.to(w))/self.w_std.to(w)
    def norm_state(self,b):
        b=dict(b); b['q_raw']=b['q'].clone()
        b['q']=(b['q']-self.q_mean.to(b['q']))/self.q_std.to(b['q'])
        b['dq']=(b['dq']-self.dq_mean.to(b['dq']))/self.dq_std.to(b['dq'])
        b['history']=b['history'].clone()
        b['history'][...,:14]=(b['history'][...,:14]-self.h_mean.to(b['history']))/self.h_std.to(b['history'])
        b['q']*=b['q_valid']; b['dq']*=b['dq_valid'][:,None]
        b['history'][...,:14]*=b['history'][...,14,None]
        return b

def synthetic_batch(n=1,seed=1,production_shapes=False):
    g=generator(seed,'synthetic')
    shapes=[(252,448),(224,224),(224,224),(224,224),(224,224)] if production_shapes else [(28,42),(28,28),(28,28),(28,28),(28,28)]
    def images(t): return [torch.rand(n,t,3,h,w,generator=g) for h,w in shapes]
    def masks(t): return [torch.ones(n,t,1,h,w) for h,w in shapes]
    q=torch.randn(n,14,generator=g)*.1; a=q[:,None].expand(n,50,14).clone()+torch.randn(n,50,14,generator=g)*.01
    a[:,:,GRIPPERS]=torch.randn(n,50,2,generator=g)*.003
    hist=torch.cat((q[:,None].expand(n,8,14),torch.ones(n,8,1),torch.arange(7,-1,-1)[None,:,None].expand(n,8,1)/30),-1)
    return dict(obs_images=images(2),obs_pixel_valid=masks(2),obs_crop_valid=torch.ones(n,2,5,dtype=torch.bool),obs_age=torch.zeros(n,2,5),obs_offsets=torch.tensor([-.2,0.])[None].expand(n,2),obs_boxes=torch.tensor([0.,0.,1.,1.])[None,None,None].expand(n,2,5,4).clone(),future_images=images(3),future_pixel_valid=masks(3),future_crop_valid=torch.ones(n,3,5,dtype=torch.bool),q=q,dq=torch.zeros(n,14),q_valid=torch.ones(n,14,dtype=torch.bool),dq_valid=torch.ones(n,dtype=torch.bool),geo=torch.zeros(n,1),state_valid=torch.tensor([1,1,0,1],dtype=torch.bool)[None].expand(n,4).clone(),history=hist,task_id=torch.ones(n,dtype=torch.long),action=a,parent_component=[f'p{i}' for i in range(n)],session_id=[f's{i//2}' for i in range(n)],window_id=[str(i) for i in range(n)])

def move(batch,device):
    return {k:([x.to(device) for x in v] if isinstance(v,list) and v and torch.is_tensor(v[0]) else v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}

def select(batch,indices):
    ids=indices.tolist()
    return {k:([x[indices] for x in v] if isinstance(v,list) and v and torch.is_tensor(v[0]) else [v[i] for i in ids] if isinstance(v,list) else v[indices] if torch.is_tensor(v) else v) for k,v in batch.items()}

def collate(items):
    # Dataset items carry an explicit batch dimension of one.
    return {k:([torch.cat([r[k][i] for r in items]) for i in range(len(v))] if isinstance(v,list) and v and torch.is_tensor(v[0]) else sum([r[k] for r in items],[]) if isinstance(v,list) else torch.cat([r[k] for r in items]) if torch.is_tensor(v) else v) for k,v in items[0].items()}

class SyntheticDataset(Dataset):
    fixture=True
    def __init__(self,n=12,seed=1,production_shapes=False):
        self.n=n; self.seed=seed; self.production_shapes=production_shapes
        self.rows=[dict(parent_component=f'p{i//3}',session_id=f's{i//6}',task_id=1,outcome_pool='demonstration' if i%2 else 'other') for i in range(n)]
    def __len__(self): return self.n
    def __getitem__(self,i):
        b=synthetic_batch(1,derive(self.seed,i),self.production_shapes)
        for k in ('parent_component','session_id'): b[k]=[self.rows[i][k]]
        b['window_id']=[str(i)]; return b

class ParentSampler:
    def __init__(self,rows,seed=8101,stage='P3_joint'):
        self.seed=seed; self.stage=stage; self.groups=defaultdict(lambda:defaultdict(list))
        self.pools=defaultdict(list)
        for i,r in enumerate(rows):
            self.groups[int(r.get('task_id',0))][r['parent_component']].append(i)
        self.outcome_samplers=[]
        if stage=='P1' and rows and all(r.get('outcome_reviewed',False) for r in rows):
            pools=defaultdict(list)
            for r in rows:pools['demonstration' if r.get('outcome_pool')=='demonstration' else 'other'].append(r)
            if len(pools)==2:
                for pool,subset in sorted(pools.items()):
                    indices=[i for i,r in enumerate(rows) if r in subset]
                    self.outcome_samplers.append((ParentSampler(subset,seed,'P1_outcome_'+pool),indices))
        self.strategy='reviewed_50_50_outcomes' if self.outcome_samplers else 'task_parent_equal_no_reviewed_outcome_pools'
        self.tasks=sorted(self.groups)
        self.parents={t:sorted(self.groups[t]) for t in self.tasks}
        if not self.tasks: raise ValueError('No eligible windows')
    def sample(self,step,n):
        g=generator(self.seed,self.stage,step,'data'); result=[]
        if self.outcome_samplers:
            for slot in range(n):
                chosen=int(torch.randint(2,(1,),generator=g)); sampler,indices=self.outcome_samplers[chosen]
                result.append(indices[sampler.sample(step*n+slot,1)[0]])
            return result
        draw=lambda k:int(torch.randint(k,(1,),generator=g))
        for _ in range(n):
            t=self.tasks[draw(len(self.tasks))]; p=self.parents[t][draw(len(self.parents[t]))]; ids=self.groups[t][p]
            result.append(ids[draw(len(ids))])
        return result

class ManifestDataset(Dataset):
    """Explicit window reader; source table remains immutable. No guessed timing/pairing."""
    fixture=False
    def __init__(self,path,stage,split='train'):
        self.path=Path(path); self.stage=stage; self.rows=[]
        with self.path.open() as f:
            for line in f:
                r=json.loads(line)
                if r['split']!=split:continue
                if split=='train' and not r.get('allow_p1' if stage=='P1' else 'allow_p3',False):continue
                if split!='train' and not r.get('allow_eval',False):continue
                if any(r.get(x,False) for x in ('sim','retarget','model_generated_video')): raise ValueError('Non-real source')
                if stage!='P1' and r.get('robot_schema')!='piper14_next_feedback_v1': raise ValueError('Unverified action schema')
                if split!='train' and r.get('truth_source') not in ('human','recorded_feedback'): raise ValueError('Evaluation requires independent truth')
                self.rows.append(r)
        if not self.rows: raise ValueError(f'No qualified {stage}/{split} windows')
        seen={}
        for r in self.rows:
            p=r['parent_component']
            if p in seen and seen[p]!=r['split']: raise ValueError('Parent split leakage')
            seen[p]=r['split']
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        from .source import read_window
        return read_window(self.rows[i],self.stage)

class DataPackageDataset(Dataset):
    """Adapter for E82-D capture+numeric/window NPZ package. No source action re-pairing."""
    fixture=False
    def __init__(self,root,stage,profile=False,limit=None):
        self.root=Path(root); ready=read_ready(self.root); self.stage=stage; self.rows=[]
        entry=ready['READY_data.json'].get('entrypoints',{})
        self.captures_path=Path(entry.get('captures',self.root/'manifest/captures.jsonl'))
        captures=[json.loads(l) for l in self.captures_path.read_text().splitlines()]
        routes={}
        if entry.get('video_routes'):
            routes={r['canonical_id']:r for r in map(json.loads,Path(entry['video_routes']).read_text().splitlines())}
        parent_splits={}
        for r in captures:
            p=r['parent_component']
            if p in parent_splits and parent_splits[p]!=r['split']:raise ValueError('Cross-split parent in data package')
            parent_splits[p]=r['split']
        prefix='p1' if stage=='P1' else 'p3'
        for r in captures:
            if r['split']!='train' or not r.get('allow_p1' if stage=='P1' else 'allow_p3'):continue
            if not r.get('train_ready_p1' if stage=='P1' else 'train_ready_p3') and not profile:continue
            if r.get('schema_version')!='e82-piper-next-feedback-v1': raise ValueError('Unknown data action schema')
            if not r.get('window_cache'):continue
            if routes:
                route=routes.get(r['canonical_id'])
                if not route or not route.get('all_available'):continue
                r=dict(r,camera_paths={c['camera']:c['path'] for c in route['cameras']})
            with np.load(r['window_cache'],allow_pickle=False) as z:
                anchors=z[prefix+'_anchor']
                for j,a in enumerate(anchors):
                    self.rows.append(dict(r,anchor=int(a),O_video_rows=z[prefix+'_O_video_rows'][j].tolist(),W_video_rows=z[prefix+'_W_video_rows'][j].tolist(),O_age_s=z[prefix+'_O_age_s'][j].tolist(),decision_time=float(z[prefix+'_decision_time'][j]),profile_candidate_only=profile and not r.get('train_ready_p1' if stage=='P1' else 'train_ready_p3',False)))
                    if limit and len(self.rows)>=limit:break
            if limit and len(self.rows)>=limit:break
        if not self.rows:raise ValueError(f'No admitted {stage} train windows; profile flag admits candidates for timing only')
        self.refs=dict(parent_ready=sha256(self.root/'READY_parent_split.json'),data_ready=sha256(self.root/'READY_data.json'))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        from .source import read_window
        return read_window(self.rows[i],self.stage)
