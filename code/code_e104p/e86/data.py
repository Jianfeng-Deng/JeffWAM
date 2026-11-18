"""Pinned rev4/P3 V3 reader (explicit immutable V1 rollback). No writes to publishers or source directories."""
from pathlib import Path
from collections import defaultdict,OrderedDict
import json,numpy as np,torch
from torch.utils.data import Dataset
from .common import *
from .v3_contract import *
CAMS=('cam_high','cam_left_wrist','cam_right_wrist')
JOINTS=list(range(6))+list(range(7,13));GRIPS=[6,13]
P3_RECEIPT_HASH='a6df5dbada0811af6585ed27ff73c62e1b7b6397f07ecd47045e7d0310901ca0'
def verify_file(path,expected):
    if sha(path)!=expected:raise ValueError('Hash mismatch: '+str(path))
def verify_receipt(path,expected=None):
    if expected:verify_file(path,expected)
    d=json.loads(Path(path).read_text())
    for e in d.get('files',[]):verify_file(e['path'],e['sha256'])
    return d
def assert_parents(captures,split):
    seen={}
    for r in captures:
        p=r['parent_component'];s=r['split']
        if split.get(p)!=s or (p in seen and seen[p]!=s):raise ValueError('Parent split leakage: '+p)
        seen[p]=s
class Package:
    def __init__(self,root=R,version="V3"):
        self.root=Path(root)
        self.parent=verify_receipt(self.root/'READY_parent_split_rev4.json',PARENT_HASH)
        self.ready=verify_receipt(self.root/'READY_data_rev4.json',DATA_HASH)
        verify_file(self.ready['parent_split']['path'],self.ready['parent_split']['sha256'])
        # Rev4 snapshot pins inherited rev3 numeric/window files. Original immutable receipt supplies hashes.
        old=self.ready['derived_from'];verify_file(old['path'],old['sha256']);olddata=json.loads(Path(old['path']).read_text())
        self.hashes={e['path']:e['sha256'] for e in olddata['files']+self.ready['files']+self.parent['files']}
        for key in ('captures','video_routes'):
            path=self.ready['entrypoints'][key]
            if path not in self.hashes:raise ValueError('Unpublished manifest dependency: '+path)
            verify_file(path,self.hashes[path])
        self.captures=rows(self.ready['entrypoints']['captures']);self.routes={r['canonical_id']:r for r in rows(self.ready['entrypoints']['video_routes'])}
        self.split=json.loads(next(Path(e['path']).read_text() for e in self.parent['files'] if Path(e['path']).name=='locked_split_rev4.json'))['components']
        assert_parents(self.captures,self.split)
        if version not in ('V1','V3'):raise ValueError('Unknown P3 version')
        self.version=version
        self.exclusion=verify_receipt(EXCLUSION_PATH,EXCLUSION_HASH)
        self.excluded={e['canonical_id'] for e in self.exclusion['excluded']}
        if self.excluded!={EXCLUDED_CANONICAL} or not self.exclusion['permanent']:raise ValueError('Exclusion changed')
        self.receipt_path=self.root/('P3_ready_'+version+'.json')
        self.receipt=verify_receipt(self.receipt_path,V3_RECEIPT_HASH if version=='V3' else P3_RECEIPT_HASH)
        expected=V3_HASH if version=='V3' else V1_HASH
        src=self.receipt['source'];verify_file(src['path'],expected)
        if src['sha256']!=expected or self.receipt['status']!='ready_frozen_'+version:raise ValueError('Unapproved P3 source')
        self.p3=json.loads(Path(src['path']).read_text())
        if version=='V3':
            adoption=self.receipt['adoption_record'];verify_file(adoption['path'],adoption['sha256'])
            authority=json.loads(Path(adoption['path']).read_text())
            if not authority['approved'] or authority['pool_sha256']!=expected:raise ValueError('Missing V3 user approval')
            if self.receipt['exclusion_list']!={'path':EXCLUSION_PATH,'sha256':EXCLUSION_HASH}:raise ValueError('Exclusion reference mismatch')
            verify_entries(self.p3)
        self.entries={e.get('entry_id',e['canonical_id']):e for e in self.p3['entries']}
        self.by_canonical=defaultdict(list)
        for e in self.entries.values():
            if e['canonical_id'] in self.excluded:raise ValueError('Permanently excluded canonical in P3')
            if self.split.get(e['parent_id'])!='train' or e['split']!='train' or e.get('recovery'):raise ValueError('P3 ineligible parent/recovery')
            self.by_canonical[e['canonical_id']].append(e)
        if len(self.entries)!=(284 if version=='V3' else 206):raise ValueError('P3 entry count mismatch')
        self.expected_windows=34777 if version=='V3' else 24477
        if sum(e['windows'] for e in self.entries.values())!=self.expected_windows:raise ValueError('P3 count mismatch')
        self.refs=dict(parent=PARENT_HASH,data=DATA_HASH,p3=sha(self.receipt_path),pool=expected,exclusion=EXCLUSION_HASH,version=version)
    def checked_np(self,path):
        if path not in self.hashes:raise ValueError('Unpublished numeric/window dependency: '+path)
        verify_file(path,self.hashes[path]);return dict(np.load(path,allow_pickle=False))
    def windows(self,stage):
        if stage not in ('P1','P3'):raise ValueError('Training stage must be P1/P3')
        result=[];prefix=stage.lower();seen=set()
        for c in self.captures:
            cid=c['canonical_id']
            if cid in self.excluded or c['split']!='train' or not c.get('train_ready_p1') or (stage=='P3' and cid not in self.by_canonical):continue
            if not self.routes.get(cid,{}).get('all_available'):continue
            w=self.checked_np(c['window_cache']);nums=self.checked_np(c['numeric_cache'])
            entries=self.by_canonical[cid] if stage=='P3' else [None]
            available={int(a):j for j,a in enumerate(w[prefix+'_anchor'])}
            for entry in entries:
                anchors=entry['window_anchor_raw_rows'] if entry else list(available)
                for a in anchors:
                    if a not in available:raise ValueError('Anchor absent from authoritative cache')
                    if (cid,a) in seen:raise ValueError('Duplicate P3 window')
                    seen.add((cid,a));j=available[a]
                    if stage=='P3':validate_interval(nums,a,w[prefix+'_O_video_rows'][j],w[prefix+'_W_video_rows'][j],entry)
                    row=build_window(c,nums,a,w[prefix+'_O_video_rows'][j],w[prefix+'_W_video_rows'][j],stage,entry)
                    if entry:row.update(entry_id=entry.get('entry_id',cid),cycle_id=entry.get('cycle_id'),subinterval_raw_rows_exclusive=entry.get('subinterval_raw_rows_exclusive'))
                    row['stratum']=entry['stratum'] if entry else stratum(c)
                    result.append(row)
        if stage=='P3' and len(result)!=self.expected_windows:raise ValueError(f'P3 {self.version} window mismatch {len(result)}')
        if stage=='P1' and len(result)!=107057:raise ValueError(f'P1 exclusion/count mismatch {len(result)}')
        return result

def stratum(c):
    source=c['source_collection']
    if 'demo6' in source:
        # Dataset manifests use one source_collection for both variants. Identify by
        # existing image receipts, then verify mapping against every V3 entry.
        for root in (C/'artifacts/features_v3',C/'artifacts/dev_features_v1'):
            receipt=root/c['canonical_id']/'receipt.json'
            if receipt.exists():
                cameras=json.loads(receipt.read_text())['cameras'];hw=next(x['original_hw'] for x in cameras if x['camera']=='cam_high')
                if hw==[480,640]:return 'demo6early'
                if hw==[720,1280]:return 'demo6native'
                raise ValueError('Unknown demo6 source resolution')
        raise ValueError('Unresolved demo6 stratum; no cached image metadata')
    return next((k for k in ('demo3','demo4') if k in source),source)

def verify_entries(pool):
    import hashlib
    digest=lambda d:hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    for e in pool['entries']:
        if digest({k:v for k,v in e.items() if k!='entry_sha256'})!=e['entry_sha256']:raise ValueError('Entry hash mismatch')
        if len(set(e['window_anchor_raw_rows']))!=e['windows']:raise ValueError('Duplicate/count entry anchors')
        if e['canonical_id']==EXCLUDED_CANONICAL:raise ValueError('Permanently excluded canonical')
    if digest(pool['entries'])!=pool['entries_sha256']:raise ValueError('Entry collection hash mismatch')

def validate_interval(z,a,oidx,widx,entry):
    interval=entry.get('subinterval_raw_rows_exclusive')
    if not interval:return # V1 legacy whole capture; original cache remains authoritative.
    lo,hi=interval
    if not (lo<=a-6 and a+50<hi):raise ValueError('Anchor crosses interval')
    for grid in (oidx,widx):
        grid=np.asarray(grid)
        if not ((grid>=lo)&(grid<hi)).all():raise ValueError('O/W camera crosses interval')
    actions=z['original_action_index'][a:a+50]
    if not np.array_equal(actions,np.arange(a+1,a+51)) or not ((actions>=lo)&(actions<hi)).all():raise ValueError('Original action crosses interval')
    if 'original_observation_index' in z and not np.array_equal(z['original_observation_index'][a-6:a+51],np.arange(a-6,a+51)):raise ValueError('Original observation mapping changed')

def build_window(c,z,a,oidx,widx,stage,p3entry=None):
    if c['canonical_id']==EXCLUDED_CANONICAL:raise ValueError('Permanently excluded canonical')
    if p3entry:validate_interval(z,a,oidx,widx,p3entry)
    seg=z['continuous_segment'];times=z['timestamp'];avail=z['state_available_time'];ct=z['camera_exposure_timestamp'];rc=z['camera_received_timestamp']
    # Current+future coverage uses unchanged E82 pairing and full segment contract.
    if seg[a-6]!=seg[a+50]:raise ValueError('Window crosses seam')
    hist=[];hvalid=[];meta=[]
    for off in (-24,-18,-12,-6,0):
        j=a+off;valid=j>=0 and seg[max(j,0)]==seg[a]
        if valid:
            valid=bool(np.isfinite(avail[j]) and avail[j]<=avail[a]+1e-7)
            if 'state_source_brackets' in z:
                brackets=z['state_source_brackets'][j];valid=bool(valid and np.isfinite(brackets).all() and (brackets[:,1]-brackets[:,0]).max()<=.075)
        if valid and c['split']=='train':valid=bool(z['p1_source_mask' if stage=='P1' else 'p3_source_mask'][j:a+1].all())
        if p3entry and p3entry.get('subinterval_raw_rows_exclusive'):
            lo,hi=p3entry['subinterval_raw_rows_exclusive'];valid &= lo<=j<hi
        indices=[];ages=[]
        for k in range(3):
            if valid:
                ac=np.maximum.accumulate(np.maximum(ct[:,k],rc[:,k]));r=int(np.searchsorted(ac,avail[j],side='right')-1)
                ok=r>=0 and seg[max(r,0)]==seg[a] and 0<=avail[j]-ct[max(r,0),k]<=.100001
            else:r=-1;ok=False
            if p3entry and p3entry.get('subinterval_raw_rows_exclusive'):
                lo,hi=p3entry['subinterval_raw_rows_exclusive'];ok=bool(ok and lo<=r<hi)
            indices.append(r if ok else -1);ages.append(float(avail[j]-ct[r,k]) if ok else 0.)
        good=valid and min(indices)>=0
        hist.append(indices if good else [-1]*3);hvalid.append(good)
        meta.append([[float(times[max(j,0)]-times[a]),ages[k]] for k in range(3)])
    if not hvalid[-1]:raise ValueError('Current image unavailable')
    if not np.array_equal(np.asarray(hist[-1]),oidx[-1]):raise ValueError('Current O mapping changed')
    if stage=='P3' and not np.array_equal(z['original_action_index'][a:a+50],np.arange(a+1,a+51)):raise ValueError('Next feedback pairing changed')
    return dict(canonical_id=c['canonical_id'],parent_component=c['parent_component'],session_id=c.get('session_id'),split=c['split'],anchor=a,obs_rows=hist,history_valid=hvalid[:4],future_rows=np.asarray(widx).tolist(),obs_meta=meta,decision_wait_s=float(avail[a]-times[a]),demo3='demo3' in c['source_collection'],quality_flag=('accept' if stage=='P3' else 'uncertain') if c['split']=='train' else 'unknown')

class CappedParentSampler:
    """Global parent-balanced, constrained only by the hard cap. Hard cap by number of drawn windows (cumulative)."""
    def __init__(self,rows,seed=86,cap=.45,strategy="parent_equal",stratum_weights=None):
        if not 0<=cap<=.45:raise ValueError('demo3 cap must be <=.45')
        if strategy not in ('parent_equal','stratified'):raise ValueError('Unknown sampling strategy')
        if any(r.get('split','train')!='train' or r.get('canonical_id')==EXCLUDED_CANONICAL for r in rows):raise ValueError('Ineligible training sampler row')
        self.strategy=strategy;self.stratum_weights=stratum_weights or {};self.strata=defaultdict(lambda:defaultdict(list))
        for i,r in enumerate(rows):self.strata[r.get('stratum','unknown')][r['parent_component']].append(i)
        self.stratum_names=sorted(self.strata);weights=[self.stratum_weights.get(k,1.) for k in self.stratum_names]
        if any(v<0 for v in weights) or sum(weights)<=0:raise ValueError('Invalid stratum weights')
        self.stratum_prob=torch.tensor(weights,dtype=torch.float64);self.stratum_prob/=self.stratum_prob.sum()
        self.rows=rows;self.seed=seed;self.cap=cap;self.groups={False:defaultdict(list),True:defaultdict(list)}
        for i,r in enumerate(rows):self.groups[bool(r['demo3'])][r['parent_component']].append(i)
        self.parents={k:sorted(v) for k,v in self.groups.items()};self.all_groups=defaultdict(list)
        for i,r in enumerate(rows):self.all_groups[r['parent_component']].append(i)
        self.all_parents=sorted(self.all_groups);self.count=0;self.demo3_count=0
        if not self.parents[False]:raise ValueError('Cannot enforce <=45% with only demo3')
    def sample(self,n):
        g=generator(self.seed,self.count,'sampler');out=[]
        draw=lambda n:int(torch.randint(n,(),generator=g))
        for _ in range(n):
            # Global parent-equal first, then window-equal within parent. Cap only vetoes forbidden demo3 draws.
            eligible=self.demo3_count+1<=int((self.count+1)*self.cap+1e-9)
            if self.strategy=='stratified':
                st=self.stratum_names[int(torch.multinomial(self.stratum_prob,1,generator=g))];groups=self.strata[st];parents=sorted(groups);p=parents[draw(len(parents))];ids=groups[p]
            else:
                p=self.all_parents[draw(len(self.all_parents))];ids=self.all_groups[p]
            chosen=ids[draw(len(ids))]
            if self.rows[chosen]['demo3'] and not eligible:
                parents=self.parents[False];p=parents[draw(len(parents))];ids=self.groups[False][p];chosen=ids[draw(len(ids))]
            demo=bool(self.rows[chosen]['demo3']);out.append(chosen)
            self.count+=1;self.demo3_count+=int(demo)
        return out
    def state_dict(self):return dict(count=self.count,demo3_count=self.demo3_count,seed=self.seed,cap=self.cap,strategy=self.strategy,stratum_weights=self.stratum_weights)
    def load_state_dict(self,s):
        if s.get('strategy','parent_equal')!=self.strategy or s.get('stratum_weights',{})!=self.stratum_weights:raise ValueError('Sampler strategy mismatch')
        if s['seed']!=self.seed or s['cap']!=self.cap:raise ValueError('Sampler config mismatch')
        self.count=s['count'];self.demo3_count=s['demo3_count']

def fit_stats(package,windows):
    """One fixed P3 train-only scale; equal parents, up to 8 windows per parent."""
    by=defaultdict(list)
    for w in windows:by[w['parent_component']].append(w)
    cs={c['canonical_id']:c for c in package.captures};numeric={};means=[];seconds=[];qmeans=[];qseconds=[]
    for p,ws in sorted(by.items()):
        actions=[];qs=[]
        for w in ws[::max(1,len(ws)//8)][:8]:
            cid=w['canonical_id'];z=numeric.setdefault(cid,package.checked_np(cs[cid]['numeric_cache'])) if cid not in numeric else numeric[cid]
            a=w['anchor'];q=z['q'][a].copy();anchor=q.copy();anchor[GRIPS]=0
            actions.append(z['action'][a:a+50]-anchor);qs.append(q)
        x=np.asarray(actions);q=np.asarray(qs);means.append(x.mean(0));seconds.append((x*x).mean(0));qmeans.append(q.mean(0));qseconds.append((q*q).mean(0))
    mean=np.mean(means,0);floor=np.array([.01]*14);floor[GRIPS]=.001;std=np.maximum(np.sqrt(np.maximum(np.mean(seconds,0)-mean**2,0)),floor)
    qm=np.mean(qmeans,0);qs=np.maximum(np.sqrt(np.maximum(np.mean(qseconds,0)-qm**2,0)),floor)
    return dict(action_mean=mean.tolist(),action_std=std.tolist(),q_mean=qm.tolist(),q_std=qs.tolist(),fitted_split='train',parents=len(by),source=package.refs,gripper_contract='signed absolute, no clamp; only 12 joints anchored',world='raw fixed SigLIP2 grid, no learned target projection')
class CachedDataset(Dataset):
    def __init__(self,windows,captures,cache,stats,text_cache=None):
        if any(r.get('canonical_id')==EXCLUDED_CANONICAL for r in windows):raise ValueError('Permanently excluded canonical')
        self.rows=windows;self.captures={c['canonical_id']:c for c in captures};self.cache=Path(cache);self.stats=stats;self.text_cache=text_cache or {};self.lru=OrderedDict();self.num={};self.parents={p:i for i,p in enumerate(sorted(set(r['parent_component'] for r in windows)))}
    def __len__(self):return len(self.rows)
    def features(self,cid,cam,ids):
        key=(cid,cam)
        if key not in self.lru:
            roots=[self.cache]
            overlay=self.cache/'overlay.json'
            if overlay.exists():
                meta=json.loads(overlay.read_text());verify_file(Path(meta['base_cache'])/'contract.json',meta['base_contract_sha256'])
                roots.append(Path(meta['base_cache']))
            arrays=[];lookup={}
            for root in roots:
                dest=root/cid
                if not (dest/'receipt.json').exists():continue
                meta=json.loads((dest/'receipt.json').read_text());rec=meta['cameras'][cam]
                if rec['camera']!=CAMS[cam]:raise ValueError('Camera mapping changed')
                z=np.load(dest/(CAMS[cam]+'.npy'),mmap_mode='r');ix=np.load(dest/(CAMS[cam]+'_rows.npy'))
                source=len(arrays);arrays.append(z)
                for i,a in enumerate(ix):
                    if int(a) in lookup:raise ValueError('Duplicate overlay cache row')
                    lookup[int(a)]=(source,i)
            if not arrays:raise ValueError('No cache capture: '+cid)
            self.lru[key]=(arrays,lookup)
            if len(self.lru)>12:self.lru.popitem(last=False)
        self.lru.move_to_end(key);arrays,lookup=self.lru[key]
        def feature(i):
            if i<0:return np.zeros((8,768),np.float32)
            source,j=lookup[int(i)];return arrays[source][j].astype(np.float32)
        return np.stack([feature(i) for i in ids])
    def __getitem__(self,i):
        r=self.rows[i];cid=r['canonical_id'];c=self.captures[cid];a=r['anchor']
        if r['split']!='train' and (cid in self.text_cache or (cid,a) in self.text_cache):raise ValueError('Model language on evaluation parent')
        if cid not in self.num:
            self.num[cid]=dict(np.load(c['numeric_cache'],allow_pickle=False))
            if len(self.num)>8:self.num.pop(next(iter(self.num)))
        z=self.num[cid];q=z['q'][a].copy();anchor=q.copy();anchor[GRIPS]=0
        obs=np.stack([self.features(cid,k,np.array(r['obs_rows'])[:,k]) for k in range(3)],axis=1)
        world=np.stack([self.features(cid,k,np.array(r['future_rows'])[:,k]) for k in range(3)],axis=1).reshape(72,768)
        text=self.text_cache.get((cid,a),self.text_cache.get(cid,np.empty((0,768),np.float32)))
        if r['split']!='train':text=self.text_cache.get('__human_default__',np.empty((0,768),np.float32))
        if r['split']!='train' and (cid in self.text_cache or (cid,a) in self.text_cache):raise ValueError('Model language on evaluation parent')
        at=self.text_cache.get(('align',cid,a),np.zeros(768,np.float32))
        def tensor(x,dtype=torch.float32):return torch.as_tensor(np.array(x),dtype=dtype)
        return dict(obs=tensor(obs),world=tensor(world),obs_meta=tensor(r['obs_meta']),history_valid=tensor(r['history_valid'],torch.bool),current_valid=torch.ones(3,dtype=torch.bool),world_valid=torch.ones(72,dtype=torch.bool),text=tensor(text),text_valid=torch.ones(len(text),dtype=torch.bool),q_normalized=tensor((q-np.array(self.stats['q_mean']))/np.array(self.stats['q_std'])),q_anchor=tensor(anchor),action=tensor(z['action'][a:a+50]),action_mean=tensor(self.stats['action_mean']),action_std=tensor(self.stats['action_std']),state_valid=torch.tensor(True),parent_index=torch.tensor(self.parents[r['parent_component']]),decision_wait_s=torch.tensor(r['decision_wait_s']),align_text=tensor(at),align_valid=torch.tensor(bool(np.any(at))),quality_flag=torch.tensor((1 if r['quality_flag']=='accept' else 2) if r['split']=='train' else 0))
def collate(items):
    ntext=max(1,max(len(r['text']) for r in items));out={}
    for k in items[0]:
        if k in ('text','text_valid'):
            zs=[]
            for r in items:
                z=r[k];pad=torch.zeros((ntext-len(z),)+z.shape[1:],dtype=z.dtype);zs.append(torch.cat((z,pad)))
            out[k]=torch.stack(zs)
        else:out[k]=torch.stack([r[k] for r in items])
    return out
def move(b,device):return {k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}
