import copy, collections, json
from pathlib import Path
import numpy as np
import torch
from e86.common import generator, sha
from e86.data import CachedDataset, collate, Package, stratum, verify_receipt
from e88.data import Data as BaseData
from e88.audit import HorizonIndex
from .common import C, E88, INPUT
from .model import SOURCES

HORIZONS = (10, 25, 50, 100)

def sample_horizons(step, seed=90, maximum=100, continuous=True):
    fixed = tuple(h for h in (10,25,50,100,200,300) if h <= maximum)
    if not continuous: return fixed
    g = generator(seed, step, 'continuous_horizon')
    j = int(torch.randint(len(fixed)-1, (), generator=g))
    h = int(torch.randint(fixed[j], fixed[j+1]+1, (), generator=g))
    return tuple(dict.fromkeys((*fixed, h)))

class ImageDataset(CachedDataset):
    """P1 reads images and timing only; action/state arrays never become model inputs."""
    def features(self,cid,cam,ids):
        # Keep only read-only mmap handles and row indexes in RAM. No feature copies on disk.
        from e86.data import CAMS,verify_file
        if not hasattr(self,'feature_maps'):self.feature_maps={}
        key=(cid,cam)
        if key not in self.feature_maps:
            roots=[self.cache]
            if (self.cache/'overlay.json').exists():
                meta=json.loads((self.cache/'overlay.json').read_text())
                verify_file(Path(meta['base_cache'])/'contract.json',meta['base_contract_sha256'])
                roots.append(Path(meta['base_cache']))
            parts=[];seen=set()
            for root in roots:
                dest=root/cid
                if not (dest/'receipt.json').exists():continue
                receipt=json.loads((dest/'receipt.json').read_text())
                if receipt['cameras'][cam]['camera']!=CAMS[cam]:raise ValueError('Camera mapping changed')
                rows=np.load(dest/(CAMS[cam]+'_rows.npy'))
                if np.any(np.diff(rows)<=0) or seen.intersection(map(int,rows)):raise ValueError('Duplicate or unsorted cache rows')
                seen.update(map(int,rows))
                parts.append((rows,np.load(dest/(CAMS[cam]+'.npy'),mmap_mode='r')))
            if not parts:raise ValueError('Missing cached capture')
            self.feature_maps[key]=parts
        ids=np.asarray(ids);out=np.zeros((len(ids),8,768),np.float32);found=ids<0
        for rows,features in self.feature_maps[key]:
            j=np.searchsorted(rows,ids);ok=(j<len(rows))
            ok=ok & (rows[np.minimum(j,len(rows)-1)]==ids)
            out[ok]=features[j[ok]];found|=ok
        if not found.all():raise KeyError('Frame absent from immutable cache')
        return out

    def __getitem__(self, i):
        r = self.rows[i]; cid = r['canonical_id']; a = r['anchor']
        if r['split'] != 'train' and any(k in self.text_cache for k in (cid, (cid,a))):
            raise ValueError('Model language on frozen evaluation parent')
        obs = np.stack([self.features(cid,k,np.array(r['obs_rows'])[:,k]) for k in range(3)], axis=1)
        text = self.text_cache.get((cid,a), self.text_cache.get('__human_default__', np.empty((0,768),np.float32)))
        if r['split'] != 'train': text = self.text_cache['__human_default__']
        return dict(obs=torch.tensor(obs), obs_meta=torch.tensor(r['obs_meta']),
                    history_valid=torch.tensor(r['history_valid']), current_valid=torch.ones(3,dtype=torch.bool),
                    text=torch.tensor(text), text_valid=torch.ones(len(text),dtype=torch.bool),
                    parent_index=torch.tensor(self.parents[r['parent_component']]),
                    align_valid=torch.tensor(False), align_text=torch.zeros(768))

class Data(BaseData):
    def __init__(self, rows, captures, cache, stats, residual_stats, text_cache=None, stage='P1'):
        self.base = (ImageDataset if stage == 'P1' else CachedDataset)(rows,captures,cache,stats,text_cache)
        self.rows=rows; self.index=HorizonIndex(cache,captures); self.norm=residual_stats['scale']
        self.labels={}; self.narration_vectors={}; self.stage=stage

    def get(self, i, horizons=HORIZONS, language='instruction', stage='P1'):
        # "long" in E88 preserves the existing per-anchor E83 instruction. No narration is added here.
        b = super().get(i,horizons,'long' if language=='instruction' else language,stage)
        b['source_id'] = torch.tensor(SOURCES.index(self.rows[i].get('source','piper')))
        return b

    def batch(self, ids, horizons=HORIZONS, language='instruction', stage='P1'):
        b=collate([self.get(i,horizons,language,stage) for i in ids]); b['world_positions']=b['world_positions'][0]
        return b

    def shifted(self, i, k, language='instruction', stage='P1'):
        """Reentry at the exact first-hop encoded endpoint, causal shifted real history.

        Does not require irrelevant shifted +10/+25/+50 frames or read actions.
        """
        r=copy.deepcopy(self.rows[i]); z,_=self.index.get(r['canonical_id']); middle,_=self.index.resolve(r,k)
        endpoint,_=self.index.resolve(r,2*k)
        if middle is None or endpoint is None:return None
        a=r['anchor']+k; original=r['anchor']; r['anchor']=a
        r['obs_rows']=[]; r['obs_meta']=[]; r['history_valid']=[]
        lo,hi=r.get('subinterval_raw_rows_exclusive') or (0,len(z['timestamp']))
        for off in (-24,-18,-12,-6,0):
            j=a+off; ok=lo<=j<hi and z['continuous_segment'][max(lo,j)]==z['continuous_segment'][a]
            ids=[]; meta=[]
            for cam in range(3):
                if off==0: idx=middle[cam]
                elif ok:
                    available=np.maximum.accumulate(np.maximum(z['camera_exposure_timestamp'][:,cam],z['camera_received_timestamp'][:,cam]))
                    idx=int(np.searchsorted(available,z['state_available_time'][j],side='right')-1)
                else: idx=-1
                valid=lo<=idx<hi and z['continuous_segment'][max(lo,idx)]==z['continuous_segment'][a]
                if off!=0 and valid:
                    valid=0<=z['state_available_time'][j]-z['camera_exposure_timestamp'][idx,cam]<=.100001
                    valid=valid and z['p1_source_mask'][j:a+1].all()
                if valid:
                    try:self.base.features(r['canonical_id'],cam,[idx])
                    except (KeyError,ValueError):valid=False
                ids.append(idx if valid else -1)
                meta.append([float(z['timestamp'][max(lo,j)]-z['timestamp'][a]),float(z['state_available_time'][a if off==0 else max(lo,j)]-z['camera_exposure_timestamp'][idx,cam]) if valid else 0.])
            good=min(ids)>=0
            if off==0 and not good:return None
            r['obs_rows'].append(ids if good else [-1]*3); r['obs_meta'].append(meta)
            if off!=0:r['history_valid'].append(good)
        r['_stage']='P1'
        temp=copy.copy(self); temp.rows=[r]; temp.base=copy.copy(self.base); temp.base.rows=[r]
        b=temp.get(0,(k,),language,'P1')
        # Original and shifted queries must select exactly the same +2k frame rows.
        got,_=temp.index.resolve(r,k)
        if got!=endpoint:raise ValueError('Chain endpoint mismatch')
        return b

def instruction_cache(rows):
    root=C/'artifacts/language_v1'; receipt=verify_receipt(root/'READY_language.json')
    ix=json.loads((root/'index.json').read_text()); vectors=np.load(root/'vectors.npy',mmap_mode='r')
    sl=lambda ab:vectors[ab[0]:ab[1]].astype('f4')
    cache={'__human_default__':sl(ix['human_instruction'])}; bycid={}; covered=0
    for r in rows:
        e=ix['instructions'].get(r['canonical_id'])
        if r['split']=='train' and e and e['range'][0]<=r['anchor']<=e['range'][1]:
            cid=r['canonical_id']
            if cid not in bycid:bycid[cid]=sl(e['slice'])
            cache[(cid,r['anchor'])]=bycid[cid]; covered+=1
    return cache,dict(existing_instruction_windows=covered,fallback_windows=len(rows)-covered,
                      narration=False,language_ready_sha256=sha(root/'READY_language.json'))

def load_data():
    p=Package(); ready=verify_receipt(INPUT/'READY_V3_cache_windows.json')
    if p.refs!=ready['refs']:raise ValueError('READY drift')
    rows=json.loads(Path(ready['windows']['P1']).read_text())
    if len(rows)!=107057 or len({r['parent_component'] for r in rows})!=242:raise ValueError('P1 pool changed')
    rows=[dict(r,_stage='P1',source='piper') for r in rows]
    if any(p.split[r['parent_component']]!='train' or r['canonical_id'] in p.excluded for r in rows):raise ValueError('Ineligible P1 row')
    stats=json.loads((INPUT/'stats.json').read_text()); norm=json.loads((E88/'artifacts/cpu_audit_01/residual_stats.json').read_text())
    if norm['fitted_split']!='train' or norm['refs']!=p.refs:raise ValueError('Residual stats provenance')
    text,coverage=instruction_cache(rows)
    train=Data(rows,p.captures,ready['cache'],stats,norm,text)
    caps=json.loads((C/'artifacts/dev_v1/captures.json').read_text())
    devrows=json.loads((C/'artifacts/dev_v1/windows_v2.json').read_text())
    cm={c['canonical_id']:c for c in caps}
    devrows=[dict(r,_stage='P1',source='piper',stratum=stratum(cm[r['canonical_id']])) for r in devrows]
    if any(p.split[r['parent_component']]!='dev' for r in devrows):raise ValueError('Frozen dev split changed')
    dev=Data(devrows,caps,C/'artifacts/dev_features_v1',stats,norm,{'__human_default__':text['__human_default__']})
    return p,ready,train,dev,coverage
