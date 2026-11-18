import collections,json
from pathlib import Path
from .common import *
from .data import load_data,HORIZONS

def main():
    dest=ART/'preflight_01';dest.mkdir(exist_ok=False)
    p,ready,train,dev,coverage=load_data();summary={}
    for name,data in (('train',train),('dev',dev)):
        counts=collections.Counter(); parents=collections.defaultdict(set)
        for i,row in enumerate(data.rows):
            for h in HORIZONS:
                ids,reason=data.index.resolve(row,h);counts[(h,reason or 'valid')]+=1
                if ids is not None:parents[h].add(row['parent_component'])
            if i%5000==0:atomic(dest/'status.json',dict(status='checking',pool=name,done=i,total=len(data.rows)))
        summary[name]={str(h):dict(windows=counts[(h,'valid')],parents=len(parents[h]),excluded=len(data.rows)-counts[(h,'valid')],reasons={reason:n for (hh,reason),n in counts.items() if hh==h and reason!='valid'}) for h in HORIZONS}
    # A +100 missing frame removes that horizon supervision, not valid +10/+25/+50 supervision.
    refs=dict(ready=sha(INPUT/'READY_V3_cache_windows.json'),P1_windows=sha(ready['windows']['P1']),
              parent=p.refs['parent'],data=p.refs['data'],exclusion=p.refs['exclusion'],
              residual_stats=sha(E88/'artifacts/cpu_audit_01/residual_stats.json'),
              dev_windows=sha(C/'artifacts/dev_v1/windows_v2.json'),dev_captures=sha(C/'artifacts/dev_v1/captures.json'),
              train_cache_contract=sha(Path(ready['cache'])/'contract.json'),dev_cache_contract=sha(C/'artifacts/dev_features_v1/contract.json'),
              language=sha(C/'artifacts/language_v1/READY_language.json'))
    result=dict(schema='e90-data-manifest-v1',refs=refs,manifest_hash=object_hash(refs),availability=summary,
                text=coverage,train_windows=len(train.rows),train_parents=242,
                frozen_dev_total_parents=sum(s=='dev' for s in p.split.values()),cached_dev_parents=len({r['parent_component'] for r in dev.rows}),
                excluded_canonical=list(p.excluded),no_new_cache=True,mask_policy='per-horizon invalid endpoints masked; never shift frames or labels')
    atomic(dest/'manifest.json',result);atomic(dest/'status.json',dict(status='completed',brt=brt()))
    report('## 数据与缓存复核\n'+json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(availability=summary,text=coverage),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
