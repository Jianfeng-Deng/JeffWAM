import os,sys,json,time,collections,traceback
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e93_dev_diag_20261114_01';sys.path.insert(0,str(T/'code_e90'));os.environ['PYTHONDONTWRITEBYTECODE']='1'
from e86.common import atomic,sha
from e90.sampling import MultiSourceSampler
from evaluate_cpu_tail import aggregate
started=time.monotonic()
try:
 ready=json.loads((T/'code_e86/artifacts/E86-V3-20261113T144520/READY_V3_cache_windows.json').read_text());path=Path(ready['windows']['P1']);manifest=json.loads((T/'code_e90/artifacts/preflight_01/manifest.json').read_text());assert sha(path)==manifest['refs']['P1_windows'];rows=json.loads(path.read_text());cfg=json.loads((T/'code_e90/artifacts/e90_p1_20261114_01/config.json').read_text());sampler=MultiSourceSampler(rows,seed=90,weights=cfg['source_weights'],caps=cfg['source_caps'],demo3_cap=cfg['demo3_cap']);counts=collections.Counter()
 for step in range(5000):
  counts.update(sampler.sample(32))
  if step%500==0:atomic(R/'exposure_status.json',dict(status='replaying_indices_only',step=step))
 expected=json.loads((T/'code_e90/artifacts/e90_p1_20261114_01/train_summary.json').read_text())['sampler'];assert sampler.state_dict()==expected,(sampler.state_dict(),expected)
 bykey={(r['canonical_id'],r['anchor']):i for i,r in enumerate(rows)};sel=json.loads((R/'train64_selection.json').read_text());ex=[dict(r,index=i,exposures=counts[bykey[r['canonical_id'],r['anchor']]]) for i,r in enumerate(sel['windows'])];base=json.loads((R/'train64_v1_records.json').read_text());xs={r['index']:r['exposures'] for r in ex};summary={}
 for group,pred in [('seen',lambda n:n>0),('unseen_in_training_draws',lambda n:n==0),('seen_at_least3',lambda n:n>=3)]:
  summary[group]={str(h):aggregate([r for r in base if r['h']==h and pred(xs[r['index']])],['model','copy','linear'],2000) for h in (10,25,50,100)}
 atomic(R/'training_exposure.json',dict(draws=sum(counts.values()),unique_windows=len(counts),pool_windows=len(rows),raw_pool_multiple=sum(counts.values())/len(rows),train64_windows=len(ex),train64_exposure_histogram=dict(collections.Counter(r['exposures'] for r in ex)),windows=ex,matched_checkpoint_sampler_state=True,summary=summary,note='Pure deterministic CPU sampler replay; no model construction, optimizer, backward or parameter update. Main5000x32 draws only; adapter/chain sub-samples not counted as extra main draws.'));atomic(R/'exposure_status.json',dict(status='completed',seconds=time.monotonic()-started,GPUh=0))
 print('completed',len(counts),dict(collections.Counter(r['exposures'] for r in ex)))
except BaseException as e:atomic(R/'exposure_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
