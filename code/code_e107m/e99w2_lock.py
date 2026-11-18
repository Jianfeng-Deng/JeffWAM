import sys
sys.dont_write_bytecode=True
from e99w2_common import *
import json
files=['e99w2_common.py','e99w2_train.py','e99w2_driver04.py','e99w2_data.py','e99w2_model.py','e99w2_objective.py','e96a/common.py','e96a/resource.py']
lock=dict(brt=brt(),training_sources={s:sha(ROOT/s) for s in files},initialization_sha256=sha(INIT),condition_extension='H51..100 terminal hold',flow=dict(steps=4,seed=90,N=8),noise='per step/dimension RMS on first80 unique train parents last16; Gaussian independent seed990000+dev_index; no dev scale fitting',frozen_eval_coverage_sha256=sha(ART/'preflight_dev_coverage.json'),before_dev_model_forward=True)
with (ART/'IMPLEMENTATION_LOCK.json').open('x') as f:json.dump(lock,f,indent=2)
# Only metadata/hashes for read-only files; no writes outside E99-W2 outputs.
protected=[]
for folder in ('code_e82','code_e86','code_e88','code_e96a','code_e98w'):
 for p in sorted((T/folder).rglob('*.py')):
  if '__pycache__' not in str(p):protected.append(dict(path=str(p),sha256=sha(p)))
for p in [INIT,T/'data/e82_piper_v1/P3_ready_V3.json']:protected.append(dict(path=str(p),sha256=sha(p)))
atomic(ART/'protected_sources_lock.json',protected)
print('locked',len(files),'own training sources;',len(protected),'read-only protected files')
