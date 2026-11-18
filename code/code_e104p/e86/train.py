"""Default E86 launcher: immutable V3 engineering smoke only."""
import argparse,json
from pathlib import Path
from .v3_train import smoke,DEFAULT_INPUT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run_id');p.add_argument('--cache');p.add_argument('--inputs',default=str(DEFAULT_INPUT));p.add_argument('--clean-aux',type=float,default=.25);p.add_argument('--mode-probabilities',nargs=4,type=float,default=[.3,.3,.3,.1]);p.add_argument('--sampler',choices=['parent_equal','stratified'],default='parent_equal');p.add_argument('--demo3-cap',type=float,default=.45);p.add_argument('--stratum-weights',default='{}');a=p.parse_args()
 if a.cache:
  ready=json.loads((Path(a.inputs)/'READY_V3_cache_windows.json').read_text())
  if Path(a.cache).resolve()!=Path(ready['cache']).resolve():raise ValueError('Cache differs from pinned V3 bundle')
 smoke(a.run_id,Path(a.inputs),a.clean_aux,tuple(a.mode_probabilities),a.sampler,a.demo3_cap,json.loads(a.stratum_weights))
