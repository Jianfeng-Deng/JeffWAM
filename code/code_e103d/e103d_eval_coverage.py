import sys,json,collections,traceback
sys.dont_write_bytecode=True
from e103d_runtime import *
try:
 from e99w2_data import load
 from e99w2_complete_h100 import UnionImage,EXTRA
 from e88.audit import HorizonIndex
 from e90.data import Data
 from e96a.train import configure
 configure();_,_,tr,dev=load();dev.base.__class__=UnionImage;dev.index=HorizonIndex(EXTRA,list(dev.index.captures.values()));counts={}
 for h in [10,25,50,100]:
  for i in range(len(dev.rows)):assert dev.batch([i],(h,),'short')['world_valid'].all()
  counts[h]=dict(windows=len(dev.rows),parents=len({r['parent_component'] for r in dev.rows}))
 er=T/'results/e93_dev_diag_20261114_01';caps=json.loads((er/'dev/captures_full.json').read_text());rows=json.loads((er/'dev/windows_full.json').read_text());d=Data(rows,caps,er/'features',tr.base.stats,{'scale':tr.norm},{'__human_default__':dev.base.text_cache['__human_default__']});expanded={}
 for h in [10,25,50,100]:
  valid=[i for i in range(len(rows)) if bool(d.batch([i],(h,),'short','P1')['world_valid'].all())];expanded[h]=dict(windows=len(valid),parents=len({rows[i]['parent_component'] for i in valid}))
 out=dict(status='passed',fixed=counts,expanded=expanded,no_model_forward=True,brt=brt());atomic(CONTROL/'EVAL_COVERAGE.json',out);report('## 评估覆盖 CPU 预检（无模型前向）\n'+json.dumps(out,ensure_ascii=False));print(out,flush=True)
except BaseException:atomic(CONTROL/'EVAL_COVERAGE_ERROR.json',dict(error=traceback.format_exc(),brt=brt()));raise
