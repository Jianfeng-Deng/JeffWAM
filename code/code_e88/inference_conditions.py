"""Matched fixed-stage checkpoint input-only language ablation; no optimization."""
import torch,json
from e88.common import *
from e88.train import make_data,configure,eval_ids,source_hash
from e88.model import WorldModel
from e88.resource import Guard
from e88.evaluation import world_evaluate
run=ART/'smoke_03';dest=run/'conditional_inference_01';dest.mkdir(exist_ok=False);p,r,s,full,small,lc=make_data();configure();guard=Guard('E88_fixed_checkpoint_text_only_inference',180)
try:
 m=WorldModel().to('cuda:0');path=run/'P3_long_end.pt';m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);ids=eval_ids(full['P3']);out={}
 for condition in ('none','short','long'):
  out[condition]=world_evaluate(m,full['P3'],ids,'cuda:0',guard,condition);atomic(dest/'results.json',out)
 # Existing dev cache only; no model narration/labels, no newly extracted frames.
 from e88.data import Data
 from e86.data import stratum
 caps=json.loads((C/'artifacts/dev_v1/captures.json').read_text());lookup={c['canonical_id']:stratum(c) for c in caps};devrows=json.loads((C/'artifacts/dev_v1/windows_v2.json').read_text());devrows=[dict(row,stratum=lookup[row['canonical_id']],_stage='P3') for row in devrows]
 dev=Data(devrows,caps,C/'artifacts/dev_features_v1',s,{'scale':full['P3'].norm},text_cache=lc)
 atomic(dest/'dev_world_short.json',world_evaluate(m,dev,list(range(len(devrows))),'cuda:0',guard,'short'))
 atomic(dest/'READY.json',dict(status='ready',fixed_checkpoint=str(path),checkpoint_sha256=sha(path),code_sha256=source_hash(),same_noise_seed=8801,world_only=True,source='same16fixedtrainingwindows, no language eval truth',training=False,checkpoint_choice='fixed P3_long stage end by contract, no score selection',elapsed_s=guard.check()))
finally:guard.finish()
