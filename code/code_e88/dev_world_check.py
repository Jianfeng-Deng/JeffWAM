import json,torch
from e88.common import *
from e88.train import make_data,configure,source_hash
from e88.data import Data
from e86.data import stratum
from e88.model import WorldModel
from e88.resource import Guard
from e88.evaluation import world_evaluate
run=ART/'smoke_03';dest=run/'dev_world_01';dest.mkdir(exist_ok=False);p,r,stats,full,small,lc=make_data();configure();caps=json.loads((C/'artifacts/dev_v1/captures.json').read_text());lookup={c['canonical_id']:stratum(c) for c in caps};rows=json.loads((C/'artifacts/dev_v1/windows_v2.json').read_text());rows=[dict(row,stratum=lookup[row['canonical_id']],_stage='P3') for row in rows];ds=Data(rows,caps,C/'artifacts/dev_features_v1',stats,{'scale':full['P3'].norm},text_cache=lc);guard=Guard('E88_dev_world_existing_cache_only',180)
try:
 m=WorldModel().to('cuda:0');m.load_state_dict(torch.load(run/'P3_long_end.pt',map_location='cpu',weights_only=False)['model']);out=world_evaluate(m,ds,list(range(len(ds.rows))),'cuda:0',guard,'short');atomic(dest/'metrics.json',out);atomic(dest/'READY.json',dict(status='ready',existing_dev_windows=48,existing_dev_parents=6,new_language_labels=0,new_features=0,text='existing human short instruction only',code_sha256=source_hash(),elapsed_s=guard.check()))
finally:guard.finish()
