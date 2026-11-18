import json,collections
from e88.train import make_data
from e88.audit import causal_segments
from e88.common import ART,atomic
p,r,s,full,small,lc=make_data();out={}
for stage,ds in full.items():
 c=collections.defaultdict(collections.Counter)
 for row in ds.rows:
  st=row['stratum'];cid=row['canonical_id'];a=row['anchor'];inst=(cid,a) in lc;narr=bool(causal_segments(ds.labels.get(cid,[]),a))
  for key,v in [('windows',1),('short_human_instruction',1),('long_instruction',inst),('causal_narration',narr),('any_long_condition',inst or narr)]:c[st][key]+=int(v)
 out[stage]={st:dict(v) for st,v in c.items()}
atomic(ART/'language_coverage.json',out);print(json.dumps(out))
