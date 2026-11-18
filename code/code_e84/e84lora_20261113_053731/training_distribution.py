from common import *
import collections
items=json.loads((L/'training_items.json').read_text());sel=json.loads((P/'selection_full_v4.json').read_text());coh={s['canonical_id']:s['cohort'] for s in sel['samples']};by=collections.defaultdict(list)
for r in items:by[(r['kind'],coh[r['canonical_id']])].append(r)
summary={kind+'/'+cohort:dict(pairs=len(rr),parents=len({r['parent'] for r in rr})) for (kind,cohort),rr in by.items()};counts=[0]*48
for r in items:
 if r['kind']=='fk':counts[r['label']]+=1
obj=dict(timestamp_brt=brt(),distribution=summary,fk48class_counts=counts,missing_fk48classes=[i for i,n in enumerate(counts) if not n],e83_unique_full_captions=len({r['text'] for r in items if r['kind']=='e83'}),e83_text_parent_mapping=[dict(text=r['text'],parent=r['parent'],canonical_id=r['canonical_id'],anchor_source_row=r['anchor_source_row']) for r in items if r['kind']=='e83']);atomic(P/'training_distribution.json',obj);print({k:v for k,v in obj.items() if k!='e83_text_parent_mapping'})
