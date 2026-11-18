from common import *
import collections,numpy as np
items=json.loads((L/'training_items.json').read_text());mapping=collections.defaultdict(list);allowed={r['parent'] for r in items}
for file in OLDP.glob('e83_*_snapshot_*.jsonl'):
 for line in file.open():
  r=json.loads(line)
  if r['parent_id'] not in allowed:continue
  for c in r.get('keyframe_captions',[]):mapping[(r['canonical_id'],c['caption_en'])].append(c['source_frame_index'])
rows=[]
for r in items:
 if r['kind']!='e83':continue
 source=min(mapping[(r['canonical_id'],r['text'])],key=lambda n:abs(n-r['anchor_source_row']));rows.append(dict(canonical_id=r['canonical_id'],parent=r['parent'],image_anchor_source_row=r['anchor_source_row'],original_e83_source_frame_index=source,source_delta_frames=r['anchor_source_row']-source,text=r['text']))
obj=dict(timestamp_brt=brt(),pairs=len(rows),exact=sum(r['source_delta_frames']==0 for r in rows),delta_counts=dict(collections.Counter(r['source_delta_frames'] for r in rows)),note='Original E83 records remain unchanged. These are weak nearest-image positives, not retimed robot/FK/pixel ground-truth labels; source indices preserved explicitly here. No action/FK labels shifted.',rows=rows);atomic(P/'e83_training_pair_provenance.json',obj);print({k:v for k,v in obj.items() if k!='rows'})
