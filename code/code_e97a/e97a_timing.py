from e97a_core import *
_,tr,_=datasets();by=collections.defaultdict(list)
for r in tr.rows:by[r['canonical_id']].append(r)
measure=collections.defaultdict(lambda:collections.defaultdict(list))
for cid,rs in by.items():
 c=tr.index.captures[cid];z=np.load(c['numeric_cache']);a=np.array([r['anchor'] for r in rs]);src=c['source_collection'];br=z['state_source_brackets'][a];m=measure[src]
 m['state_wait_ms'].extend(((z['state_available_time'][a]-z['timestamp'][a])*1000).tolist());m['bracket_width_max_ms'].extend(((br[:,:,1]-br[:,:,0]).max(1)*1000).tolist());m['top_image_age_ms'].extend([r['obs_meta'][-1][0][1]*1000 for r in rs]);m['left_image_age_ms'].extend([r['obs_meta'][-1][1][1]*1000 for r in rs]);m['right_image_age_ms'].extend([r['obs_meta'][-1][2][1]*1000 for r in rs])
out={s:{k:dict(n=len(v),min=float(np.min(v)),median=float(np.median(v)),p95=float(np.quantile(v,.95)),max=float(np.max(v))) for k,v in mm.items()} for s,mm in measure.items()};save('timing_train.json',out);print(json.dumps({s:d['state_wait_ms'] for s,d in out.items()}))
