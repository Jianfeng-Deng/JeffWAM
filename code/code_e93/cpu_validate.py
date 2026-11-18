from common import *
from e88.audit import HorizonIndex
oldroot=T/'code_e86/artifacts/dev_v1';rows=json.loads((oldroot/'windows_v2.json').read_text());caps=json.loads((oldroot/'captures.json').read_text());ix=HorizonIndex(T/'code_e86/artifacts/dev_features_v1',caps)
old=json.loads((T/'results/e90_p1_20261114_01/dev_records.json').read_text())['world'];new=json.loads((R/'dev_v1_records.json').read_text());lookup={(r['index'],r['h']):r for r in new};pairs=[];counter=0
for i,row in enumerate(rows):
 for h in H:
  ids,why=ix.resolve(dict(row,_stage='P1'),h)
  if ids is None:continue
  a=old[counter];b=lookup[i,h];assert a['h']==h and a['parent']==b['parent'];pairs.append((a,b));counter+=1
assert counter==len(old)
diffs={k:max(abs(a[k]-b[k]) for a,b in pairs) for k in ('model','copy','linear')};assert max(diffs.values())<1e-5,diffs
# New additions create47 extra +100 records among the original48 rows; they are not old v1 records.
atomic(R/'v1_replay.json',dict(max_abs_difference=diffs,records=len(pairs),backend='CPU float32',Euler=4,seed=9001,note='Matched original145 records by original index/horizon availability; expanded +100 records are additional.'))
checks=[]
for c in caps:
 for cam in ('cam_high','cam_left_wrist','cam_right_wrist'):
  oldp=T/'code_e86/artifacts/dev_features_v1'/c['canonical_id'];newp=R/'features'/c['canonical_id'];rr=np.load(oldp/(cam+'_rows.npy'));nr=np.load(newp/(cam+'_rows.npy'));a=np.load(oldp/(cam+'.npy'),mmap_mode='r');b=np.load(newp/(cam+'.npy'),mmap_mode='r');err=float(np.max(abs(a.astype('f4')-b[np.searchsorted(nr,rr)].astype('f4'))));checks.append(err)
assert max(checks)==0,checks
atomic(R/'cpu_validated_status.json',dict(status='completed',v1_max_difference=diffs,new_cache_overlap_max_difference=max(checks),new_records=len(new),original_records=len(old),GPUh=0,brt=brt(),failure_preserved='cpu_status.json post-check initially compared unequal record counts after +100 cache expansion; no inference rerun or changed metrics'))
print(json.dumps(dict(original_records=len(old),differences=diffs,cache_overlap=max(checks))))
