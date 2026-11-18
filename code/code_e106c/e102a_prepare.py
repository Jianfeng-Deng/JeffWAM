import collections,json,numpy as np
from e99w2_common import *
from e99w2_data import load
from e102a_contract import onset_labels,causal_features
from e96a.train import configure
configure();_,_,tr,dev=load();assert not (CONTROL/'GRIPPER_CONTRACT.json').exists()
by=collections.defaultdict(list)
for i,r in enumerate(tr.rows):by[r['canonical_id']].append(i)
qs=[];ds=[]
for cid,ids in by.items():
 z,q,p,g=tr.states.get(tr,cid);mask=np.zeros(len(q),bool)
 for i in ids:
  r=tr.rows[i];a=r['anchor'];mask[max(0,a-24):min(len(q),a+51)]=True
 mask &= g
 qs.append(q[mask][:,[6,13]]);ok=mask[1:]&mask[:-1]&(z['continuous_segment'][1:]==z['continuous_segment'][:-1]);ds.append(np.abs(np.diff(q[:,[6,13]],axis=0))[ok])
q=np.concatenate(qs);d=np.concatenate(ds);threshold=[];noise=[];hist=[]
for k in range(2):
 edges=np.arange(-.01,.08001,.0005);counts,_=np.histogram(q[:,k],edges);smooth=np.convolve(counts,[1,2,3,2,1],mode='same')/9;centers=(edges[1:]+edges[:-1])/2;sel=np.flatnonzero((centers>=.005)&(centers<=.015));minimum=smooth[sel].min();ties=sel[smooth[sel]==minimum];ix=ties[len(ties)//2];threshold.append(float(centers[ix]));tiny=d[:,k][d[:,k]<=.0001];noise.append(float(max(1e-6,np.quantile(tiny,.99))));hist.append(dict(edges=edges.tolist(),counts=counts.tolist(),smooth=smooth.tolist()))
c=dict(fitted_split='train',parents=145,threshold_m=threshold,noise_m=noise,closed_m=[float(np.median(q[q[:,k]<=threshold[k],k])) for k in range(2)],open_m=[float(np.median(q[q[:,k]>threshold[k],k])) for k in range(2)],histogram=hist,threshold_method='0.5mm bins; triangular [1,2,3,2,1]/9 smoothing; minimum in [5,15]mm; middle tied bin',noise_method='99th percentile of absolute adjacent increments <=0.1mm; floor 0.001mm',onset_method='backtrack from crossing <=120 frames, same segment and valid; stop at opposite >noise or 3 consecutive quiet; first directional >noise increment; no crossing before preceding event',input_states='observed threshold states only; onset labels never input',loss='BCE weight1, no switch reweighting',command_truth='motion-onset proxy from feedback, not measured sent command; disclosed limitation')
# Train-only standardization and onset lag summary; fixed frames deduplicated above.
features=[];lags=[[],[]];event_counts=[0,0]
for cid,ids in by.items():
 z,q,p,g=tr.states.get(tr,cid);lab,ev=onset_labels(q,z['continuous_segment'],g,c);mask=np.zeros(len(q),bool)
 for i in ids:
  r=tr.rows[i];a=r['anchor'];mask[a:a+51]=True
 for e in ev:
  if mask[e['crossing']]:lags[e['gripper']].append(e['lag']);event_counts[e['gripper']]+=int(mask[e['crossing']])
 for i in ids[::max(1,len(ids)//32)]:
  r=tr.rows[i];features.append(causal_features(q,p,r['anchor'],r['history_valid'],z['continuous_segment'],g,c))
x=np.stack(features);c.update(history_mean=x.mean(0).tolist(),history_std=np.maximum(x.std(0),.01).tolist(),history_fit_windows=len(x),onset_lag_frames={str(k):dict(n=len(lags[k]),quantiles=np.quantile(lags[k],[0,.25,.5,.75,.9,.95,1]).tolist(),values=lags[k]) for k in range(2)})
atomic(CONTROL/'GRIPPER_CONTRACT.json',c)
# Strict reuse of E101-A fixed sample identities.
f=json.loads((T/'results/e101a_diag_20261115_01/fixed_windows.json').read_text());atomic(CONTROL/'fixed_windows.json',f)
report('## 训练parent夹爪合同（训练前锁定）\n'+json.dumps({k:v for k,v in c.items() if k not in ('histogram','history_mean','history_std')},ensure_ascii=False))
print(json.dumps({k:c[k] for k in ('threshold_m','noise_m','closed_m','open_m','history_fit_windows')},ensure_ascii=False))
print('fixed window keys',list(f))
