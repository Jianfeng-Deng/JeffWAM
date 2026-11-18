import os,sys,json,time,collections,hashlib,traceback,datetime
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';R=T/'notes/e109g_gripper_generalization_20261116.md';sys.path.insert(0,str(T/'code_e108g'));import collect as old;old.common.CONTROL=T/'results/e106c_control_20261115_02';import analyze as ev
import numpy as np,torch,h5py
from e90.sampling import MultiSourceSampler
from e86.common import sha

def put(n,x):
 p=O/n;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));os.replace(tmp,p)
def quant(a):return np.quantile(a,[0,.25,.5,.75,1]).tolist() if len(a) else None

def main():
 start=time.monotonic();old.configure();p,ready,tr,dev=old.datasets();c=old.contract();ar=json.loads((O/'ALIGNED_WINDOW_LOCK.json').read_text())['rows'];oldr=json.loads((T/'results/e108g_diag_20261116_01/predictions.json').read_text());p1rows=[dict(r,source='piper') for r in json.loads(Path(ready['windows']['P1']).read_text())];s3=MultiSourceSampler(tr.rows,seed=90,weights={'piper':1.},caps={},demo3_cap=.45);s1=MultiSourceSampler(p1rows,seed=10390,weights={'piper':1.},caps={},demo3_cap=.45);n3=np.zeros(len(tr.rows),dtype='i8');n1=np.zeros(len(p1rows),dtype='i8');a2=np.zeros_like(n3);hash3=hashlib.sha256();hash1=hashlib.sha256();prefix={}
 for step in range(1,20001):
  ids=s3.sample(32);ii=s1.sample(32);np.add.at(n3,ids,1);np.add.at(n1,ii,1)
  if step<=2000:np.add.at(a2,ids,1)
  hash3.update(np.asarray(ids,dtype='<i8').tobytes());hash1.update(np.asarray(ii,dtype='<i8').tobytes())
  if step in (2000,10000,15000,20000):prefix[str(step)]=dict(P3=hash3.hexdigest(),P1=hash1.hexdigest(),draws=step*32)
  if step==10000:
   expected=json.loads((T/'results/e104p_f2_s90_20261115_03/summary_10000.json').read_text());assert prefix[str(step)]['P3']==expected['P3_samples_sha256'] and prefix[str(step)]['P1']==expected['P1_samples_sha256'];hash3=hashlib.sha256();hash1=hashlib.sha256()
  if step in (15000,20000):
   expected=json.loads((T/f'results/e106c_f2_s90_20261115_02/summary_{step:05d}.json').read_text());assert hash3.hexdigest()==expected['P3_samples_sha256'] and hash1.hexdigest()==expected['P1_samples_sha256']
  if step%2000==0:put('metadata_status.json',dict(stage='exact_sampler_replay',step=step,elapsed_s=time.monotonic()-start));print('replay',step,round(time.monotonic()-start,1),flush=True)
 expected=json.loads((T/'results/e102a_a2_s90_20261115_03/train_summary.json').read_text());assert prefix['2000']['P3']==expected['sample_sha256']
 np.savez_compressed(O/'sampling_counts.npz',p3=n3,p1=n1,a2=a2)
 by=collections.defaultdict(lambda:dict(P3_20k=0,P3_A2_2k=0,P1_20k=0,P3_windows=0,P1_windows=0))
 for rows,counts,prefixkey,win in [(tr.rows,n3,'P3_20k','P3_windows'),(p1rows,n1,'P1_20k','P1_windows')]:
  for i,r in enumerate(rows):v=by[r['parent_component']];v[prefixkey]+=int(counts[i]);v[win]+=1
 for i,r in enumerate(tr.rows):by[r['parent_component']]['P3_A2_2k']+=int(a2[i])
 for v in by.values():v['direct_grip_draws']=v['P3_20k']+v['P3_A2_2k'];v['draws_per_P3_window']=v['direct_grip_draws']/v['P3_windows'] if v['P3_windows'] else None
 put('EXPOSURE.json',dict(parents=by,prefix_hashes=prefix,exact_hashes_match=True,P3_draws=640000,A2_draws=64000,P1_draws=640000,P1_direct_grip_supervision=False,P1_parent_count=len({r['parent_component'] for r in p1rows})))
 # Outcome/stage/model labels deliberately not read. Source timing and explicit physical/source metadata only.
 full={};metadata=[];attrs_seen=collections.Counter()
 for split,data,rows in [('train_aligned',tr,ar),('dev',dev,dev.rows)]:
  for cid in dict.fromkeys(r['canonical_id'] for r in rows):
   cap=data.index.captures[cid];z,q,_,good=data.states.get(data,cid);lab=old.onset_labels(q,z['continuous_segment'],good,c)[0];seg=z['continuous_segment'];dt=np.diff(z['timestamp']);validpair=good[:-1]&good[1:]&(seg[:-1]==seg[1:])&(dt>0)&(dt<=.050001);es=[];intervals=[];amp=[]
   for k in range(2):
    times=np.flatnonzero((lab[1:,k]!=lab[:-1,k])&validpair)+1
    for t in times:es.append([int(t),k,int(lab[t,k])])
    for a,b in zip(times[:-1],times[1:]):
     if seg[a]==seg[b]:intervals.append((z['timestamp'][b]-z['timestamp'][a]))
    for t in times:
     end=min(int(t)+30,len(q)-1)
     if seg[t-1]==seg[end] and good[t-1:end+1].all():amp.append([k,float(abs(q[end,[6,13][k]]-q[t-1,[6,13][k]])*1000)])
   dates={};operator={};scene={};explicit_phase={}
   src=cap.get('source_path')
   if src and Path(src).is_file():
    with h5py.File(src,'r') as f:
     for key in f.attrs.keys():
      attrs_seen[key]+=1
      kl=key.lower()
      if any(s in kl for s in ('operator','demonstrator','scene','environment','stage','phase')):
       val=f.attrs[key];val=val.decode(errors='replace') if isinstance(val,bytes) else str(val)
       if any(s in kl for s in ('operator','demonstrator')):operator[key]=val
       if any(s in kl for s in ('scene','environment')):scene[key]=val
       # Only disclose field availability, do not use stage contents for stratification.
       if any(s in kl for s in ('stage','phase')):explicit_phase[key]='present_unused'
   first=float(z['timestamp'][0]);dates['UTC']=datetime.datetime.fromtimestamp(first,datetime.timezone.utc).strftime('%Y-%m-%d');dates['BRT']=datetime.datetime.fromtimestamp(first,datetime.timezone(datetime.timedelta(hours=-3))).strftime('%Y-%m-%d');par=cap['parent_component'];hw={cam:obj.get('sdk_serial') for cam,obj in cap.get('camera',{}).items()};meta=dict(split=split,parent=par,cid=cid,source_collection=cap.get('source_collection'),session_id=cap.get('session_id'),recording_date=dates,operator_explicit=operator,scene_explicit=scene,phase_field_availability=explicit_phase,camera_sdk_serial=hw,source=src,source_owner_is_not_operator=cap.get('owner'))
   metadata.append(meta);full[cid]=dict(parent=par,split=split,n_frames=len(q),valid_minutes=float(dt[validpair].sum()/60),events=es,event_count=len(es),event_frequency_per_min=float(len(es)/(dt[validpair].sum()/60)),interval_s=intervals,amplitude_30frame_mm=amp,opening_p05_p50_p95_mm=np.quantile(q[good][:,[6,13]]*1000,[.05,.5,.95],axis=0).tolist(),metadata=meta)
  print('full_stats',split,len(metadata),round(time.monotonic()-start,1),flush=True)
 # Old16 may use different canonicals, retain full capture stats for coverage independently.
 for r in oldr:
  if r['split']!='train' or r['cid'] in full:continue
  cid=r['cid'];z,q,_,good=tr.states.get(tr,cid);lab=old.onset_labels(q,z['continuous_segment'],good,c)[0];seg=z['continuous_segment'];dt=np.diff(z['timestamp']);vp=good[:-1]&good[1:]&(seg[:-1]==seg[1:])&(dt>0)&(dt<=.050001);es=[[int(t),k,int(lab[t,k])] for k in range(2) for t in np.flatnonzero((lab[1:,k]!=lab[:-1,k])&vp)+1];full[cid]=dict(parent=r['parent'],split='train16_extra',n_frames=len(q),valid_minutes=float(dt[vp].sum()/60),events=es,event_count=len(es))
 put('FULL_CAPTURE_STATS.json',full);put('PARENT_METADATA.json',dict(records=metadata,raw_attribute_keys=attrs_seen))
 coverage={}
 for suite,rows in [('train16',[r for r in oldr if r['split']=='train']),('dev',[r for r in oldr if r['split']=='dev']),('train_aligned',ar)]:
  pp=collections.defaultdict(list)
  for r in rows:pp[r.get('parent',r.get('parent_component'))].append(r)
  result={}
  for par,rr in sorted(pp.items()):
   cover=collections.defaultdict(set);win_events=[];inits=[];anchors=[];history=[]
   for r in rr:
    cid=r.get('cid',r.get('canonical_id'));a=r['anchor'];cover[cid].update(range(a+1,a+51));win_events.extend((cid,t,k,d) for t,k,d in full[cid]['events'] if a<t<=a+50);anchors.append(a/(full[cid]['n_frames']-1));
    if 'current' in r:inits.append(int(np.any(np.array(r['current'])!=np.array(r['label_init']))))
    if 'history_valid' in r:history.append(sum(r['history_valid']))
   cids=list(cover);total=sum(full[cid]['event_count'] for cid in cids);covered_unique=set(win_events);intervals=[]
   for cid in cids:
    for k in range(2):
     ts=sorted(t for cc,t,kk,d in covered_unique if cc==cid and kk==k);intervals.extend(np.diff(ts).astype(float).tolist())
   result[par]=dict(windows=len(rr),window_events=len(win_events),unique_covered_events=len(covered_unique),full_capture_events=total,full_capture_minutes=sum(full[cid]['valid_minutes'] for cid in cids),unique_frames=sum(len(x) for x in cover.values()),full_frames=sum(full[cid]['n_frames'] for cid in cids),target_frame_coverage=sum(len(x) for x in cover.values())/sum(full[cid]['n_frames'] for cid in cids),event_coverage=len(covered_unique)/total if total else None,overlap_factor=len(rr)*50/sum(len(x) for x in cover.values()),event_intervals_frames=intervals,anchor_fraction=anchors,initial_mismatch_windows=sum(inits) if inits else None,history_valid=history)
  coverage[suite]=result
 put('COVERAGE.json',coverage);put('METADATA_COMPLETE.json',dict(status='complete',elapsed_s=time.monotonic()-start,GPUh=0));print('metadata complete',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:put('METADATA_FAILED.json',dict(traceback=traceback.format_exc()));raise
