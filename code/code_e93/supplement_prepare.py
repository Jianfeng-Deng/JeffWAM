from common import *
import h5py
from e86.data import build_window,CAMS
from e88.audit import HorizonIndex
D=R/'dev';caps=json.loads((D/'captures.json').read_text());rows=json.loads((D/'windows.json').read_text());missing=json.loads((R/'missing_parent_audit.json').read_text());nodes={n['node']:n for n in map(json.loads,(T/'data/e82_piper_v1/parent_graph/raw_nodes.jsonl').read_text().splitlines())};extra=[]
for m in missing:
 reason='metadata_only_no_recorded_capture';raw=[x[4:] for x in m['members'] if x.startswith('raw:')]
 if raw:
  path=raw[0]
  with h5py.File(path,'r') as f:
   required=('timestamp','camera_timestamps','camera_received_timestamps','state_source_brackets')
   absent=[x for x in required if x not in f]
   if absent:reason='missing_clock_contract:'+','.join(absent)
   else:
    assert not f.attrs.get('sim',False);t=f['timestamp'][:];ct=f['camera_timestamps'][:];rc=f['camera_received_timestamps'][:];br=f['state_source_brackets'][:];n=len(t)
    if n<78:reason='too_short_for_original_anchor24_plus50_window'
    else:
     record=nodes['raw:'+path];cid='piper_'+record['raw_time_state_sha256'][:24];assert cid!='piper_4dc107335389f5711a0f65e0'
     c=dict(canonical_id=cid,parent_component=m['parent'],split='dev',source_path=path,source_collection=record['source_collection']);avail=np.maximum(t,np.max(br,axis=(1,2)));seg=np.cumsum(np.r_[0,(np.diff(t)<=0)|(np.diff(t)>.05)]);valid=np.isfinite(br).all((1,2)) & ((br[:,:,1]-br[:,:,0]).max(1)<=.075);z=dict(timestamp=t,camera_exposure_timestamp=ct,camera_received_timestamp=rc,state_available_time=avail,continuous_segment=seg,original_action_index=np.arange(1,n),p1_source_mask=np.ones(n,bool),p3_source_mask=np.ones(n,bool));ws=[]
     for a in np.unique(np.linspace(24,n-53,16).astype(int)//4*4):
      if a<24 or a+50>=n or seg[a-24]!=seg[a+50] or not valid[a-24:a+51].all():continue
      oi=np.empty((2,3),np.int64);wi=np.empty((3,3),np.int64)
      for k in range(3):
       ac=np.maximum.accumulate(np.maximum(ct[:,k],rc[:,k]));oi[:,k]=np.searchsorted(ac,avail[[a-6,a]],side='right')-1
       for j,off in enumerate((10,25,50)):
        rr=np.arange(max(0,a+off-2),min(n,a+off+3));wi[j,k]=rr[np.argmin(abs(ct[rr,k]-t[a+off]))]
      if any(abs(ct[wi[j,k],k]-t[a+off])>1/60+1e-6 for j,off in enumerate((10,25,50)) for k in range(3)):continue
      try:w=build_window(c,z,int(a),oi,wi,'P1')
      except ValueError:continue
      ws.append(dict(w,_stage='P1',source='piper',stratum='demo6'))
      if len(ws)==8:break
     if ws:
      num=D/(cid+'_numeric.npz');assert not num.exists();np.savez_compressed(num,**z);c['numeric_cache']=str(num);caps.append(c);rows+=ws;ix=HorizonIndex(R/'features',[c]);ss=[set() for _ in range(3)]
      for w in ws:
       for k in range(3):ss[k].update(x[k] for x in w['obs_rows'] if x[k]>=0)
       for h in H:
        ids,why=ix.resolve(w,h,False)
        if ids is not None:
         for k in range(3):ss[k].add(ids[k])
      cams=[dict(camera=cam,path=str(Path(path).parent/'video'/cam/(Path(path).stem+'.mp4')),rows=sorted(ss[k])) for k,cam in enumerate(CAMS)];assert all(Path(x['path']).exists() for x in cams);extra.append(dict(canonical_id=cid,parent_component=m['parent'],split='dev',cameras=cams));reason='recovered_raw_failure_capture_same_clock_and_video_contract'
     else:reason='no_valid_original_quantile_window'
 m['E93_status']=reason
atomic(R/'missing_parent_audit_resolved.json',missing);atomic(D/'captures_full.json',caps);atomic(D/'windows_full.json',rows);frames=sum(len(cam['rows']) for c in extra for cam in c['cameras']);atomic(R/'cache_plan_supplement.json',dict(captures=extra,unique_camera_frames=frames,estimated_bytes=frames*12300+1000000,reason='raw dev source omitted from published capture manifest, frozen parent unchanged'))
print(json.dumps(dict(parents=len({r['parent_component'] for r in rows}),windows=len(rows),supplement=len(extra),frames=frames,reasons=dict(collections.Counter(m['E93_status'] for m in missing)))))
