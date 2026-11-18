"""Small preselected rev4 dev subset, real recorded feedback only, no model language."""
import numpy as np,json,h5py
from pathlib import Path
from .common import *
from .data import Package,build_window,CAMS

def main():
    dest=C/'artifacts/dev_v1';dest.mkdir(exist_ok=False);p=Package();caps=[];windows=[];selected=[];seen=set();exclusions=[]
    # Deterministic canonical-id order, up to six parents, eight windows/parent. Test stays unopened.
    for c in p.captures:
        if c['split']!='dev' or c['parent_component'] in seen:continue
        route=p.routes.get(c['canonical_id'],{})
        if not route.get('all_available'):continue
        before=Path(c['source_path']).stat()
        try:
            with h5py.File(c['source_path'],'r') as f:
                if f.attrs.get('sim',False):raise ValueError('Simulation')
                q=f['observations/qpos'][:].astype('f4');qr=f['observations/qpos_raw'][:].astype('f4');t=f['timestamp'][:];ct=f['camera_timestamps'][:];rc=f['camera_received_timestamps'][:];br=f['state_source_brackets'][:]
                if not np.array_equal(f['action'][:],q):raise ValueError('Unknown raw action/feedback contract')
            assert before.st_mtime_ns==Path(c['source_path']).stat().st_mtime_ns
            n=len(t);avail=np.maximum(t,np.max(br,axis=(1,2)));bad=(np.diff(t)<=0)|(np.diff(t)>.05);seg=np.cumsum(np.r_[0,bad]);valid_state=np.isfinite(br).all((1,2)) & ((br[:,:,1]-br[:,:,0]).max(1)<=.075)
            action=q[1:].copy();action[:,[6,13]]=np.where(qr[1:,[6,13]]<0,qr[1:,[6,13]],action[:,[6,13]])
            z=dict(q=q,qpos_raw=qr,action=action,timestamp=t,camera_exposure_timestamp=ct,camera_received_timestamp=rc,state_available_time=avail,continuous_segment=seg,original_action_index=np.arange(1,n),p1_source_mask=np.ones(n,bool),p3_source_mask=np.ones(n,bool))
            # Candidate anchors are fixed quantiles, then first eight valid; no model outcomes consulted.
            anchors=np.unique(np.linspace(24,n-53,16).astype(int)//4*4);ws=[]
            for a in anchors:
                if a<24 or seg[a-24]!=seg[a+50] or not valid_state[a-24:a+51].all():continue
                oi=np.empty((2,3),np.int64);wi=np.empty((3,3),np.int64)
                for k in range(3):
                    ac=np.maximum.accumulate(np.maximum(ct[:,k],rc[:,k]));oi[:,k]=np.searchsorted(ac,avail[[a-6,a]],side='right')-1
                    for j,off in enumerate((10,25,50)):
                        rr=np.arange(max(0,a+off-2),min(n,a+off+3));idx=rr[np.argmin(abs(ct[rr,k]-t[a+off]))];wi[j,k]=idx
                if any(abs(ct[wi[j,k],k]-t[a+off])>1/60+1e-6 for j,off in enumerate((10,25,50)) for k in range(3)):continue
                try:w=build_window(c,z,int(a),oi,wi,'P3')
                except ValueError:continue
                ws.append(w)
                if len(ws)>=8:break
            if not ws:continue
            num=dest/(c['canonical_id']+'_numeric.npz');np.savez_compressed(num,**z);c=dict(c,numeric_cache=str(num));caps.append(c);windows.extend(ws);seen.add(c['parent_component'])
            cams=[]
            for k,cam in enumerate(CAMS):
                ids=sorted(set(w[group][j][k] for w in ws for group in ('obs_rows','future_rows') for j in range(len(w[group])) if w[group][j][k]>=0));r=next(x for x in route['cameras'] if x['camera']==cam);cams.append(dict(camera=cam,path=r['path'],rows=ids))
            selected.append(dict(canonical_id=c['canonical_id'],parent_component=c['parent_component'],split='dev',cameras=cams))
        except (KeyError,OSError,ValueError) as e:exclusions.append(dict(canonical_id=c['canonical_id'],reason=str(e)))
        if len(seen)>=6:break
    atomic(dest/'captures.json',caps);atomic(dest/'windows.json',windows)
    frames=sum(len(x['rows']) for c in selected for x in c['cameras']);atomic(dest/'cache_plan.json',dict(schema='e86-cache-plan-v1',refs=p.refs,captures=selected,unique_camera_frames=frames,estimated_bytes=frames*8*768*2+1_000_000))
    atomic(dest/'READY_dev.json',dict(status='ready_real_feedback_only',parents=len(seen),windows=len(windows),camera_frames=frames,exclusions=exclusions,selection='canonical order, fixed anchor quantiles, no performance selection',language_labels=0,files=[dict(path=str(x),sha256=sha(x)) for x in sorted(dest.glob('*.json'))]));print(json.dumps(dict(parents=len(seen),windows=len(windows),frames=frames,exclusions=len(exclusions))),flush=True)
if __name__=='__main__':main()
