import os,sys,json,hashlib,collections,struct,datetime
from pathlib import Path
import numpy as np
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu'); O=T/'data/e103d_cache'; C=T/'results/e103d_control_20261115_01'
sys.path.append(str(T/'data/abc_inventory_v1/env/venv/lib/python3.12/site-packages'))
from mcap.reader import make_reader
CAM=['/top-camera','/left-wrist-camera','/right-wrist-camera']; OFF=np.array([-24,-18,-12,-6,0,10,25,50,100])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def index(path):
 with open(path,'rb') as f:
  s=make_reader(f).get_summary(); ids={k:v.topic for k,v in s.channels.items() if v.topic in CAM};ts={c:[] for c in CAM}
  for ch in s.chunk_indexes:
   for cid,offset in ch.message_index_offsets.items():
    if cid not in ids:continue
    f.seek(offset);op,n=struct.unpack('<BQ',f.read(9));assert op==7;data=f.read(n);cc,ln=struct.unpack_from('<HI',data);assert cc==cid and ln%16==0
    ts[ids[cid]].extend(t for t,o in struct.iter_unpack('<QQ',data[6:6+ln]))
 return [np.array(sorted(ts[c]),dtype='i8') for c in CAM]
def windows(ts):
 start=max(a[0] for a in ts);end=min(a[-1] for a in ts);n=int((end-start)*30//10**9)+1
 anchors=np.arange(24,n-100,30);times=start+np.rint((anchors[:,None]+OFF[None])*1e9/30).astype('i8');valid=np.ones(len(anchors),bool);sel=[];why=collections.Counter();metrics=[];gaps=[]
 for a in ts:
  dt=np.diff(a);fps=(len(a)-1)*1e9/(a[-1]-a[0]);metrics.append(dict(frames=len(a),fps=float(fps),dt50_ms=float(np.median(dt)/1e6),max_gap_ms=float(dt.max()/1e6),gap_fraction=float(np.mean(dt>1e8))))
  if fps<20 or fps>65 or dt.max()>1e9 or np.mean(dt>1e8)>.005:return [],[],dict(rejected='episode_fps_or_gap',streams=metrics)
  prev=np.searchsorted(a,times,side='right')-1;ix=np.clip(prev,0,len(a)-1);nxt=np.clip(prev+1,0,len(a)-1);near=np.where(abs(a[ix]-times)<=abs(a[nxt]-times),ix,nxt);j=np.concatenate([ix[:,:5],near[:,5:]],1);age=times[:,:5]-a[j[:,:5]];err=abs(times[:,5:]-a[j[:,5:]]);tol=min(50e6,max(.1e6,np.median(dt)/2+.01e6));ok=(age.min(1)>=0)&(age.max(1)<=1e8)&(err.max(1)<=tol);why['age_or_target']+=int((~ok).sum());valid &= ok;sel.append(j)
  g=np.where(dt>1e8)[0]
  for k in g:
   ok=(times[:,-1]<a[k])|(times[:,0]>a[k+1]);why['gap_overlap']+=int((~ok).sum());valid &= ok
 sel=np.stack(sel,2)
 exposure=np.stack([a[sel[:,:,k]] for k,a in enumerate(ts)],2);skew=np.ptp(exposure,axis=2).max(1);why['skew']+=int((skew>5e7).sum());valid &= skew<=5e7
 out=[]
 for anc,tm,j,ex in zip(anchors[valid],times[valid],sel[valid],exposure[valid]):
  out.append(dict(anchor=int(anc),time_ns=int(tm[4]),obs_rows=j[:5].tolist(),target_rows={str(h):j[q+5].tolist() for q,h in enumerate(OFF[5:])},obs_meta=[np.stack([(ex[q]-tm[q])/1e9,np.zeros(3)],1).tolist() for q in range(5)]))
 return out,[np.unique(sel[valid,:,k]).tolist() for k in range(3)],dict(candidate_anchors=len(anchors),valid=len(out),reasons=dict(why),streams=metrics,span_s=float((end-start)/1e9),max_age_ms=float((times[valid,:5,None]-exposure[valid,:5]).max()/1e6) if valid.any() else None,max_skew_ms=float(skew[valid].max()/1e6) if valid.any() else None)
def main():
 s1=T/'data/abc_inventory_v1/proposed_subsets/S1.json';yp=T/'data/yam_inventory_v1/proposed_p1_yam_list.json';ep=T/'data/e92_yam_lang_v1/exclusions_success_subpool_20261114.json';abc=json.loads(s1.read_text())['episodes'];yam=json.loads(yp.read_text())['entries'];exclude=set(json.loads(ep.read_text())['clip_ids']);pool=collections.defaultdict(list)
 for r in abc:
  if r['category']=='A':pool[r['task']].append(r)
 priority=['zip_tie_the_wires','untangle_the_cables','wrap_the_cables','wrap_the_headphones','connect_and_route_the_hose','tie_the_bouquet_of_fake_flowers']
 selected=[];secs=0;bytask={}
 for task in priority:
  acc=0;rows=sorted(pool[task],key=lambda r:(not r['has_annotation'],hashlib.sha256(('e103d90'+r['canonical_id']).encode()).hexdigest()))
  for r in rows:
   d=r['duration_s']
   if acc+d>2.5*3600 or secs+d>10*3600:continue
   selected.append(r);secs+=d;acc+=d
   if acc>=min(2.35*3600,10*3600-secs+acc):break
  bytask[task]=acc/3600
  if secs>=9.95*3600:break
 # Use remaining eligible tasks if necessary, respecting a hard 2.5h/task cap.
 if secs<9.9*3600:
  for task in sorted(set(pool)-set(priority)):
   acc=0
   for r in pool[task]:
    d=r['duration_s']
    if acc+d>2.5*3600 or secs+d>10*3600:continue
    selected.append(r);secs+=d;acc+=d
   bytask[task]=acc/3600
   if secs>=9.9*3600:break
 records=[];audit=[];payload=0
 for source,rr in [('abc',selected),('yam',[r for r in yam if r['canonical_id'] not in exclude])]:
  for i,r in enumerate(rr):
   if source=='abc':ts=index(r['path'])
   else:ts=[np.rint(np.load(Path(r['representative_path'])/(c+'-timestamp.npy'))*1e6).astype('i8') for c in ['top','left','right']]
   w,rows,a=windows(ts);a.update(source=source,canonical_id=r['canonical_id']);audit.append(a)
   if w:
    rec=dict(source=source,canonical_id=r['canonical_id'],task=r.get('task',r.get('task_family')),original=r,windows=w,rows=rows,timestamps=[x.tolist() for x in ts]);records.append(rec);payload+=sum(len(x) for x in rows)*12288
   if i%20==0:dump(C/'plan_status.json',dict(source=source,completed=i+1,total=len(rr),payload_bytes=payload));print(source,i+1,len(rr),payload,flush=True)
 summary=dict(abc_selected_episodes=len(selected),abc_selected_hours=secs/3600,hours_by_task=bytask,yam_allowlist=49,yam_excluded_intersection=sorted(set(r['canonical_id'] for r in yam)&exclude),qualified_episodes=dict(collections.Counter(r['source'] for r in records)),windows={s:sum(len(r['windows']) for r in records if r['source']==s) for s in ['abc','yam']},camera_frames=sum(sum(len(x) for x in r['rows']) for r in records),payload_bytes=payload,estimated_cache_gpuh_218fps_margin25= sum(sum(len(x) for x in r['rows']) for r in records)/218/3600*1.25,input_hashes={str(p):sha(p) for p in [s1,yp,ep]},time_contract='30Hz physical offsets; 1s anchor stride; causal O age<=100ms; 3cam skew<=50ms; native nearest W tolerance median/2<=50ms; no gap>100ms across [-24,+100]; full episode fps/gap gates; ABC retains E89 S1 six-endpoint screening; YAM raw ms receipt approximations, no shifts')
 assert payload<35e9
 dump(O/'plan.json',dict(summary=summary,records=records,audit=audit));dump(C/'plan_summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
