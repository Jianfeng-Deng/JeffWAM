import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import sys,time,collections,hashlib,numpy as np,torch,cv2
sys.dont_write_bytecode=True
from common import *
from evaluation_data import load_features,motion_dataset,candidate_texts
from encoder import Encoder,preprocess
from contract import sha
cv2.setNumThreads(1);torch.set_num_threads(2)
def main():
 if (L/'READY_training_data.json').exists():return
 selection=json.loads((P/'selection_full_v4.json').read_text());lookup={(s['canonical_id'],s['anchor_source_row']):s for s in selection['samples']};groups=collections.defaultdict(list)
 import pyarrow.parquet as pq
 from evaluation_data import ROBOT_DIRS,AMPLITUDES
 accepted=json.loads((OLDP/'progress.json').read_text())['fk_gt_ready'];assert sha(R/'READY_fk_gt.json')==accepted['sha256']
 cols=['label_id','canonical_id','parent_component','probe_split','anchor_source_row','arm','horizontal_displacement_m','horizontal_direction','amplitude']
 df=pq.read_table(R/'fk/fk_motion_labels.parquet',columns=cols,filters=[('horizon_steps','=',50),('eligible_for_e84','=',True),('probe_split','=','probe-train')]).to_pandas()
 for r in df.to_dict('records'):
  key=(r['canonical_id'],r['anchor_source_row'])
  if key not in lookup or r['horizontal_displacement_m']<.005 or r['horizontal_direction'] not in ROBOT_DIRS or r['amplitude'] not in AMPLITUDES:continue
  if min(lookup[key]['endpoint_camera_rows'][1])<0:continue
  assert lookup[key]['parent_component']==r['parent_component'] and lookup[key]['probe_split']=='probe-train'
  r['label']=ROBOT_DIRS.index(r['horizontal_direction'])*3+AMPLITUDES.index(r['amplitude']);groups[(r['parent_component'],r['arm'])].append(r)
 items=[]
 for (parent,arm),rr in sorted(groups.items()):
  rr=sorted(rr,key=lambda r:hashlib.sha256(('20261112:'+r['label_id']).encode()).digest())[:4]
  for r in rr:
   s=lookup[(r['canonical_id'],r['anchor_source_row'])];ci=1 if arm=='left' else 2
   if min(s['camera_rows'][-1][0],s['camera_rows'][-1][ci],s['endpoint_camera_rows'][1][0],s['endpoint_camera_rows'][1][ci])<0:continue
   items.append(dict(kind='fk',parent=parent,canonical_id=r['canonical_id'],anchor_source_row=r['anchor_source_row'],arm=arm,label=(0 if arm=='left' else 24)+int(r['label']),frames=[(0,s['camera_rows'][-1][0]),(ci,s['camera_rows'][-1][ci]),(0,s['endpoint_camera_rows'][1][0]),(ci,s['endpoint_camera_rows'][1][ci])]))
 bycap=collections.defaultdict(list)
 for s in selection['samples']:
  if s['probe_split']=='probe-train':bycap[s['canonical_id']].append(s)
 used=set()
 for file in sorted(OLDP.glob('e83_*_snapshot_*.jsonl')):
  for line in file.open():
   r=json.loads(line);cid=r['canonical_id']
   if cid not in bycap:continue
   parent=bycap[cid][0]['parent_component'];assert r['parent_id']==parent
   for cap in r.get('keyframe_captions',[]):
    if len([x for x in used if x[0]==parent])>=2:break
    row=cap['source_frame_index'];s=min(bycap[cid],key=lambda s:abs(s['anchor_source_row']-row))
    if abs(s['anchor_source_row']-row)>15 or (parent,s['anchor_source_row']) in used:continue
    if min(s['camera_rows'][-1])<0:continue
    text=cap['caption_en'];used.add((parent,s['anchor_source_row']))
    items.append(dict(kind='e83',parent=parent,canonical_id=cid,anchor_source_row=s['anchor_source_row'],text=text,frames=[(0,s['camera_rows'][-1][0]),(1,s['camera_rows'][-1][1]),(2,s['camera_rows'][-1][2]),(0,s['camera_rows'][-1][0])]))
 assert all(json.loads((R/'split/probe_split.json').read_text())['parents'].get(x['parent'])=='probe-train' for x in items)
 requests=collections.defaultdict(lambda:collections.defaultdict(list))
 for i,r in enumerate(items):
  for slot,(ci,row) in enumerate(r['frames']):
   cam=['cam_high','cam_left_wrist','cam_right_wrist'][ci];route=next(z for z in selection['source_routes'][r['canonical_id']]['cameras'] if z['camera']==cam);requests[route['path']][row].append((i,slot))
 plan_sha=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest();atomic(L/'training_items.json',items)
 resume=json.loads((L/'prepare_progress.json').read_text()) if (L/'prepare_progress.json').exists() else {};same=resume.get('plan_sha256')==plan_sha
 pix=np.lib.format.open_memmap(L/'train_pixels.npy',mode='r+' if same else 'w+',dtype='u1',shape=(len(items),4,3,224,224));sources=resume.get('completed_sources',{}) if same else {};started=time.time()
 def decode(task):
  path,rows=task;st=pathlib.Path(path).stat();before=[st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino]
  if path in sources:
   assert sources[path]==before;return path,before
  c=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);cursor=-1
  for row,slots in sorted(rows.items()):
   if row-cursor>60:
    assert c.set(cv2.CAP_PROP_POS_FRAMES,int(row)),('seek',path,row);cursor=row-1
   while cursor<row:
    ok=c.grab();cursor+=1
    if not ok:raise RuntimeError(('decode',path,row))
   ok,bgr=c.retrieve();assert ok and abs(c.get(cv2.CAP_PROP_POS_FRAMES)-(row+1))<.5,('exact position',path,row)
   v=preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))[0];v=((v+1)*127.5).round().byte().numpy()
   for i,slot in slots:pix[i,slot]=v
  c.release();st=pathlib.Path(path).stat();assert before==[st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino]
  return path,before
 from concurrent.futures import ThreadPoolExecutor,as_completed
 with ThreadPoolExecutor(max_workers=8) as pool:
  jobs=[pool.submit(decode,t) for t in sorted(requests.items())]
  for k,future in enumerate(as_completed(jobs)):
   path,before=future.result();sources[path]=before
   atomic(L/'prepare_progress.json',dict(timestamp_brt=brt(),completed_videos=k+1,total_videos=len(requests),elapsed_s=time.time()-started,plan_sha256=plan_sha,completed_sources=sources))
   if k%50==0:print('PREPARE',k,len(requests),round(time.time()-started),flush=True)
 pix.flush();atomic(L/'training_items.json',items)
 # Teacher globals are reused from frozen cache; only allowed selected samples accessed.
 teacher=np.zeros((len(items),4,768),dtype='f4');byp=collections.defaultdict(list)
 for i,r in enumerate(items):byp[r['canonical_id']].append((i,r))
 for cid,rr in byp.items():
  f=np.load(R/'features'/OLD/'frozen'/(cid+'.npz'));anchors=f['anchor_source_row']
  for i,r in rr:
   ix=int(np.flatnonzero(anchors==r['anchor_source_row'])[0]);ci=1 if r.get('arm')=='left' else 2
   teacher[i]=np.stack([f['global_features'][ix,-1,z] for z in [0,1,2,0]]) if r['kind']=='e83' else np.stack([f['global_features'][ix,-1,0],f['global_features'][ix,-1,ci],f['endpoint_global_features'][ix,1,0],f['endpoint_global_features'][ix,1,ci]])
  f.close()
 np.save(L/'teacher_globals.npy',teacher)
 atomic(L/'READY_training_data.json',dict(timestamp_brt=brt(),items=len(items),fk_pairs=sum(x['kind']=='fk' for x in items),e83_pairs=sum(x['kind']=='e83' for x in items),parents=len({x['parent'] for x in items}),sources=sources,selection_sha256=sha(P/'selection_full_v4.json'),pixel_label_usage=False,source_stat_changes=0,elapsed_s=time.time()-started))
 update(status='training_data_ready');print('READY_TRAIN',len(items),flush=True)
if __name__=='__main__':main()
