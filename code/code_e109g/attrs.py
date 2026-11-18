import json,collections,datetime
from pathlib import Path
import h5py
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';d=json.loads((O/'PARENT_METADATA.json').read_text());keys=collections.Counter();missing=[];relocated=[]
for r in d['records']:
 src=r['source'];path=Path(src) if src else None
 if path and not path.exists():path=Path(src.replace('/mnt/workspace/colleague-a/Datasets/piper/','/mnt/workspace/dataset/piper/',1))
 r['source_resolved']=str(path) if path else None
 if path and path.is_file():
  stat=path.stat();r['operator_explicit']={};r['scene_explicit']={};r['phase_field_availability']={}
  with h5py.File(path,'r') as f:
   # Ensure same recording: source numeric first timestamp and length checked from own cache metadata.
   cid=r['cid'];full=json.loads((O/'FULL_CAPTURE_STATS.json').read_text())[cid];assert len(f['timestamp'])==full['n_frames'];r['raw_identity_check']='timestamp length plus date UTC';date=datetime.datetime.fromtimestamp(float(f['timestamp'][0]),datetime.timezone.utc).strftime('%Y-%m-%d');assert date==r['recording_date']['UTC']
   for k in f.attrs.keys():
    keys[k]+=1;kl=k.lower()
    if any(s in kl for s in ('operator','demonstrator','scene','environment','phase','stage')):
     v=f.attrs[k];v=v.decode(errors='replace') if isinstance(v,bytes) else str(v)
     if any(s in kl for s in ('operator','demonstrator')):r['operator_explicit'][k]=v
     if any(s in kl for s in ('scene','environment')):r['scene_explicit'][k]=v
     if any(s in kl for s in ('phase','stage')):r['phase_field_availability'][k]='present_unused'
  assert (stat.st_size,stat.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
  if str(path)!=src:relocated.append(dict(cid=r['cid'],old=src,resolved=str(path)))
 else:missing.append(dict(cid=r['cid'],source=src,resolved=str(path)))
d.update(raw_attribute_keys=dict(keys),missing_raw_metadata=missing,relocated_sources=relocated);out=O/'PARENT_METADATA_RESOLVED.json';assert not out.exists();out.write_text(json.dumps(d,ensure_ascii=False,indent=2));print('metadata',len(d['records']),'missing',len(missing),'resolved',len(relocated),'operator',sum(bool(r['operator_explicit']) for r in d['records']),'scene',sum(bool(r['scene_explicit']) for r in d['records']));print('attributes',dict(keys))
