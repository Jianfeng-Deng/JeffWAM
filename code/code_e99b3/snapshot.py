from common import *
paths=[]
for folder in ('code_e82','code_e86','code_e88','code_e98w'):
 paths.extend(p for p in (T/folder).rglob('*.py') if '__pycache__' not in p.parts)
paths.extend(T/p for p in ('AGENTS.md','notes/e97b_stages_predicates_astra_20261114.md','notes/e98_dev_labels_20261114.md','notes/e98_train_labels_20261114.md','notes/e98w_robot_conditioned_20261114.md','notes/e96a_spatial_bypass_20261114.md'))
put('protected_snapshot.json',{str(p):sha(p) for p in paths});print('Protected files',len(paths))
models={}
for folder in ('qwen3_vl_4b','siglip2'):
 d=Path('/mnt/workspace/dengjianfeng/models')/folder;models[folder]={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=sha(p) if p.suffix=='.json' else None) for p in d.iterdir() if p.is_file()}
put('model_provenance.json',models)
# Original read-only video/HDF metadata; no data copied or source moved.
source={}
for s in read('sources.json'):
 for p in [s['source'],*s['videos'].values()]:
  st=Path(p).stat();source[p]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
put('source_stat_snapshot.json',source)
