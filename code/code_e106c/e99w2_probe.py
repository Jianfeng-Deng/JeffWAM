import sys,json,numpy as np
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');sys.path.insert(0,str(T/'code_e99w2'))
from e86.data import Package
from e90.common import INPUT
p=Package();ready=json.loads((INPUT/'READY_V3_cache_windows.json').read_text());rs=json.loads(Path(ready['windows']['P3']).read_text());r=rs[0];cs={c['canonical_id']:c for c in p.captures};z=dict(np.load(cs[r['canonical_id']]['numeric_cache']))
print('input',str(INPUT),'ready',ready['windows'],'rows',len(rs),'parents',len({r['parent_component'] for r in rs}));print('row',r);print('stats',json.loads((INPUT/'stats.json').read_text()));print('numeric',[(k,list(v.shape),str(v.dtype)) for k,v in z.items()]);a=r['anchor'];print('action first',z['action'][a].tolist(),'qnext',z['q'][a+1].tolist(),'rawnext',z['qpos_raw'][a+1].tolist());print('files',str(cs[r['canonical_id']]['numeric_cache']))
