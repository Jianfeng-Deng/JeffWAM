import json,collections,math
from pathlib import Path
import numpy as np
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=Path((T/'code_e105y/run_path.txt').read_text().strip());base=T/'results/e104p_control_20261115_03';raw={g:json.loads((base/('eval_'+g)/'evaluation.json').read_text())['world'] for g in ['F0','F1','F2']};out=[]
for b in ['F0','F2']:
 for h in [10,25,50,100]:
  aa={(r['index'],r['parent']):r for r in raw['F1'] if r['method']=='no_traj' and r['h']==h};bb={(r['index'],r['parent']):r for r in raw[b] if r['method']=='no_traj' and r['h']==h};keys=sorted(aa.keys()&bb.keys());cp=np.mean([aa[k]['copy'] for k in keys]);d=np.array([100*(bb[k]['model']-aa[k]['model'])/cp for k in keys]);parent=collections.defaultdict(list)
  for k,v in zip(keys,d):parent[k[1]].append(v)
  pd=np.array([np.mean(v) for v in parent.values()]);out.append(dict(pair='F1-'+b,h=h,windows=len(keys),parents=len(parent),copy_mean=cp,window_sd_pp=float(d.std(ddof=1)),parent_sd_pp=float(pd.std(ddof=1)),mean_pp=float(d.mean())))
f=R/'variance.json';assert not f.exists();f.write_text(json.dumps(out,indent=2));print(out)
