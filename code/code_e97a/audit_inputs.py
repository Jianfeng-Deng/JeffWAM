import os,sys,json,time,traceback
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu");R=T/"results/e97a_robot_env_20261114_01"
sys.path.insert(0,str(T/"code_e96a"))
import numpy as np
from e90.data import load_data
try:
 p,ready,tr,dev,_=load_data()
 c=next(c for c in p.captures if c["canonical_id"]==tr.rows[0]["canonical_id"])
 z=np.load(c["numeric_cache"])
 out=dict(capture=c,row=tr.rows[0],ready=ready,cache=str(tr.base.cache),keys={k:dict(shape=z[k].shape,dtype=str(z[k].dtype),sample=z[k][:2].tolist()) for k in z.files})
 (R/"inspection.json").write_text(json.dumps(out,indent=2))
 print(json.dumps(dict(status="completed",keys={k:list(z[k].shape) for k in z.files},capture=c,row=tr.rows[0]),ensure_ascii=False))
except Exception:
 (R/"inspection_failure.txt").write_text(traceback.format_exc());raise
