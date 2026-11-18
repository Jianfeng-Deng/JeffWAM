import os,time,json,subprocess
from e103d_runtime import *
while not (CACHE/'plan.json').exists():
 time.sleep(10)
plan=json.loads((CACHE/'plan.json').read_text())
for r in plan['records']:
 for w in r['windows']:
  meta=[]
  for q,off in enumerate([-24,-18,-12,-6,0]):
   tm=w['time_ns']+round(off*1e9/30)
   meta.append([[off/30,(tm-r['timestamps'][k][w['obs_rows'][q][k]])/1e9] for k in range(3)])
  w['obs_meta']=meta
raw=CACHE/'plan_raw_cpu.json';assert not raw.exists();(CACHE/'plan.json').rename(raw);atomic(CACHE/'plan.json',plan)
report('## 数据与资源预估（提取前）\n```json\n'+json.dumps(plan['summary'],ensure_ascii=False)+'\n```')
assert plan['summary']['payload_bytes']<35e9
assert plan['summary']['estimated_cache_gpuh_218fps_margin25']<4.8
uuid=wait_gpu();env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
atomic(CONTROL/'cache_launch.json',dict(gpu=uuid,pid=os.getpid(),brt=brt()))
with (CONTROL/'cache.log').open('x') as log:
 ret=subprocess.run(['/mnt/workspace/dengjianfeng/venvs/e29/bin/python','-u','e103d_cache.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
atomic(CONTROL/'cache_launch_status.json',dict(status='complete' if ret==0 else 'failed',returncode=ret,brt=brt()))
