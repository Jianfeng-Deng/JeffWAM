import os,sys,time,json,subprocess,datetime
from pathlib import Path
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e101a_diag_20261115_01';P=T/'code_e101a/diag.py'
while True:
 s=json.loads((O/'status_prepare.json').read_text()) if (O/'status_prepare.json').exists() else {}
 if s.get('status')=='complete':break
 if s.get('status')=='failed':sys.exit(1)
 time.sleep(3)
jobs=[]
for mode in ('gpu','probe'):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',XDG_CACHE_HOME=str(O/'cache'),MPLCONFIGDIR=str(O/'cache/mpl'))
 if mode=='probe':env['CUDA_VISIBLE_DEVICES']=''
 f=(O/(mode+'.log')).open('x');p=subprocess.Popen([sys.executable,'-u',str(P),mode],cwd=T,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs.append((mode,p,f));(O/(mode+'_pid.json')).write_text(json.dumps(dict(pid=p.pid,mode=mode)))
for mode,p,f in jobs:
 rc=p.wait();f.close();(O/(mode+'_exit.json')).write_text(json.dumps(dict(returncode=rc)))
