import os,sys,json,subprocess,datetime
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
PYTHON="/mnt/workspace/dengjianfeng/venvs/e29/bin/python"
TRAIN0=T/"results/e111f_flow_20261116_02";TRAIN1=T/"results/e111f_flow_20261116_04";TRAIN2=T/"results/e111f_flow_20261116_05"
DRIVER=T/"results/e111f_eval_driver_20261116_02"
def brt(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def atomic(p,x):
 p=Path(p);tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));os.replace(tmp,p)
def main():
 assert not DRIVER.exists()
 DRIVER.mkdir()
 gpu=os.environ.get("CUDA_VISIBLE_DEVICES","")
 assert gpu.startswith("GPU-")
 atomic(DRIVER/"status.json",dict(status="starting",gpu=gpu,checkpoints=[4000,8000,12000,16000],brt=brt()))
 rows=[]
 for add in (4000,8000,12000,16000):
  ckpt=(TRAIN0 if add==4000 else (TRAIN1 if add==8000 else TRAIN2))/f"ckpt_add_{add:05d}.pt"
  out=T/f"results/e111f_eval_add_{add:05d}"
  assert ckpt.is_file() and not out.exists(),(ckpt,out)
  env=dict(os.environ,E102_GROUP="A2",E102_SEED="90",CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1")
  p=subprocess.run([PYTHON,str(T/"code_e111f/e111f_eval.py"),str(ckpt),str(out)],cwd=T,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  (DRIVER/f"eval_{add:05d}.log").write_text(p.stdout)
  assert p.returncode==0,(add,p.stdout[-3000:])
  q=subprocess.run([PYTHON,str(T/"code_e111f/e111f_finalize.py"),str(out)],cwd=T,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  (DRIVER/f"finalize_{add:05d}.log").write_text(q.stdout)
  assert q.returncode==0,(add,q.stdout[-3000:])
  summary=json.loads((out/"summary.json").read_text())
  rows.append(dict(additional_step=add,checkpoint_sha256=summary["checkpoint_sha256"]["F"],summary=str(out/"summary.json"),gpu_seconds=summary["gpu_seconds"],peak_mib=summary["peak_mib"],brt=summary["brt"]))
  atomic(DRIVER/"status.json",dict(status="evaluating",completed=rows,remaining=[n for n in (4000,8000,12000,16000) if n>add],gpu=gpu,brt=brt()))
 atomic(DRIVER/"summary.json",dict(status="complete",checkpoints=rows,gpu=gpu,brt=brt()))
 atomic(DRIVER/"status.json",dict(status="complete",completed=rows,gpu=gpu,brt=brt()))
if __name__=="__main__":
 main()
