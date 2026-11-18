import os,json,sys,collections,time,fcntl
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu'); C=T/'code_e94b';R=T/'results/e94b_dinov3_20261114_01';CACHE=T/'data/e94b_dinov3_cache';E93=T/'results/e93_dev_diag_20261114_01'
from e86.common import atomic,sha,brt
H=(10,25,50,100)
def report(s):
 with (T/'notes/e94b_dinov3_20261114.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+s+'\n');f.flush()
def status(stage,**kw):atomic(R/'status.json',dict(stage=stage,brt=brt(),**kw))
