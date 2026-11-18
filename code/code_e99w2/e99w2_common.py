from e86.common import *
ROOT=T/'code_e99w2'
ART=T/'results/e99w2_control_20261115_01'
RUN=T/'results/e99w2_p3_20261115_04'
REPORT=T/'notes/e99w2_planned_traj_20261115.md'
INIT=T/'results/e98w_a_20261114_04/ckpt_05000.pt'
def report(s):
 with REPORT.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+s+'\n')
def source_hash():
 h=hashlib.sha256()
 for p in sorted(ROOT.rglob('*.py')):h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
 return h.hexdigest()
