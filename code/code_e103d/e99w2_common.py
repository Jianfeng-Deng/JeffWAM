from e86.common import *
ROOT=T/'code_e103d'
GROUP=os.environ.get('E100_GROUP','C2'); SEED=int(os.environ.get('E100_SEED','90'))
CONTROL=T/'results/e103d_control_20261115_01'
ART=CONTROL/os.environ.get('E103_GROUP','D0')
RUN=T/f"results/e103d_{os.environ.get('E103_GROUP','D0').lower()}_s90_20261115_01"
REPORT=T/'notes/e103d_external_data_20261115.md'
INIT=T/'results/e100c_c2_s90_20261115_03/ckpt_05000.pt'
def report(s):
 with REPORT.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+s+'\n')
def source_hash():
 h=hashlib.sha256()
 for p in sorted(ROOT.rglob('*.py')):h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
 return h.hexdigest()
