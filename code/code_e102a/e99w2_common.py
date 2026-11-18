from e86.common import *
ROOT=T/'code_e102a'
GROUP=os.environ.get('E102_GROUP','A0'); SEED=int(os.environ.get('E102_SEED','90'))
CONTROL=T/'results/e102a_control_20261115_03'
ART=CONTROL/f'{GROUP}_s{SEED}'
RUN=T/f'results/e102a_{GROUP.lower()}_s{SEED}_20261115_03'
REPORT=T/'notes/e102a_action_fix_20261115.md'
INIT=T/'results/e100c_c2_s90_20261115_03/ckpt_05000.pt'
def report(s):
 with REPORT.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+s+'\n')
def source_hash():
 h=hashlib.sha256()
 for p in sorted(ROOT.rglob('*.py')):h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
 return h.hexdigest()

STEPS=2000
