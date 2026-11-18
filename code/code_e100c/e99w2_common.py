from e86.common import *
ROOT=T/'code_e100c'
GROUP=os.environ.get('E100_GROUP','C2'); SEED=int(os.environ.get('E100_SEED','90'))
CONTROL=T/'results/e100c_control_20261115_03'
ART=CONTROL/f'{GROUP}_s{SEED}'
RUN=T/f'results/e100c_{GROUP.lower()}_s{SEED}_20261115_03'
REPORT=T/'notes/e100c_chain_fix_20261115.md'
INIT=T/'results/e98w_a_20261114_04/ckpt_05000.pt'
def report(s):
 with REPORT.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+s+'\n')
def source_hash():
 h=hashlib.sha256()
 for p in sorted(ROOT.rglob('*.py')):h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
 return h.hexdigest()
