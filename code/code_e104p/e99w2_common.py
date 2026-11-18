from e86.common import *
ROOT=T/'code_e104p'
GROUP='A2'; SEED=int(os.environ.get('E102_SEED','90'))
CONTROL=T/'results/e104p_control_20261115_03'
ART=CONTROL/f'{GROUP}_s{SEED}'
RUN=T/f'results/e102a_{GROUP.lower()}_s{SEED}_20261115_03'
REPORT=T/'notes/e104p_pretrain_finetune_20261115.md'
INIT=T/'results/e102a_a2_s90_20261115_03/ckpt_02000.pt'
def report(s):
 with REPORT.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+s+'\n')
def source_hash():
 h=hashlib.sha256()
 for p in sorted(ROOT.rglob('*.py')):h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
 return h.hexdigest()

STEPS=2000
