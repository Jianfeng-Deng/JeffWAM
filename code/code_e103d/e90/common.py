from e86.common import *
OUT = T / 'code_e90'
ART = OUT / 'artifacts'
E88 = T / 'code_e88'
INPUT = C / 'artifacts/E86-V3-20261113T144520'
REPORT = T / 'notes/e90_full_v31_p1_20261114.md'

def report(text):
    with REPORT.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write('\n' + text + '\n')

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def source_hash():
    h = hashlib.sha256()
    for folder in ('e86', 'e88', 'e90'):
        for p in sorted((OUT / folder).glob('*.py')):
            h.update(str(p.relative_to(OUT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()
