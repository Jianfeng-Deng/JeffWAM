from e86.common import *
OUT = T / 'code_e99w2'
ART = T / 'results' / 'e99w2_control_20261115_01'
E88 = T / 'code_e88'
INPUT = C / 'artifacts/E86-V3-20261113T144520'
REPORT = T / 'notes/e99w2_planned_traj_20261115.md'

def report(text):
    with REPORT.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write('\n' + text + '\n')

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def source_hash():
    h = hashlib.sha256()
    for folder in ('e86', 'e88', 'e90', 'e94a', 'e96a'):
        for p in sorted((OUT / folder).glob('*.py')):
            h.update(str(p.relative_to(OUT)).encode()); h.update(p.read_bytes())
    for p in sorted(OUT.glob("e98w_*.py")):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()
