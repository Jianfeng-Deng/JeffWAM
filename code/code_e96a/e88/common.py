from e86.common import *
OUT=T/'code_e88'
ART=OUT/'artifacts'
INPUT=C/'artifacts/E86-V3-20261113T144520'
REPORT=T/'notes/e88_W_v31_20261114.md'
def report(s):
    with REPORT.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+s+'\n');f.flush()
