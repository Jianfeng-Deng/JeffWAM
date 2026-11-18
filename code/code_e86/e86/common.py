from pathlib import Path
import hashlib,json,os,datetime,fcntl
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu')
R=T/'data/e82_piper_v1'
C=T/'code_e86'
V1_HASH='261e0552cc0fa1b34e69f14475f3d1bf3fcd51b547a17fdb0303c784694a99e5'
PARENT_HASH='df1d73ec2f951f53001fb9c7df2d0cfac3b18100ce303112c588508326a2e356'
DATA_HASH='23f71c27bb27afd3bde5fa13b1650aa159cf3bc3a9f6568fbb8558de6f18b139'
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def brt():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def atomic(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def append_report(text):
    with (T/'notes/e86_M_v3_smoke_20261113.md').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+text+'\n');f.flush()
def seed_for(seed,*keys):return int(hashlib.sha256(str((seed,keys)).encode()).hexdigest()[:15],16)%(2**63-1)
def generator(seed,*keys,device='cpu'):
    import torch
    return torch.Generator(device=device).manual_seed(seed_for(seed,*keys))
