import json, pathlib, os, fcntl, datetime, time
T=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu')
RUN='e84lora_20261113_042425'
OLD='e84p_20261113_021548'
R=T/'data/e84_piper_align_v1'
P=R/'probes'/RUN
C=T/'code_e84'/RUN
def brt(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def atomic(path,obj):
    path=pathlib.Path(path); tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w') as f: json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def update(**kw):
    with (P/'progress.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        p=json.loads((P/'progress.json').read_text()); p.update(kw); p['updated_brt']=brt(); atomic(P/'progress.json',p)
def append_report(s):
    p=json.loads((P/'progress.json').read_text())
    with pathlib.Path(p['report']).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX); f.write('\n'+s+'\n'); f.flush(); os.fsync(f.fileno())

OLDP=R/'probes'/OLD
L=R/'lora'/RUN
