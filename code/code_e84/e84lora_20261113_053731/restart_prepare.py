import os,signal,pathlib,json,subprocess,datetime,time
T=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');C=T/'code_e84/e84lora_20261113_042425';P=T/'data/e84_piper_align_v1/probes/e84lora_20261113_042425'
pid=303752
try:
 cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes();assert b'e84lora_20261113_042425/prepare_train.py' in cmd;os.kill(pid,signal.SIGTERM)
except FileNotFoundError:pass
j=json.loads((P/'progress.json').read_text());j['cpu_preparation_restart']='own PID303752 stopped before any GPU usage; sequential native video decoding too slow, fixed sparse seek with exact returned-frame-position check. Discard incomplete new-run pixel cache and rebuild. No old result or other process modified.';(P/'progress.json').write_text(json.dumps(j,indent=2));time.sleep(1)
f=(P/'prepare_sparse.log').open('a');p=subprocess.Popen(['/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python','-u',str(C/'prepare_train.py')],cwd=T,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);print('PREPARE_RESTART',p.pid)
