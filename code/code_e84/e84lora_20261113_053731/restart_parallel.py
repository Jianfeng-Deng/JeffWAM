import pathlib,os,signal,subprocess,time,json
T=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');C=T/'code_e84/e84lora_20261113_042425';P=T/'data/e84_piper_align_v1/probes/e84lora_20261113_042425'
pid=396086
try:
 assert b'e84lora_20261113_042425/prepare_train.py' in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes();os.kill(pid,signal.SIGTERM)
except FileNotFoundError:pass
time.sleep(1)
j=json.loads((P/'progress.json').read_text());j['cpu_preparation_parallel_rebuild']='Native1280 video sparse seeking remains CPU bottleneck. Stop only own PID396086 before GPU, rebuild pending cache using8 CPU decode workers on192CPU host/load37. Same fixed selection/hash pairs/labels. New plan SHA + completed source stat checkpoints supports resume.';(P/'progress.json').write_text(json.dumps(j,indent=2))
f=(P/'prepare_parallel.log').open('a');p=subprocess.Popen(['/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python','-u',str(C/'prepare_train.py')],cwd=T,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);print('PARALLEL_PREPARE',p.pid)
