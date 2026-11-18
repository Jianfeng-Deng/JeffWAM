import pathlib,subprocess,sys,json
T=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');C=T/'code_e84/e84lora_20261113_042425';P=T/'data/e84_piper_align_v1/probes/e84lora_20261113_042425'
script=sys.argv[1];args=sys.argv[2:];f=(P/(pathlib.Path(script).stem+'_launcher.log')).open('a');p=subprocess.Popen(['/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python','-u',str(C/script),*args],cwd=T,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);print(json.dumps(dict(pid=p.pid,script=script)))
