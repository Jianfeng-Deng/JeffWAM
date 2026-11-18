from pathlib import Path
import ast,json,hashlib,subprocess,os
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'code_e107m';C=T/'results/e107m_control_20261116_01'
for p in R.rglob('*.py'):ast.parse(p.read_text())
lock=json.loads((C/'PRETRAIN_LOCK.json').read_text());lock['source_sha256']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*.py')};lock['e107p_input_sha256']={n:hashlib.sha256((T/'results/e107p_diagnosis_20261116_02'/n).read_bytes()).hexdigest() for n in ('projection.npz','grip_probe.pkl','neighbors.json','dev_numeric.json')};(C/'PRETRAIN_LOCK.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2))
assert not (C/'pipeline.log').exists()
with (C/'pipeline.log').open('x') as f:
 p=subprocess.Popen(['nohup','/mnt/workspace/dengjianfeng/venvs/e29/bin/python','-u','e107m_pipeline.py'],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(C/'LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,source_files=len(lock['source_sha256']),cpu_tests=json.loads((C/'CPU_TESTS.json').read_text()))))
print('pipeline PID',p.pid,'CPU passed; AST passed; source locked')
