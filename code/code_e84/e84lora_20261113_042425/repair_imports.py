import pathlib,json
C=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84lora_20261113_042425')
p=C/'fk_consumption.py';s=p.read_text().replace('from language_audit import stats','# prior audit retained').replace("(P/'progress.json')", "(OLDP/'progress.json')");p.write_text(s)
p=C/'prepare_train.py';s=p.read_text().replace("selection['parents'].get(x['parent'])", "json.loads((R/'split/probe_split.json').read_text())['parents'].get(x['parent'])");p.write_text(s)
P=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84lora_20261113_042425');j=json.loads((P/'progress.json').read_text());j['implementation_fixes_before_gpu']=['import dependency removed, inherit accepted old READY receipt','selection parent-count metadata distinguished from locked parent mapping'];(P/'progress.json').write_text(json.dumps(j,indent=2))
print('fixed')
