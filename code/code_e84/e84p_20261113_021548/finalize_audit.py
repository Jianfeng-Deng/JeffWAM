import pathlib,json,hashlib
T=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu')
run='e84p_20261113_021548'
P=T/'data/e84_piper_align_v1/probes'/run
r=T/'notes'/f'e84_P_{run}.md'
c=T/'code_e84'/run/'resume_audit.py'
s=r.read_text().replace('## 输入核验\n\n进行中。','## 输入核验\n\n已完成；详见下方交付及字段核验。')
s=s.replace('## 像素真值局限\n\n待追加。','## 像素真值局限\n\n已在下方“像素真值局限与结论降级”中记录。')
if '## 收尾核对' not in s:
    s+='\n## 收尾核对\n\n仅本会话前置只读目录扫描PID1855954已核对cmdline/父PID后终止；无自有等待、训练或推理任务残留。本次新增文件不足0.1MB（不含共享日志追加），远低于30GB预算；没有新增特征。\n'
r.write_text(s)
s=c.read_text()
marker='now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()'
guard="""if json.loads((P/'progress.json').read_text()).get('status') in {'stopped_no_free_gpu','stopped_after_initial_no_free_gpu'}:
    print('Run finalized; inspect progress.json. Do not regenerate receipts or duplicate logs; use a new resume entry point.')
    raise SystemExit(0)

"""
if guard not in s:
    s=s.replace(marker,guard+marker)
c.write_text(s)
compile(s,str(c),'exec')
files=list(P.glob('*.json'))+[c,r,T/'notes/e84_alignment_summary_20261113.md']
items=[{'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files if f.name!='ARTIFACT_SHA256.json']
(P/'ARTIFACT_SHA256.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
print(json.dumps({'status':json.loads((P/'progress.json').read_text())['status'],'own_artifact_bytes':sum(x['bytes'] for x in items),'artifact_hashes':len(items),'code_compile':'passed','experiment_log_entry':run in (T/'docs/experiment_log/EXPERIMENT_LOG.md').read_text(),'worklog_entry':run in (T/'notes/worklog.md').read_text()}))
