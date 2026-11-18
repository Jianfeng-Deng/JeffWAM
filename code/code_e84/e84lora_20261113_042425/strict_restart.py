import pathlib,json,os,signal,time,subprocess,datetime,shutil,hashlib
T=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'data/e84_piper_align_v1';pilot='e84lora_20261113_042425';oldC=T/'code_e84'/pilot;oldP=R/'probes'/pilot;oldL=R/'lora'/pilot
raw=subprocess.check_output(['ps','-eo','pid,args'],text=True);stopped=[]
for line in raw.splitlines():
 bits=line.strip().split(None,1)
 if len(bits)!=2 or not bits[0].isdigit():continue
 pid=int(bits[0]);args=bits[1]
 if pid==os.getpid() or str(oldC) not in args:continue
 if any(str(oldC/name) in args for name in ['driver_gpu.py','extract.py','cpu_eval.py','wait_analysis.py','run_pixel.py']):
  try:os.kill(pid,signal.SIGTERM);stopped.append(pid)
  except ProcessLookupError:pass
time.sleep(2);prior=0
for p in (oldP/'gpu_ledger').glob('*.json'):
 d=json.loads(p.read_text())
 if d['status']!='finished':d.update(status='finished',elapsed_s=time.time()-d['start_utc'],stopped_reason='strict E83 no-nearest-pair compliance correction');p.write_text(json.dumps(d,indent=2))
 prior+=d['elapsed_s']/3600
s=json.loads((oldP/'progress.json').read_text());s.update(status='superseded_E83_nearest_pairs_not_primary',superseded_reason='320/331 E83 pairs used near anchors; strict no-retiming final run retains exact-source E83 positives only',gpu_hours=prior,lora_gpu_hours=prior,stopped_own_pids=stopped);(oldP/'progress.json').write_text(json.dumps(s,indent=2))
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3)));run='e84lora_'+now.strftime('%Y%m%d_%H%M%S');C=T/'code_e84'/run;P=R/'probes'/run;L=R/'lora'/run
for d in [C,P,L]:d.mkdir(parents=True,exist_ok=False)
for f in oldC.glob('*.py'):
 if f.name in ['strict_restart.py','common.py']:continue
 shutil.copy2(f,C/f.name)
common=(oldC/'common.py').read_text().replace("RUN='"+pilot+"'", "RUN='"+run+"'")+f"\nPILOT='{pilot}'\nPRIOR_LORA_GPU_HOURS={prior!r}\n";(C/'common.py').write_text(common)
for name in ['selection_full_v4.json','pixel_metadata_audit.json']:
 shutil.copy2(oldP/name,P/name)
report=T/'notes'/f'e84_P_{run}.md';report.write_text(f'# E84-P 严格同源LoRA验证\n\nID `{run}`；BRT {now.isoformat()}。此前 `{pilot}` 的近邻E83正配对不作为正式结论，已停止自己的后台任务并完整保留。其GPU {prior:.6f}h计入本阶段4h总预算。新run保持原selection/split/FK标签与超参数，只保留同锚点E83 caption；固定768步，E83每8步一次以避免11个精确caption过度重复，其他步骤4个FK对。旧结果不覆盖、不据val选checkpoint，源动作/FK/像素标签从未平移。\n\n## 训练\n待完成。\n\n## 评估\n待完成。\n\n## 结论\n待完成。\n')
(P/'progress.json').write_text(json.dumps(dict(run_id=run,status='preparing_exact_E83',report=str(report),started_brt=now.isoformat(),selection_sha256=hashlib.sha256((P/'selection_full_v4.json').read_bytes()).hexdigest(),prior_run=pilot,prior_lora_gpu_hours=prior,lora_gpu_hours=prior,gpu_hours=prior,dev_test_samples_read=0),indent=2))
# Reuse only immutable, already decoded train inputs and teachers, selecting exact E83 source anchors.
import numpy as np
items=json.loads((oldL/'training_items.json').read_text());pairs=json.loads((oldP/'e83_training_pair_provenance.json').read_text())['rows'];exact={(r['canonical_id'],r['image_anchor_source_row'],r['text']) for r in pairs if r['source_delta_frames']==0};keep=[i for i,r in enumerate(items) if r['kind']=='fk' or (r['canonical_id'],r['anchor_source_row'],r['text']) in exact];newitems=[items[i] for i in keep]
for r in newitems:
 if r['kind']=='e83':r['original_e83_source_frame_index']=r['anchor_source_row'];r['source_delta_frames']=0
src=np.load(oldL/'train_pixels.npy',mmap_mode='r');dest=np.lib.format.open_memmap(L/'train_pixels.npy',mode='w+',dtype='u1',shape=(len(keep),4,3,224,224))
for j,i in enumerate(keep):dest[j]=src[i]
dest.flush();np.save(L/'teacher_globals.npy',np.load(oldL/'teacher_globals.npy')[keep]);(L/'training_items.json').write_text(json.dumps(newitems,ensure_ascii=False,indent=2))
ready=json.loads((oldL/'READY_training_data.json').read_text());ready.update(timestamp_brt=now.isoformat(),items=len(newitems),fk_pairs=sum(r['kind']=='fk' for r in newitems),e83_pairs=sum(r['kind']=='e83' for r in newitems),parents=len({r['parent'] for r in newitems}),source_cache=str(oldL/'train_pixels.npy'),source_cache_sha256=hashlib.file_digest((oldL/'train_pixels.npy').open('rb'),'sha256').hexdigest(),strict_e83_exact_anchor=True,all_e83_source_delta_frames=0);(L/'READY_training_data.json').write_text(json.dumps(ready,indent=2))
# Both remaining phase and all previous debug/extraction time share the same4GPUh cap.
p=C/'resource_guard.py';text=p.read_text().replace("total=sum(json.loads", "total=PRIOR_LORA_GPU_HOURS*3600+sum(json.loads");p.write_text(text)
p=C/'train_lora.py';text=p.read_text().replace("n_e=1 if len(eids) else 0", "n_e=1 if len(eids) and step%8==0 else 0");text=text.replace("if captions else base_static", "if captions else base_static");text=text.replace("e83_weight=.5", "e83_weight=.5,e83_every_n_steps=8,strict_e83_exact_anchor=True");p.write_text(text)
p=C/'report.py';text=p.read_text().replace("hours=sum(x['elapsed_s'] for x in ledger)/3600", "hours=PRIOR_LORA_GPU_HOURS+sum(x['elapsed_s'] for x in ledger)/3600");p.write_text(text)
p=C/'verify_results.py';text=p.read_text().replace("sum(json.loads(p.read_text())['elapsed_s'] for p in (P/'gpu_ledger').glob('*.json'))<4*3600", "PRIOR_LORA_GPU_HOURS*3600+sum(json.loads(p.read_text())['elapsed_s'] for p in (P/'gpu_ledger').glob('*.json'))<4*3600");p.write_text(text)
(P/'superseded_run_receipt.json').write_text(json.dumps(dict(run=pilot,gpu_hours=prior,reason=s['superseded_reason'],stopped_own_pids=stopped),indent=2))
shutil.copytree(oldP/'matched_frozen',P/'matched_frozen')
(C/'launch.py').write_text((oldC/'launch.py').read_text().replace("C=T/'code_e84/e84lora_20261113_042425'","C=pathlib.Path(__file__).parent").replace("P=T/'data/e84_piper_align_v1/probes/e84lora_20261113_042425'","P=T/'data/e84_piper_align_v1/probes'/C.name"))
print(json.dumps(dict(run=run,prior_gpu_hours=prior,items=len(newitems),fk=ready['fk_pairs'],e83=ready['e83_pairs'],stopped=stopped)))
