import sys,shutil,hashlib,json,subprocess
from common import *
from contract import sha
old=R/'probes/e84p_20261112_153338'
progress=json.loads((P/'progress.json').read_text())
assert progress['gpu_hours']==0 and progress['pixel_metadata_audit_complete']
accepted=json.loads((P/'accepted_receipts.json').read_text())
for name,entry in accepted.items():assert sha(R/name)==entry['sha256'],f'Input changed: {name}'
summary=T/'notes/e84_alignment_summary_20261113.md'
backup=summary.with_name(summary.name+'.pre_shared_resume_'+brt().replace(':','').replace('-','')+'.bak')
shutil.copy2(summary,backup)
source=old/'selection_full_v4.json';target=P/source.name
if not target.exists():shutil.copy2(source,target)
assert sha(target)==sha(source)
progress.update(status='shared_gpu_resume_prepared',fk_gt_ready=json.loads((old/'progress.json').read_text())['fk_gt_ready'],selection_path=str(target),selection_sha256=sha(target),summary_backup=str(backup),scope_gate='revised_user_approved_150_model_approximate_pixel_gt_no_projection',resumed_brt=brt(),lora_trigger='requires_user_confirmation_no_autorun',awaiting=None)
atomic(P/'progress.json',progress)
append_report('## 共享显存授权续跑 '+brt()+'\n\n沿用本run与前序字段审计，旧汇总已备份 `'+str(backup)+'`。沿用旧run的固定selection_full_v4（SHA '+sha(target)+'），probe-train/val=11376/10854，242parent；不重新选样。新增resource_guard.py、extract.py、pixel_features.py及机器人系探针入口；只用GPU4，小batch8，估算峰值8192MiB，准入空闲>=12288MiB，allocator上限8192MiB，10秒显存/预算检查。无自动LoRA，像素运动无真值，绝不读取projection_motion.parquet。已核验三READY未变化，不重做已完成审计。')
for path in [R/'features'/RUN,R/'lora'/RUN]:path.mkdir(exist_ok=True)
atomic(P/'resume_source_manifest.json',{'timestamp_brt':brt(),'files':{str(f):sha(f) for f in C.glob('*.py')},'summary_backup':str(backup),'selection':str(target)})
print(json.dumps({'run':RUN,'status':'prepared','selection_sha256':sha(target),'summary_backup':str(backup)}))
