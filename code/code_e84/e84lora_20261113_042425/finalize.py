import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import sys
sys.dont_write_bytecode=True
from common import *
from contract import sha
import shutil,fcntl

def append_locked(path,text):
 with pathlib.Path(path).open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text);f.flush();os.fsync(f.fileno())
def main():
 assert json.loads((P/'verification.json').read_text())['all_passed'];assert (C/'conclusion.txt').exists();assert not (P/'resource_stop.json').exists();assert not list(P.glob('*failure*.json'))
 from report import make
 make();receipt=json.loads((P/'report_draft_receipt.json').read_text());summary=T/'notes/e84_alignment_summary_20261113.md';stamp=brt().replace(':','').replace('-','').replace('.','');backup=summary.with_name(summary.name+'.pre_'+RUN+'_'+stamp+'.bak');assert not backup.exists();shutil.copy2(summary,backup)
 text=(T/'notes/e84_lora_summary_20261113.md').read_text();summary.write_text(text.replace('# E84 LoRA对比微调验证','# E84 语言—图像—动作对齐：冻结与LoRA汇总',1)+f'\n\n上一版冻结完整汇总已备份：`{backup}`。\n')
 state=json.loads((P/'progress.json').read_text());conclusion=(C/'conclusion.txt').read_text();metrics=json.loads((P/'paired_comparisons.json').read_text());runrow='\n\n## E84-P '+RUN+' — LoRA对照完成（'+brt()+' BRT）\n'
 runrow+='- 目标：受预算约束验证SigLIP2双塔LoRA能否修复方向/颜色/反事实失败点，仅probe-train训练，probe-val评估。\n'
 runrow+='- 改动：最后4层Q/V rank8，196608参数，lr1e-5、768固定步；真实/打乱同步数，FK+E83训练文本、反向/换臂/颜色难负例、0.1编码偏离约束；原模型不合并不替换；新增192793参数冻结小头对照。\n'
 runrow+=f'- 命令：/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python {C}/launch.py driver_gpu.py；CPU cpu_eval.py lora/shuffled；matched_head.py；analyze.py；verify_results.py；figures.py；finalize.py。GPU子进程CUDA_VISIBLE_DEVICES=4，PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8。\n'
 runrow+=f'- 目录：代码{C}；权重{L}；特征{R/"features"/RUN}；结果{P}；报告{T/"notes/e84_lora_summary_20261113.md"}；汇总{summary}；旧汇总备份{backup}。\n'
 runrow+=f'- 结果：训练1483FK+331E83/191parent，两组各768步；评估沿用22230时刻/542capture、150近似像素帧（val40/32parent）；parent等权2000bootstrap成对CI；本轮含调试GPU {receipt["gpu_hours"]:.4f}/4h，E84累计{1.276657608018981+receipt["gpu_hours"]:.4f}/10h；派生{receipt["derived_total_bytes"]/1e9:.3f}GB/<30GB。\n'
 runrow+='- 结论：'+conclusion.replace('\n',' ')+'\n'
 runrow+='- 状态：完成本轮LoRA验证与受控对照；通用检索无本地样本未评估，画面像素运动无真值未评估；150帧只诊断，禁用FK投影像素GT；无dev/test样本、无下载/仿真、无他人进程修改。首次设备放置错误在优化步0已修正，CPU准备性能/字段修正均留档，调试GPU时间全计入。\n'
 receiptpath=P/'final_logs_appended.json'
 if not receiptpath.exists():
  append_locked(T/'docs/experiment_log/EXPERIMENT_LOG.md',runrow)
  append_locked(T/'notes/worklog.md',f'\n- {brt()} BRT E84-P `{RUN}`：双塔rank8最后4层Q/V、真实/打乱各768步；原selection全量复提与实验1–3、近参数量冻结头、parent CI/漂移完成；GPU{receipt["gpu_hours"]:.4f}h，原模型未合并；结论见 `notes/e84_lora_summary_20261113.md`，alignment汇总已备份更新。\n')
  atomic(receiptpath,dict(timestamp_brt=brt(),experiment_log=True,worklog=True))
 append_report('## 最终完成\n\n'+brt()+'\n\n'+conclusion+f'\n\n报告 `{T/"notes/e84_lora_summary_20261113.md"}`；汇总 `{summary}`；备份 `{backup}`；验证 `{P/"verification.json"}`；GPU {receipt["gpu_hours"]:.4f}h。')
 update(status='completed_lora_validation',completed_brt=brt(),lora_gpu_hours=receipt['gpu_hours'],gpu_hours=receipt['gpu_hours'],total_e84_gpu_hours=1.276657608018981+receipt['gpu_hours'],summary=str(summary),summary_backup=str(backup),lora_summary=str(T/'notes/e84_lora_summary_20261113.md'),derived_total_bytes=receipt['derived_total_bytes'],adapters_merged=False,generic_retrieval='not_evaluated_no_local_generic_sample',pixel_motion='not_evaluated_no_truth')
 paths=[]
 for root in [C,L,P,R/'features'/RUN]:
  paths.extend(p for p in root.rglob('*') if p.is_file() and p.name not in ['ARTIFACT_SHA256.json','progress.lock'] and '__pycache__' not in p.parts)
 paths.extend([summary,T/'notes/e84_lora_summary_20261113.md',backup])
 atomic(P/'ARTIFACT_SHA256.json',dict(timestamp_brt=brt(),files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))]))
 atomic(P/'READY_lora_results.json',dict(timestamp_brt=brt(),status='completed',run=RUN,summary=str(summary),summary_sha256=sha(summary),lora_summary=str(T/'notes/e84_lora_summary_20261113.md'),lora_summary_sha256=sha(T/'notes/e84_lora_summary_20261113.md'),manifest=str(P/'ARTIFACT_SHA256.json'),manifest_sha256=sha(P/'ARTIFACT_SHA256.json'),verification=str(P/'verification.json')))
 print('FINALIZED',RUN,'GPUh',receipt['gpu_hours'],'backup',backup)
if __name__=='__main__':main()
