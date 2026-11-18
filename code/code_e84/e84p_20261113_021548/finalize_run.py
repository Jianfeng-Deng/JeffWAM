import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,pathlib,fcntl,shutil,hashlib,sys
from common import *
from contract import sha

def write_append(path,txt):
    with pathlib.Path(path).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(txt);f.flush();os.fsync(f.fileno())
def main(conclusion_path):
    receipt=json.loads((P/'summary_receipt_draft.json').read_text());draft=pathlib.Path(receipt['draft']);assert sha(draft)==receipt['sha256']
    conclusion=pathlib.Path(conclusion_path).read_text();assert len(conclusion)>100 and '待填写' not in conclusion
    summary=T/'notes/e84_alignment_summary_20261113.md'
    stamp=brt().replace(':','').replace('-','');backup=summary.with_name(summary.name+'.before_commit_'+stamp+'.bak');shutil.copy2(summary,backup)
    text=draft.read_text().replace('待填写：必须结合实际表格由执行者给出明确建议和左右可判定性。LoRA未运行，用户最新修订要求等待确认，不能把此项N/A写成失败。更强VLM只可建议，不下载。',conclusion)
    limits=json.loads((P/'pixel_statistical_limits.json').read_text())
    text=text.replace('## 阶段一建议与LoRA状态',f'全零经验bootstrap可能退化为[0,0]，不代表总体率确定为0。左右trusted可分r子集仅0/{limits["parents"]}次双句成功；每parent一帧时独立二项Clopper–Pearson95%上界{limits["supplemental_independent_parent_binomial_CP95_upper"]*100:.2f}%，更不涵盖标注共有误差。\n\n## 阶段一建议与LoRA状态')
    text+='\n结果图：`'+str(P/'frozen/figures/motion_and_counterfactual.png')+'`；方向混淆矩阵图：`'+str(P/'frozen/figures/h50_direction_confusions.png')+'`。完整数值JSON及NPZ为表格来源。\n'
    tmp=summary.with_name(summary.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(text);os.replace(tmp,summary)
    report=T/'notes'/f'e84_P_{RUN}.md';r=report.read_text();r=r.replace('# E84-P 续跑：资源门槛停止','# E84-P 冻结特征探针完成（共享显存续跑）\n\n最新完成结果见文末及当日汇总；以下保留之前资源阻塞的断点历史。',1)
    lang=json.loads(sorted(P.glob('e83_language_information_*.json'))[-1].read_text())
    r=r.replace('。103条记录对应102parent；训练应按parent去重/等权，避免跨tier重复加权。',f'。当前快照{lang["records"]}条记录/{lang["unique_parents"]}parent，A/B共用{len(lang["A_B_parent_overlap"])}parent；训练应按parent去重/等权，避免跨tier重复加权。')
    r=r.replace('绝不读取projection_motion.parquet','不解析或使用projection_motion.parquet作像素真值（READY完整性校验可能读取其字节哈希）')
    report.write_text(r)
    write_append(report,'\n## 本轮完成 '+brt()+'\n\n'+conclusion+'\n\n最终完整表格、CI、对照和局限：`'+str(summary)+'`。实际GPU '+f'{receipt["gpu_hours"]:.4f} GPUh，LoRA0；E84总派生{receipt["total_e84_bytes"]/1e9:.3f}GB。'+' 无WAM训练、无下载、dev/test样本读取0；F/G和旧run只读。\n')
    code_manifest={str(p):sha(p) for p in C.glob('*.py')};atomic(P/'final_source_manifest.json',{'timestamp_brt':brt(),'files':code_manifest,'scope':'frozen feature/probe experiments only; no LoRA launcher'})
    result_files=[p for p in (P/'frozen').rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.npz','.pt','.png','.svg']]
    atomic(P/'result_manifest.json',{'timestamp_brt':brt(),'files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(result_files)]})
    files=[summary,report,P/'frozen/pixel_results.json',P/'frozen/READY_motion_results.json',P/'frozen/READY_behavior_results.json',R/'features'/RUN/'frozen/READY_features.json',P/'pixel_statistical_limits.json',P/'method_checks.json',P/'map_visual_review.json',P/'e83_64token_information.json',P/'result_manifest.json',P/'lora_trigger_evidence.json',P/'READY_feature_verification.json']
    atomic(P/'READY_probe_results.json',{'status':'completed_frozen_experiments_lora_not_run','run_id':RUN,'timestamp_brt':brt(),'files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in files],'feature_samples':22230,'pixel_frames':150,'gpu_hours':receipt['gpu_hours'],'lora_gpu_hours':0,'dev_test_samples_read':0,'fk_projection_used_as_pixel_truth':False,'summary_backup':str(backup)})
    selection=json.loads((P/'selection_full_v4.json').read_text());groups={}
    for r in selection['samples']:groups.setdefault(r['canonical_id'],[]).append(r)
    actual_model_frames=150+sum(len({int(r[k][hi][ci]) for r in rows for k,count in [('camera_rows',5),('endpoint_camera_rows',2)] for hi in range(count)}-{-1}) for rows in groups.values() for ci in range(3))
    update(status='completed_frozen_experiments_lora_await_user',cpu_pipeline_failed=False,cpu_failure_recovered=True,gpu_hours=receipt['gpu_hours'],probe_gpu_hours=receipt['gpu_hours'],lora_gpu_hours=0,feature_times=22230,model_frames=actual_model_frames,raw_images_read=actual_model_frames,raw_images_read_definition='encoded source image instances including150 GT PNGs; excludes skipped decoder grabs',model_image_frames_definition='deduplicated video rows per capture/view across current+4history+2observed endpoints, plus150pixel frames',train_runs=24,train_runs_definition='4 motion datasets x4 heads (linear/contrastive xtrue/shuffle)=16;2arms x3displacement conditions=6; plus2 discarded CPU linear fits before boolean-contrast metric fix',successful_train_runs=22,discarded_cpu_fit_attempts=2,completed_brt=brt(),summary=str(summary),lora_trigger='not_run_requires_user_confirmation; laterality_inconclusive',lora_motion_rule_evidence=str(P/'lora_trigger_evidence.json'),dev_test_samples_read=0,awaiting='Only user decision on LoRA; authorized frozen experiments complete')
    marker=P/'final_logs_appended.json'
    if not marker.exists():
        entry=f'''\n\n## E84-P {RUN} — 共享显存续跑完成（{brt()} BRT）
- 目标：继承断点，冻结SigLIP2提特征，完成150帧近似像素图文诊断、机器人系FK运动探针和语言条件位移反事实，不训练WAM。
- 改动：仅code_e84/{RUN}及features/probes/{RUN}；小batch8、单GPU4、8GiB allocator cap、空闲>=12GiB准入/运行压力检查；像素只读G近似标注，拒绝FK投影运动真值。沿用固定selection和已完成审计；LoRA不启动。
- 命令：setsid nohup env CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 {sys.executable} -u {C}/extract.py full cuda:0；CUDA_VISIBLE_DEVICES= {sys.executable} -u {C}/cpu_pipeline.py；{sys.executable} {C}/finish_report.py；{sys.executable} {C}/finalize_run.py {conclusion_path}。
- 目录：{R}/features/{RUN}；{P}；报告{summary}，旧版有备份。
- 结果：22230时刻（train11376/val10854），150像素帧（val40/32parent），运动4数据集、行为2臂；GPU{receipt['gpu_hours']:.4f}GPUh，LoRA0；总派生{receipt['total_e84_bytes']/1e9:.3f}GB；CI按parent2000bootstrap。6项方法验证、特征receipt SHA与源只读stat核对通过。
- 结论：{conclusion.replace(chr(10),' ')}
- 状态：冻结实验完成；LoRA等待用户确认。像素运动无真值未评估；没有dev/test样本读取、权重/数据下载、仿真、他人进程操作或FK投影像素GT。
'''
        write_append(T/'docs/experiment_log/EXPERIMENT_LOG.md',entry)
        write_append(T/'notes/worklog.md',f'\n- {brt()} BRT E84-P `{RUN}` 共享GPU4续跑完成：22230时刻冻结特征、150帧粗像素诊断、FK运动及位移反事实；GPU {receipt["gpu_hours"]:.4f}h/LoRA0；左右不可判定，LoRA等用户确认；汇总 `{summary}`（已备份旧版）。\n')
        atomic(marker,{'timestamp_brt':brt(),'experiment_log':str(T/'docs/experiment_log/EXPERIMENT_LOG.md'),'worklog':str(T/'notes/worklog.md')})
    print(json.dumps({'status':'completed','summary':str(summary),'receipt':str(P/'READY_probe_results.json'),'gpu_hours':receipt['gpu_hours']},ensure_ascii=False))
if __name__=='__main__':main(sys.argv[1])
