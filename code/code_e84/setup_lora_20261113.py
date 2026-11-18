import pathlib,datetime,json,shutil,hashlib
T=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu'); R=T/'data/e84_piper_align_v1'; old='e84p_20261113_021548'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3)));run='e84lora_'+now.strftime('%Y%m%d_%H%M%S');C=T/'code_e84'/run;P=R/'probes'/run;L=R/'lora'/run
for d in [C,P,L]:d.mkdir(parents=True,exist_ok=False)
for name in ['common.py','contract.py','fk_consumption.py','encoder.py','evaluation_data.py','heads.py','train_heads.py','metrics.py','run_motion.py','run_behavior.py','run_pixel.py','pixel_features.py','extract.py']:
 shutil.copy2(T/'code_e84'/old/name,C/name)
shutil.copy2(T/'code_e84/e84p_20261112_153338/lora_module.py',C/'lora_module.py')
s=(C/'common.py').read_text().replace("RUN='e84p_20261113_021548'",f"RUN='{run}'\nOLD='{old}'\nOP=R if False else None")
s=s.replace("OP=R if False else None\n",'');s += "\nOLDP=R/'probes'/OLD\nL=R/'lora'/RUN\n";(C/'common.py').write_text(s)
for name in ['selection_full_v4.json','pixel_metadata_audit.json']:
 shutil.copy2(R/'probes'/old/name,P/name)
report=T/'notes'/f'e84_P_{run}.md';report.write_text(f'# E84-P LoRA受控验证\n\nRun `{run}`；BRT {now.isoformat()}。用户明确授权，继承`{old}`固定selection与split，不重做已完成审计。\n\n## 预注册\n\n最后4层双塔Q/V rank8、lr1e-5、固定768优化步/batch4；真实文本与打乱文本同步数，同seed初始化。对比+方向反转/换臂/颜色负例+原编码余弦漂移0.1。E83仅训练parent既有短caption；像素150帧仅事后诊断，不入训练、不选checkpoint。只保存最终adapter，不合并；原模型只读。训练/复提/调试合计≤4GPUh，GPU4单卡，allocator cap12GiB，准入空闲≥18GiB，运行空闲<18GiB立即停止。估算1.43GiB模型+0.01GiB LoRA/优化器+双塔末4层梯度激活(batch4,4图/对,文本≤64)+CUDA工作区，保守峰值12GiB，实际记录峰值。派生总量≤30GB。\n\n## 训练\n待完成。\n\n## 实验1–3及漂移\n待完成。\n\n## 结论\n待完成。\n')
(P/'progress.json').write_text(json.dumps(dict(run_id=run,status='prepared',started_brt=now.isoformat(),report=str(report),selection_sha256=hashlib.sha256((P/'selection_full_v4.json').read_bytes()).hexdigest(),old_run=old,gpu_hours=0,lora_gpu_hours=0,dev_test_samples_read=0),indent=2))
(C/'encoder.py').write_text((C/'encoder.py').read_text().replace('inject(self.model);','inject(self.model,layers=4);'))
print(run)
