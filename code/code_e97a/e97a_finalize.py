from e97a_core import *
import fcntl
assert (R/'EVALUATION_COMPLETE.json').exists();assert (R/'tables_written.json').exists();assert not (R/'FINAL_STATUS.json').exists()
s=json.loads((R/'dev_summary.json').read_text());energy=json.loads((R/'energy_decomposition.json').read_text());attribution=json.loads((R/'gain_attribution.json').read_text());pairs=json.loads((R/'paired_incremental.json').read_text());records=json.loads((R/'dev_records.json').read_text())
# Check frozen reference R0 point errors without any new model forward.
refs={(r['index'],r['h']):r for r in json.loads((E93/'linear_records.json').read_text())};r0audit=[]
for h in H:
 rs=[r for r in records if r['h']==h and r['camera']=='all' and r['method']=='R0' and r['section']=='full'];d=max(abs(r['model']-refs[r['index'],h]['model']) for r in rs);assert d<2e-5;r0audit.append(dict(h=h,max_abs_old_mse_diff=d))
save('R0_dev_replay_audit.json',r0audit)
protected=json.loads((R/'protected_hashes.json').read_text());changed=[p for p,h in protected.items() if sha(p)!=h];assert not changed
restored=json.loads((R/'dev_state_restore_receipt.json').read_text());sources_changed=[]
for r in restored:
 p=Path(r['source']);st=p.stat()
 if [st.st_size,st.st_mtime_ns]!=r['source_stat']:sources_changed.append(str(p))
assert not sources_changed
addmax=max(d['additivity_error'] for d in energy);pmax=max(d['subspace_additivity_error'] for d in energy);gmax=max(d['additivity_error'] for d in attribution)
audit=dict(brt=now(),protected_files=len(protected),protected_changed=changed,restored_raw_sources=len(restored),restored_raw_source_stat_changes=sources_changed,R0_replay=r0audit,energy_identity_max_error=addmax,orthogonal_energy_max_error=pmax,gain_identity_max_error=gmax,Schur=json.loads((R/'equation_validation.json').read_text()),dev_pass='one completed CPU prediction pass; prior coverage-preflight and CUDA-admission both produced zero predictions/scores',ckpt_sha256=sha(T/'results/e96a_m1_p1_20261114_01/ckpt_05000.pt'))
save('final_audit.json',audit)
report=T/'notes/e97a_robot_env_decomp_20261114.md'
conclusion='''
## 11. 结论与建议（完成，不执行训练）

**这轮支持把未来机器人轨迹作为条件，并保留独立的剩余预测能力；尚不支持宣称已经分出了物理手臂/环境的真实百分比。**

1. 状态关联的线性可解释份额随时域增加。总体 R1 skill 为 +10 **0.915%** [0.609,1.239]、+25 **4.747%** [3.320,6.215]、+50 **10.220%** [7.481,12.959]、+100 **26.219%** [21.062,31.030]。对应未解释误差份额为 **99.085% /95.253% /89.780% /73.781%**。这是给定当前/历史/未来实测状态及FK的线性诊断结果，不是机器人造成变化的上界或下界：非线性机器人外观尚未解释，物体动作响应又会计入机器人关联信号。
2. 图像与状态互补。R2全量skill **8.101/14.999/22.741/39.438%**，比R0提高 **1.323/5.720/10.036/16.823pp**，四项配对95%CI下界都>0；比R1提高7.185/10.252/12.521/13.220pp。未来轨迹条件有统计依据，不能以“机器人可FK”推导视觉目标应全部删除，也不能将状态不解释部分视作纯环境。
3. **E96a赢复制的部分是不是主要来自手臂？不能确认物理手臂归因。** 在训练R1的124维输出span内，确有 E96a 总收益的 **92.336/93.866/96.865/98.030%**；span外只有7.664/6.134/3.135/1.970%。但该span涵盖51.347/57.573/63.423/75.616%的真实残差能量，远高于R1实际解释份额；它也覆盖任务阶段、环境对运动的响应和移动腕相机看到的背景变化。SigLIP2表示本身有通道相关性，这里没有匹配秩的通用视觉子空间负对照，更没有手臂mask或静止相机反事实。故只能说“E96a的MSE收益高度集中在状态相关的输出方向”，不能写成“92%–98%的收益由手臂像素造成”。
4. **剩余有可预测成分。** 减法剩余上，E96a skill为 **3.495/7.165/6.898/11.651%**，R2为 **7.252/10.763/13.946/17.917%**，总体CI都>0。这等价于超过未来状态R1基线，不另冒充物理分离。更严格移除整个训练状态输出span后，E96a总体skill仍 **0.690/1.673/1.407/2.813%**（CI均>0），R0为 **3.030/3.963/4.840/2.438%**；顶视+10的E96a只有0.091%，CI[−0.356,0.554]跨0。统计剩余不是全部不可预测，但本轮没有证明细小扎带接触/穿孔响应被正确预测。
5. 双腕视角必须单列。腕相机的运动会让静止环境变化，FK末端差又没有完整路径、手眼和深度；不能把腕部状态相关收益当成对象动力学，更不能从该相关性推出“腕部相机自运动占多数”的已测百分比。+100为每camera global，前3个时域为2×4，不能把跨h提升全归因于时长。

### 下一步最小实验设计（本轮不运行）

建议先做两个同预算、同seed、固定终点的直接MSE对照，保持SigLIP2目标、原2×4、归一化、P1窗口、E96a空间旁路和固定人工dev文本。

- A：在E96a上加入同一未来机器人轨迹条件（先用原行实测q作为明确的oracle诊断），预测完整y。
- B：使用与A完全相同的输入/网络/参数预算，唯一改变为预测 y−s(q)，推理加回同一个冻结s(q)。s只从训练parent获得；训练残差采用parent交叉拟合s，dev使用训练全量冻结s；两组沿用原评分尺度，不用dev重估剩余尺度。独立环境残差分数和加回后的完整y分数都报告，避免仅因改变分母看起来更好。
- 两组均预先锁定5000步唯一终点、相同parent采样、相同优化器与损失权重；不按dev选训练长度/超参/ckpt。先用训练内parent验证实现/alpha，dev各只评一次；主比较B−A的配对MSE，逐相机/逐h并列。预计每run约0.4–0.6GPUh，仅规划，实际须先小profile并满足共享显存准入；本轮没有启动。
- 真实部署条件应来自计划的未来机器人轨迹，而非未来实测反馈；若以关节命令代替测量轨迹，须单独核验执行动力学误差。本轮oracle结果只定位视觉目标的可解释性，不承诺部署收益。
- 暂不优先“手臂区域降权”：demo4旧投影尚不足以给稳定区域真值，demo3/demo6无认证外参，2×4格内还混有手臂、扎带和背景。先另立小型独立人工像素核验/相机标定任务，届时再预注册固定区域权重；不能用模型mask作测试真值或dev阈值来源。

### 未完成项与需要用户决定

本轮数值分解、覆盖、FK只读复用、固定dev评估和报告已完成。**未识别**物理手臂像素/环境像素的真实因果能量比例，也未测出腕相机自运动的真实比例；缺少可信mask、手眼/场景几何和独立干预，不能靠本轮统计补造。没有实现投影、重标定、网络训练、实机、仿真、retarget或新模型标注。

需要用户决定的仅是是否安排上述A/B下一轮，或先补独立像素/标定证据；本轮没有执行这些后续工作，无需为当前交付补授权。
'''
with report.open('a') as f:f.write(conclusion)
text=report.read_text().replace('状态：进行中。','状态：完成；无神经网络训练。',1).replace('待核查；缺状态不推测。','已完成，详见第7节及状态恢复回执。',1).replace('待实施；沿 E93 B5 parent 协议。','已完成，详见第8节。',1).replace('待实施；机器人可解释是统计关联，不能未经空间干预等同于手臂像素因果贡献。','已完成，详见第9–10节；统计关联不等于物理手臂像素归因。',1).replace('待结果后追加。','已完成，详见第11节。',1)
report.write_text(text)
gpu=json.loads((R/'gpu_ledger.json').read_text());codebytes=sum(p.stat().st_size for p in (T/'code_e97a').rglob('*') if p.is_file());resultbytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());newbytes=codebytes+resultbytes+report.stat().st_size;assert newbytes+1024**2<5*10**9
resources=f'''
## 12. 命令、资源与最终收据

完成BRT：{now()}。训练岭CPU墙钟294.374s；唯一完成的预测/评分CPU墙钟48.593s（包括数据准备/输出）；额外审计与报告均CPU。GPU尝试仅CUDA设备准入段 **{gpu['elapsed_s']:.6f}s={gpu['GPUh']:.9f}GPUh**，实际分配峰值0；失败段计入≤0.3GPUh。没有GPU前向/训练、OOM、MPS或他人进程操作，之后全部CPU。

新增落盘实测（本目录+代码+本报告，日志及收据预留1MiB）上界 **{(newbytes+1024**2)/1e9:.6f}GB <5GB**。不复制大数据/模型，不下载权重或数据集；旧模型和特征缓存原位只读。数据状态补充5.19MB；dev_forward_snapshot与dev_predictions是小型数值诊断输出，未写回旧缓存。保护清单{len(protected)}个旧源码/固定输入/ckpt SHA变化0；恢复状态的27个原始HDF5 size/mtime变化0。E96a终点SHA={audit['ckpt_sha256']}。R0 dev逐窗MSE最大差={max(x['max_abs_old_mse_diff'] for x in r0audit):.3g}，E96a复推与旧单hMSE最大差<4.4e−7。分块R2闭式解对显式设计矩阵差1.9984e−15；能量加法最大误差{addmax:.3g}，正交能量加法{pmax:.3g}，收益加法{gmax:.3g}。

主命令：`ssh -o BatchMode=yes <gpu-server>`；`cd T/code_e97a`；`setsid nohup env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .../venvs/e29/bin/python -u e97a_core.py > .../fit_01.log 2>&1 < /dev/null &`。最终前向同样分离运行 `E97A_CPU_ONLY=1 CUDA_VISIBLE_DEVICES="" ... e97a_evaluate.py > evaluation_cpu_03.log`。审计：e97a_inventory.py/e97a_timing.py/e97a_restore_states.py；表格与收尾：e97a_report_tables.py/e97a_finalize.py。状态为 FINAL_STATUS.json；唯一dev预测锁 DEV_ONCE_LOCK.json；冻结系数/哈希 FIT_LOCK.json。

保留失败：首次审计脚本名inspect与标准库冲突，改名audit_inputs后成功，0GPUh；首次dev coverage预检因旧图像缓存无状态字段而停止，0模型前向/评分；第二次CUDA admission失败也在0模型前向/评分前停止，GPU段按账本计入。原失败日志、preflight_01_*、admission_02_*完整保留；没有在看dev分数后改alpha/模型/窗口。最终一次CPU预测生成全部841个window×h结果，后续表格仅复用这一次固定输出，不重新推理或选超参。

详细产物：state_coverage_train/dev.json、timing_train.json、calibration_inventory.json、dev_state_restore_receipt.json、fit_method.json、dev_summary.json（完整每相机每h的skill/CI/EV/cos）、energy_decomposition.json、gain_attribution.json、paired_incremental.json、final_audit.json；逐窗数据见dev_records.json，完整向量见dev_predictions_h*.npz。
'''
with report.open('a') as f:f.write(resources)
# Required shared append-only records; fcntl lock prevents interleaving concurrent sessions.
entry=f'''\n## E97-A1 | {now()} BRT | Piper机器人/环境残差统计分解\n- 目标：沿E93 B5比较R0/R1/R2，检查状态/FK/标定覆盖，量化E96a收益的状态相关方向及剩余。\n- 改动：仅新建code_e97a、results/e97a_robot_env_20261114_01、notes/e97a_robot_env_decomp_20261114.md；原缓存/ckpt只读。恢复27dev录制的精确对齐小型状态补充5.19MB，不改窗口/标签。\n- 命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup .../e29/bin/python -u e97a_core.py；E97A_CPU_ONLY=1 CUDA_VISIBLE_DEVICES="" ... e97a_evaluate.py；审计/收尾脚本见报告。\n- 目录：T/code_e97a/；T/results/e97a_robot_env_20261114_01/。\n- 结果：546 capture/242训练parent/107057 P1窗状态覆盖100%；dev27parent/211窗，h100为26/208。R1 skill=0.915/4.747/10.220/26.219%；R2=8.101/14.999/22.741/39.438%。E96a收益92.3–98.0%落在训练状态输出span，但不能等同物理手臂归因；span外E96a skill=0.690/1.673/1.407/2.813%，统计剩余仍可预测。\n- 结论：支持未来轨迹条件与独立剩余目标的下一轮A/B；无可信像素分区，暂不据此上手臂降权。所有alpha仅训练内48parent选，唯一完成dev预测pass。\n- 资源：无神经训练；失败GPU准入{gpu['GPUh']:.9f}GPUh，其余CPU；新增约{(newbytes+1024**2)/1e9:.3f}GB；保护418文件变化0。\n- 状态：COMPLETED_ANALYSIS_ONLY；物理像素归因/投影/标定/下一轮训练未执行，限制详报告。\n'''
for path,txt in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',entry),(T/'notes/worklog.md',f"\n- {now()} BRT E97-A1 完成Piper状态/FK覆盖与R0/R1/R2分解；R2 skill8.10/15.00/22.74/39.44%，E96a状态span外仍0.69–2.81%skill；GPU失败准入{gpu['GPUh']:.9f}h，未训练；报告notes/e97a_robot_env_decomp_20261114.md。\n")]:
 with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(txt);f.flush();os.fsync(f.fileno())
final=dict(status='COMPLETED_ANALYSIS_ONLY',brt=now(),report=str(report),GPUh=gpu['GPUh'],new_disk_upper_bound_bytes=newbytes+1024**2,no_neural_training=True,dev_completed_prediction_passes=1,training_parents=242,dev_parents=27,dev_windows=211,protected_changed=0,unresolved=['physical robot/environment causal pixel energy not identifiable','no certified Piper demo3/demo6 top extrinsic','no arm projection implemented','next A/B only proposed'],report_sha256=sha(report))
save('FINAL_STATUS.json',final);print(json.dumps(final,ensure_ascii=False))
