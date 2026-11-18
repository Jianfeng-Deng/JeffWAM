from pathlib import Path
import json,hashlib,fcntl,datetime,collections
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e98w_control_20261114_01';REPORT=T/'notes/e98w_robot_conditioned_20261114.md'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
def read(n):return json.loads((R/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def append(p,s):
 with p.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(s)
def fmt(v,ci):return f'{v*100:.3f} [{ci[0]*100:.3f}, {ci[1]*100:.3f}]'
def finish():
 assert not (R/'FINAL_STATUS.json').exists();ledger=read('gpu_ledger.json');assert not ledger.get('active');assert ledger['elapsed_s']<=7200
 protected=read('protected_inputs.json');changed=[p for p,v in protected.items() if sha(Path(p))!=v['sha256']];assert not changed,changed
 roots=[T/'code_e98w',*sorted((T/'results').glob('e98w_*'))];added=sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())+REPORT.stat().st_size+2*1024**2;assert added<=8e9
 done=(R/'EVALUATION_COMPLETE.json').exists();status='completed' if done else 'incomplete_resource_profile';lines=[]
 if done:
  s=read('summary.json');paired=read('paired.json');decision=read('decision.json')
  a=json.loads((T/'results/e98w_a_20261114_04/train_summary.json').read_text());b=json.loads((T/'results/e98w_b_20261114_04/train_summary.json').read_text());assert a['steps']==b['steps']==5000 and a['sample_indices_sha256']==b['sample_indices_sha256'] and a['trainable_parameters']==b['trainable_parameters']
  lines += ['\n## 统一评估结果（固定唯一终点）\n','固定dev27 parent/211窗，+100为26/208；train64为64/507，+100为61/396。单h查询；短h原2×4，+100原global，不把粒度差归因为时域。skill=1−parent等权MSE/copy能量；每窗后parent等权，2000次paired parent bootstrap/seed90。同一重抽索引同时用于模型与分母；未按dev选择任何超参/终点。\n','完整y主表，skill与CI为%，方向cos对应该相机或all整个向量。\n','|池|相机|h|方法|skill % [95%CI]|方向cos|\n|---|---|---:|---|---:|---:|\n']
  for pool in ('train64','dev'):
   for cam in ('all','top','left_wrist','right_wrist'):
    for h in (10,25,50,100):
     for m in ('E96a','A','B','R0','R1','R2'):
      z=s[f'{pool}|{h}|{cam}|full|{m}'];lines.append(f'|{pool}|{cam}|{h}|{m}|{fmt(z["skill"],z["skill_ci95"])}|{z["cosine"]:.5f}|\n')
  lines+=['\n### 减法与正交span外剩余\n','减法e=y−s，模型输出m−s，误差与完整y完全相同，分母为E||y−s||²；这是超过R1的能力，不是物理环境mask。正交Q只从冻结训练R1系数和截距构建，固定float32数值秩阈值；评分(I−QQᵀ)y。span相关的物体响应、阶段和腕相机自运动不能当作机器人像素归因。所有方法的MSE/分母/方向cos/CI均见summary.json；以下并列A/B/E96a。\n','|池|相机|h|方法|减法剩余skill % [CI]|span外剩余skill % [CI]|\n|---|---|---:|---|---:|---:|\n']
  for pool in ('train64','dev'):
   for cam in ('all','top','left_wrist','right_wrist'):
    for h in (10,25,50,100):
     for m in ('E96a','A','B'):
      z=s[f'{pool}|{h}|{cam}|subtraction|{m}'];v=s[f'{pool}|{h}|{cam}|outside_span|{m}'];lines.append(f'|{pool}|{cam}|{h}|{m}|{fmt(z["skill"],z["skill_ci95"])}|{fmt(v["skill"],v["skill_ci95"])}|\n')
  lines+=['\n### dev配对改善，单位pp\n','|相机|h|比较|完整y Δpp [95%CI]|span外 Δpp [95%CI]|\n|---|---:|---|---:|---:|\n']
  for cam in ('all','top','left_wrist','right_wrist'):
   for h in (10,25,50,100):
    for comparison in ('A-E96a','B-A','B-E96a'):
     z=paired[f'dev|{h}|{cam}|full|{comparison}'];v=paired[f'dev|{h}|{cam}|outside_span|{comparison}'];lines.append(f'|{cam}|{h}|{comparison}|{fmt(z["delta_skill"],z["ci95"])}|{fmt(v["delta_skill"],v["ci95"])}|\n')
  lines+=['\n### 预注册机械判定\n',f'未来轨迹条件有效（+25/+50/+100完整skill A−E96a CI下界均>0）：**{decision["future_condition_effective"]}**。显式分离额外收益（对应h span外B−A CI下界>0，且完整B−A CI下界≥0）：**{decision["separation_extra_benefit"]}**。各h细项见decision.json；不以+10、单相机或训练集补偿失败h，不挑选终点。\n',f'A/B各5000步、160000抽样；参数均{a["trainable_parameters"]}，原E96a9539631（机器人投影增加32000）。实际采样SHA一致：{a["sample_indices_sha256"]}。有效链A/B={a["chain_updates"]}/{b["chain_updates"]}；adapter门与链暴露可能因不同目标而变化，门/.1权重/调度未改，不补训练。\n','A新增124D条件同时包含当前、历史与未来实测状态；A−E96a检验整组轨迹条件，不单独识别未来端点相对仅当前状态的增益。实测q是oracle，部署时必须改计划轨迹并另核验跟踪误差；单seed的parent CI不包含训练seed不确定性。SigLIP2代理误差不等于扎带穿孔、接触准确度或实机成功率，未识别物理手臂/环境像素因果份额。当前指定A/B与评估已完成，无需新增用户决定；多seed、计划轨迹迁移与物理mask应另轮预注册。\n']
 else:
  lines+=['\n## 当前交付：资源约束下未完成\n','已完成新包、未来实测q/FK条件、parent交叉拟合R1与CPU合同；预算内profile不足以安全保证两组5000步与统一评估，已停止。未生成dev预测/指标，不能判定A/B有效性；未用训练损失替代最终评估、未改5000步凑结果。失败日志/状态/账本保留。\n','未完成：A/B固定5000终点、dev27/train64统一评分、配对CI与预注册结论。需要用户决定：安排更空闲GPU时段继续同一设计，或明确授权另轮预算；本轮不自行放宽预算或改设计。实测未来q仍仅为oracle，部署差距未解决。\n']
 lines+=['\n## 最终资源与只读审计\n',f'BRT完成/停止：{now()}。GPU累计含失败/加载/profile/前向：{ledger["elapsed_s"]:.3f}s = {ledger["elapsed_s"]/3600:.6f}GPUh ≤2；全部段关闭。新增落盘保守上界{added/1e9:.6f}GB ≤8（含2MiB日志/报告余量）。{len(protected)}个旧源码/冻结系数/窗口/ckpt保护文件SHA变化0；E96a终点SHA=e55c33857e59aaceef559a510e874c602d3827e232cf5817e5edeb7db6fc9a98。未写同事/dataset目录、未操作他人进程、未下载大权重/数据、未用MPS/仿真/retarget、未增加dev模型文本。\n','命令：ssh -o BatchMode=yes <gpu-server>；T下setsid nohup env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u code_e98w/e98w_prepare.py / e98w_run.py / e98w_evaluate.py；CPU e98w_finalize.py。所有产物在code_e98w与results/e98w_*，未覆盖旧run。\n']
 append(REPORT,''.join(lines))
 text=REPORT.read_text().replace('BRT；状态：实现中。','BRT；状态：'+('完成。' if done else '资源约束下未完成。')).replace('待追加。','已写入后续详细收据。')
 REPORT.write_text(text)
 receipt=dict(status=status,brt=now(),GPU_seconds=ledger['elapsed_s'],GPUh=ledger['elapsed_s']/3600,added_bytes_upper=added,protected_files=len(protected),protected_sha_changes=changed,report=str(REPORT),dev_evaluated=done)
 if done:receipt['decision']=read('decision.json')
 (R/'FINAL_STATUS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));(R/'STATUS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
 append(T/'docs/experiment_log/EXPERIMENT_LOG.md',f'\n### E98-W 最终收据 — {now()} BRT\n- 目标/实现/命令/目录见本条初始记录与notes/e98w_robot_conditioned_20261114.md。\n- 结果：状态{status}；GPU含失败{ledger["elapsed_s"]:.3f}s={ledger["elapsed_s"]/3600:.6f}GPUh；新增≤{added/1e9:.6f}GB；保护文件{len(protected)}项SHA变化0。\n- 结论：'+(json.dumps(read('decision.json'),ensure_ascii=False) if done else '共享GPU profile未能满足预留预算；没有dev指标，不作A/B有效性结论。')+'\n')
 append(T/'notes/worklog.md',f'\n- {now()} BRT | E98-W | {status}；GPU {ledger["elapsed_s"]/3600:.6f}h，新增≤{added/1e9:.6f}GB，冻结文件变化0；报告 notes/e98w_robot_conditioned_20261114.md。\n')
 print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':finish()
