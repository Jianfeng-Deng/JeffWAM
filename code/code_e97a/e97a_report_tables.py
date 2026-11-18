from e97a_core import *
import fcntl
assert (R/'EVALUATION_COMPLETE.json').exists()
summ=json.loads((R/'dev_summary.json').read_text());energy=json.loads((R/'energy_decomposition.json').read_text());gain=json.loads((R/'gain_attribution.json').read_text());cov=json.loads((R/'state_coverage_train.json').read_text());timing=json.loads((R/'timing_train.json').read_text());devcov=json.loads((R/'state_coverage_dev.json').read_text());method=json.loads((R/'fit_method.json').read_text());records=json.loads((R/'dev_records.json').read_text());receipt=json.loads((R/'EVALUATION_COMPLETE.json').read_text())
def score(h,cam,m,section='full'):return summ[f'{h}|{cam}|{section}|{m}']
def pct(x):return f'{100*x:.3f}'
def ci(r):return f"{pct(r['skill'])} [{pct(r['skill_ci95'][0])}, {pct(r['skill_ci95'][1])}]"
lines=['\n## 7. 最终数据覆盖与对齐核验（完成）\n','训练P1：546 capture /242 parent /107057窗。全部当前关节/夹爪有效；所有合法图像时域端点均有合法状态。状态单位为关节rad、signed raw夹爪m；左右各6关节+1夹爪，无动作数组shift，不使用action作为未来命令。','\n|来源|parent|capture|P1窗|+100合法图像/可配状态|状态配对率（各h）|','|---|---:|---:|---:|---:|---:|']
for src,d in cov['sources'].items():
 cs=[c for c in cov['captures'] if c['source']==src];lines.append(f"|{src}|{len({c['parent'] for c in cs})}|{len(cs)}|{d['P1_windows']}|{d['image_h100']}/{d['paired_h100']}|100%|")
lines+=['\n来源parent可重叠，不应将上表parent相加。+10/25/50合法图像均为107057窗；+100合计'+str(sum(d['image_h100'] for d in cov['sources'].values()))+'窗。训练回归采样2888/2888/2888/2375窗，不把107057称为拟合样本量。','\n|来源|状态显式等待 中位/p95/最大 ms|最大状态插值括号宽度 p95/最大 ms|顶视图像年龄 中位/p95 ms|','|---|---:|---:|---:|']
for src,d in timing.items():
 a=d['state_wait_ms'];b=d['bracket_width_max_ms'];c=d['top_image_age_ms'];lines.append(f"|{src}|{a['median']:.3f}/{a['p95']:.3f}/{a['max']:.3f}|{b['p95']:.3f}/{b['max']:.3f}|{c['median']:.3f}/{c['p95']:.3f}|")
lines+=['\n当前状态以 state_available_time 为已知时刻，不冒充名义t时零延迟可得；历史按原−24/−18/−12/−6源行及有效mask，没有借邻窗补历史。未来q直接读t+h原行，图像端点沿原±2行择最近曝光、误差≤1/60s合同；状态未重采样到曝光时刻，FK不插值投影。','\ndev：27 parent/211窗；+10/25/50均211，+100为26 parent/208窗；状态配对率100%。冻结51名单中的另24个仍不具备E93同合同观测，不扩窗、不修改冻结归属。dev状态恢复的27份独立补充合计5,190,520B，原cache不变。']
lines+=['\n## 8. R0/R1/R2 完整主表（唯一dev评分）\n','skill=1−parent等权MSE/copy能量；EV=1−加权中心化误差方差/中心化目标方差（按固定格/通道跨窗口中心化）；cos为每相机真实与预测残差的方向余弦，先窗后parent等权。EV可不同于skill；copy skill与cos均0。CI为2000次parent bootstrap，seed90。','\n|h|R0 alpha|R1 alpha|R2 alpha|训练窗|R0与E93最大预测差|','|---:|---:|---:|---:|---:|---:|']
for m in method:lines.append(f"|{m['h']}|{m['alphas']['R0']['alpha']}|{m['alphas']['R1']['alpha']}|{m['alphas']['R2']['alpha']}|{m['windows']}|{m['R0_frozen_max_abs_prediction_diff']:.2g}|")
lines+=['\n|相机|h|方法|skill % [95%CI]|EV %|方向cos|','|---|---:|---|---:|---:|---:|']
for cam in ('top','left_wrist','right_wrist','all'):
 for h in H:
  for m in ('R0','R1','R2'):
   r=score(h,cam,m);lines.append(f"|{cam}|{h}|{m}|{ci(r)}|{pct(r['explained_variance'])}|{r['cosine']:.5f}|")
lines+=['\nR0完全沿用E93内层训练PCA64及标准化；R0复拟合与原预测最大差<1e−5。R1是124D机器人量经逐camera/cell岭映射；R2采用共享R0加逐格机器人条件。所有组同一48 parent内层留出，同一6点alpha网格，PCA/标准化都不从dev学习。R1/R2提供的是未来实测轨迹端点已知的诊断条件，不是已发送控制命令，也不等于已部署无未来条件的预测器。']
lines+=['\n## 9. 机器人可解释份额与剩余\n','主口径是状态条件下留出误差减少份额：robot=skill(R1)，remaining=1−skill(R1)。这是当前线性机器人特征族的统计结果，绝不把未解释部分直接宣布为物理环境。非线性机器人外观、遮挡、照明和编码器变化也在remaining内；物体对动作的响应反过来也可被R1解释。','\n|相机|h|robot可解释 % [CI]|remaining %|预测s能量/总能量 %|交叉项/总能量 %|','|---|---:|---:|---:|---:|---:|']
for d in energy:
 r=score(d['h'],d['camera'],'R1');lines.append(f"|{d['camera']}|{d['h']}|{ci(r)}|{pct(d['environment_unexplained_share'])}|{pct(d['robot_prediction_energy_share'])}|{pct(d['cross_share'])}|")
lines+=['\n必须保留交叉项：E||y||²=E||s||²+E||y−s||²+2E<s,y−s>。表中没有裁剪负skill，也没有把不正交的s²和remaining强行归一到100%。','\n腕部图像坐标系随机械臂运动：静止环境也会产生大范围视差/旋转/遮挡变化。状态解释的腕部成分可能主要是相机自运动，而非被夹物形变。本轮没有稳定相机/背景真值或外参支持，不能把“腕部变化大部分是自运动”当已测得的百分比；只能明确这是关键混杂。']
lines+=['\n## 10. 剩余上的copy、R0、E96a与增量条件\n','A：减法剩余 e=y−s，预测m−s；copy定义为零环境增量（相当于只做R1机器人预测），skill=0，R1同样0。m−s对e的误差恒等于m对y的原误差，因此此表检验R0/E96a/R2能否超过已知未来状态的R1，不把它夸称为像素上去除手臂。','\n|相机|h|copy %|R0 residual skill %|E96a residual skill % [CI]|R2 residual skill % [CI]|','|---|---:|---:|---:|---:|---:|']
for cam in ('top','left_wrist','right_wrist','all'):
 for h in H:
  a=score(h,cam,'R0','environment_subtraction');b=score(h,cam,'E96a','environment_subtraction');c=score(h,cam,'R2','environment_subtraction');lines.append(f"|{cam}|{h}|0|{pct(a['skill'])}|{ci(b)}|{ci(c)}|")
lines+=['\nB：正交子空间补充。Q仅由锁定训练R1系数/截距构造，各格单独投影；QQᵀy与(I−QQᵀ)y正交，其能量相加为100%。这是机器人条件映射的输出方向span，可能纳入阶段/环境相关方向，也会保留R1无法准确预测的span内能量；**子空间能量不能冒充R1已解释能量或物理手臂占比**。','\n|相机|h|状态输出span能量 %|span外剩余能量 %|copy剩余skill %|R0剩余skill % [CI]|E96a剩余skill % [CI]|','|---|---:|---:|---:|---:|---:|---:|']
for d in energy:
 h=d['h'];cam=d['camera'];a=score(h,cam,'R0','environment_subspace');b=score(h,cam,'E96a','environment_subspace');lines.append(f"|{cam}|{h}|{pct(d['robot_subspace_energy_share'])}|{pct(d['environment_subspace_energy_share'])}|0|{ci(a)}|{ci(b)}|")
lines+=['\nE96a相对copy的MSE改进可在上述正交span内/外精确相加。份额允许为负或>100%，不能裁剪；这量化关联方向上的收益分布，不能识别真实手臂像素因果占比。','\n|相机|h|E96a全量skill %|总收益中的span内 %|总收益中的span外 %|','|---|---:|---:|---:|---:|']
for d in gain:lines.append(f"|{d['camera']}|{d['h']}|{pct(score(d['h'],d['camera'],'E96a')['skill'])}|{pct(d['robot_subspace_gain_share'])}|{pct(d['environment_subspace_gain_share'])}|")
# paired incremental comparisons from the same prediction pass
pair=[]
for h in H:
 for cam in ('top','left_wrist','right_wrist','all'):
  rr=[r for r in records if r['h']==h and r['camera']==cam and r['section']=='full'];ps=sorted({r['parent'] for r in rr});a=[]
  for p in ps:
   means={m:np.mean([r['model'] for r in rr if r['parent']==p and r['method']==m]) for m in ('R0','R1','R2')};cp=np.mean([r['copy'] for r in rr if r['parent']==p]);a.append([means['R0'],means['R1'],means['R2'],cp])
  a=np.array(a);dr=a[np.random.default_rng(90).integers(len(a),size=(2000,len(a)))].mean(1);mean=a.mean(0)
  pair.append(dict(h=h,camera=cam,R2_minus_R0_skill=(mean[0]-mean[2])/mean[3],R2_minus_R0_ci95=np.quantile((dr[:,0]-dr[:,2])/dr[:,3],[.025,.975]).tolist(),R2_minus_R1_skill=(mean[1]-mean[2])/mean[3],R2_minus_R1_ci95=np.quantile((dr[:,1]-dr[:,2])/dr[:,3],[.025,.975]).tolist()))
save('paired_incremental.json',pair)
lines+=['\n### 配对增量（总体，百分点）\n','|h|R2−R0 [95%CI]|R2−R1 [95%CI]|','|---:|---:|---:|']
for d in pair:
 if d['camera']=='all':lines.append(f"|{d['h']}|{pct(d['R2_minus_R0_skill'])} [{pct(d['R2_minus_R0_ci95'][0])},{pct(d['R2_minus_R0_ci95'][1])}]|{pct(d['R2_minus_R1_skill'])} [{pct(d['R2_minus_R1_ci95'][0])},{pct(d['R2_minus_R1_ci95'][1])}]|")
report=T/'notes/e97a_robot_env_decomp_20261114.md'
with report.open('a') as f:f.write('\n'.join(lines)+'\n')
save('tables_written.json',dict(brt=now(),report=str(report),lines=len(lines)))
print(json.dumps(dict(status='tables_written',lines=len(lines),all_camera=[dict(h=h,**{m:score(h,'all',m)['skill'] for m in ('R0','R1','R2','E96a')}) for h in H])))
