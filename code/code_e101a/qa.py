import os,sys,fcntl
sys.dont_write_bytecode=True
sys.path.insert(0,'/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e101a')
from diag import *
sm=json.loads((OUT/'sampling_metrics.json').read_text());pm=json.loads((OUT/'probe_metrics.json').read_text());pr=np.load(OUT/'predictions.npz');out={};summary={}
for name in ('dev','train64'):
 d=np.load(OUT/(name+'.npz'));y=d['y'];q=d['q'];p=d['parents'];avg=lambda x:np.mean([x[p==pp].mean(0) for pp in sorted(set(p))],0)
 for s in (4,8,16):
  a=pr[f'{name}_S{s}_N8'];m=pr[f'{name}_S{s}_N16'];b=2*m-a
  for h in (8,50):
   model=avg(((m[:,:h]-y[:,:h])**2).mean(1));v16=avg(((a[:,:h]-m[:,:h])**2).mean(1));limit=model-v16;copy=avg(((q[:,None]-y[:,:h])**2).mean(1));g={}
   for group,ix in [('joint',JOINTS),('gripper',GRIPS)]:g[group]=dict(N16_mse=float(model[ix].mean()),variance_of_N16_mean_estimate=float(v16[ix].mean()),infinite_mean_mse_unbiased_estimate=float(limit[ix].mean()),copy_mse=float(copy[ix].mean()),estimated_limit_skill=float(1-limit[ix].mean()/copy[ix].mean()))
   out[f'{name}_S{s}_h{h}']=g
 for method,mm in sm[name].items():
  summary[name+'_'+method]={h:dict(joint_L1_skill=1-v['joint_L1_rad']/v['copy_joint_L1_rad'],gripper_L1_skill=1-v['gripper_L1_mm']/v['copy_gripper_L1_mm'],joint_MSE_skill=v['joint_skill'],gripper_MSE_skill=v['gripper_skill']) for h,v in mm.items()}
for method,mm in pm.items():summary[method]={h:dict(joint_L1_skill=1-v['joint_L1_rad']/v['copy_joint_L1_rad'],gripper_L1_skill=1-v['gripper_L1_mm']/v['copy_gripper_L1_mm'],joint_MSE_skill=v['joint_skill'],gripper_MSE_skill=v['gripper_skill']) for h,v in mm.items()}
save('sampling_noise_decomposition.json',out);save('relative_copy_summary.json',summary)
lines=[]
for name in ('dev','train64'):
 for method in ('S4_N1','S4_N8','S16_N16','S4_zero'):
  mm=summary[name+'_'+method];lines.append(f'|{name} {method}|'+'|'.join(f"{100*mm[h][k]:.1f}" for h in ('8','50') for k in ('joint_L1_skill','gripper_L1_skill'))+'|')
report('## 最终数值复核与解释补充\n### L1相对复制改善率（避免和MSE skill混淆）\n定义1−模型parent等权L1/复制parent等权L1，正为好；完整探针同指标见relative_copy_summary.json。\n|条件|前8关节%|前8夹爪%|H50关节%|H50夹爪%|\n|---|---:|---:|---:|---:|\n'+'\n'.join(lines)+'\n\n训练64的S4N1：关节前8/H50=.01136/.06917rad，复制=.00761/.03803；夹爪3.085/6.203mm，复制=.575/3.222。N8后关节=.00915/.04489、夹爪3.204/4.790，也都未胜L1复制。因此不能把主问题单独归为dev泛化失败：训练parent分布也存在动作FM/采样误差或条件不足。零起点关节H50训练=.03561可胜.03803、dev=.05973仅略胜.05996；这是诊断中的局部例外，不能宣称神经网络任何条件都输复制，也不能以dev选出zero4作新基线。前8夹爪所有这些条件仍明显劣复制。训练FM后2000步斜率−.01014/1000仍下降，支持继续优化有空间，但不证明加步数就能解决。\n\n采样步数4→8→16在同噪声下L1普遍恶化，反对“仅Euler离散化太粗”的解释。N8→N16不保证当前48窗单组16seed上单调；没有以观察到的最佳N选型。另把前8样本均值A与后8均值B=2*N16−N8分组，估计N16均值方差为(A−N16)^2，极限均值MSE无偏估计=N16_MSE−该方差；该有限MC估计可负、不裁剪，不是实测无限N性能，完整值见sampling_noise_decomposition.json。\n\n夹爪0.5mm口径：train逐帧静止左/右98.01%/91.90%，dev96.96%/92.13%；dev前8相对锚点可保持95.05%/87.24%。S4N1前8静止误差质量占90.14%/79.72%，事件±5帧占19.33%/26.71%；H50静止占70.17%/60.34%，事件附近仅14.17%/20.68%（事件邻域总覆盖6.71%/14.92%，覆盖率基于全H50）。误差并非主要集中开合事件，保持头的价值更直接；完美事件门在前8把L/R误差3.733/4.790降至.375/.986mm，但平均仍约.680mm>copy .385，说明仅正确识别变化仍不能替代幅值/时机学习。神谕门+真幅值的平均约.0104mm是理论上限，不是可部署成绩。\n\n初始报告骨架时间取整有偏差，实际开工命令服务器15:00:37 CST，即2026-11-15 04:00:37 BRT；所有运行状态时间均由UTC−3生成。最终源/ckpt SHA复核通过，所有统计有限，GPUworker已结束。')
# Numeric and shape assertions.
for k in pr.files:assert pr[k].shape==((48 if k.startswith('dev_') else 64),50,14) and np.isfinite(pr[k]).all(),k
def walk(v):
 if isinstance(v,dict):
  for x in v.values():walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
 elif isinstance(v,(int,float)):assert np.isfinite(v)
for f in ('sampling_metrics.json','probe_metrics.json','t0_metrics.json','contract_audit.json','input_ablation_metrics.json','gripper_statistics.json','gripper_oracle_gates.json','sampling_noise_decomposition.json'):walk(json.loads((OUT/f).read_text()))
for p,h in json.loads((OUT/'provenance.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
size=sum(p.stat().st_size for root in (OUT,T/'code_e101a') for p in root.rglob('*') if p.is_file());assert size<10*1024**3
cp=json.loads((OUT/'completion.json').read_text());cp.update(new_bytes=size,qa_brt=now(),numeric_shape_checks='passed',plots_visually_checked='sampling_curve.png');save('completion.json',cp)
with (T/'docs/experiment_log/EXPERIMENT_LOG.md').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(f'\nE101-A 最终复核 {now()} BRT：训练64 N1/N8同样劣L1复制；zero4关节H50有局部微胜，未作为选型。dev夹爪静止段贡献前8误差90%/80%，并非主要集中开合；数值/shape与只读SHA复核通过，新增{size/1024**3:.4f}GiB，GPUh不变。\n')
status('complete',new_GiB=size/1024**3)
