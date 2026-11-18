import json,sys,collections,fcntl
from pathlib import Path
import numpy as np
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e93_dev_diag_20261114_01';sys.path.insert(0,str(T/'code_e90'))
from evaluate_cpu_tail import aggregate
H=(10,25,50,100)
def agg(rs):
 if not rs:return dict(windows=0,parents=0,values=None,ci95=None)
 a=aggregate([dict(r,linear=r['copy']) for r in rs],['model','copy','linear'],2000)
 for k in ('linear','skill_linear'):a['values'].pop(k,None);a['ci95'].pop(k,None)
 return a
for name in sys.argv[1:]:
 marker=R/(name+'_grid_report_done.json');assert not marker.exists()
 grid=json.loads((R/(name+'_grid_records.json')).read_text());s=json.loads((R/(name+'_grid_summary.json')).read_text());base=json.loads((R/(name+'_v1_records.json')).read_text());bym={(r['index'],r['h']):r for r in base};g1=[r for r in grid if r['Euler']==4 and r['N']==1];diff=max(abs(r['model']-bym[r['index'],r['h']]['model']) for r in g1);g16=[r for r in grid if r['Euler']==4 and r['N']==16];means={(r['index'],r['h']):r for r in g16}
 lines=[f'\n## B2 {name} Euler×样本均值完整网格',f'固定ckpt-1，CUDA float32前向；每个样本噪声由CPU按seed=9001+原窗index+100000×样本index生成，与CPU v1的第1样本相同。均值使用同一16样本池的前1/8/16样本，所有格子预先固定，没有择优替换v1。4Euler N=1相对全部对应CPU v1记录的最大MSE差={diff:.9g}；后端误差远小于报告差异。', '|+帧|Euler|N|窗/parent|MSE|copy MSE|skill % [95% parent CI]|残差方向cos|','|---|---:|---:|---|---:|---:|---|---:|']
 for h in H:
  for e in (4,8,16):
   for n in (1,8,16):
    z=s[f'E{e}_N{n}'][str(h)];v=z['values'];ci=z['ci95']['skill_copy'];lines.append(f'|{h}|{e}|{n}|{z["windows"]}/{z["parents"]}|{v["model"]:.6f}|{v["copy"]:.6f}|{v["skill_copy"]*100:.3f} [{ci[0]*100:.3f}, {ci[1]*100:.3f}]|{v["cosine"]:.5f}|')
 lines += [f'\n## B4 {name} 分解：v1与4Euler/16样本均值','以下分组用同一窗口集、训练统计归一化坐标。v1为CPU精确口径；N16为CUDA float32（上述全记录后端核对）。skill用组内parent等权MSE比值；真实变化量为真实归一化残差均方能量，按各时域/池的描述性四分位切分，不用于阈值、checkpoint或训练选择。','|分组|+帧|窗/parent|copy MSE|v1 skill %|N16 skill %|','|---|---:|---|---:|---:|---:|']
 decomps={};bounds={}
 for h in H:
  rs=[r for r in base if r['h']==h];qs=np.quantile([r['change'] for r in rs],[.25,.5,.75]);bounds[str(h)]=qs.tolist()
  def transform(r,which=None,j=None):
   rr=dict(r)
   if which=='camera':rr.update(model=r['model_camera'][j],copy=r['copy_camera'][j])
   if which=='token':
    nt=8 if h<=50 else 1;rr.update(model=float(np.mean(np.array(r['model_token']).reshape(3,nt)[:,j])),copy=float(np.mean(np.array(r['copy_token']).reshape(3,nt)[:,j])))
   return rr
  groups=[]
  for j,cam in enumerate(('top','left_wrist','right_wrist')):groups.append((cam,rs,'camera',j))
  for demo in ('demo3','demo4','demo6'):groups.append((demo,[r for r in rs if r['demo']==demo],None,None))
  for q in range(4):groups.append((f'change_Q{q+1}',[r for r in rs if int(np.searchsorted(qs,r['change'],side='right'))==q],None,None))
  groups.append(('fine_2x4' if h<=50 else 'coarse_global',rs,None,None))
  if h<=50:
   for j in range(8):groups.append((f'fine_pos{j}_across_cameras',rs,'token',j))
  for label,rrs,which,j in groups:
   a=agg([transform(r,which,j) for r in rrs]);b=agg([transform(means[r['index'],r['h']],which,j) for r in rrs]);decomps[f'{h}:{label}']=dict(v1=a,N16=b)
   if not a['windows']:lines.append(f'|{label}|{h}|0/0|NA|NA|NA|');continue
   lines.append(f'|{label}|{h}|{a["windows"]}/{a["parents"]}|{a["values"]["copy"]:.6f}|{100*a["values"]["skill_copy"]:.3f}|{100*b["values"]["skill_copy"]:.3f}|')
 lines += ['细token的pos0–7为同一2×4池化网格按行优先位置，按三相机平均；+100为每路8token的global平均，细/粗与时域同时变化，不能由该表单独断言粒度因果效应。所有分组配对bootstrap区间保存在decomposition JSON，主表为点估计。']
 # Error = copy + predicted-energy -2 dot; quantifies noisy samples without assuming model is calibrated.
 lines += [f'\n### {name} 误差分解（4Euler）','归一化坐标逐窗满足 MSE−copy = E(pred_delta²)−2E(pred_delta·true_delta)，再parent等权。','|+帧|N|预测残差能量|预测×真实内积|MSE−copy|','|---|---:|---:|---:|---:|']
 energy={}
 for h in H:
  for n in (1,8,16):
   rr=[r for r in grid if r['h']==h and r['Euler']==4 and r['N']==n];parents=sorted({r['parent'] for r in rr});av=lambda k:float(np.mean([np.mean([r[k] for r in rr if r['parent']==p]) for p in parents]));en=av('prediction_energy');dot=av('prediction_truth_dot');gap=av('model')-av('copy');energy[f'{h}:N{n}']=dict(prediction_energy=en,dot=dot,error_excess=gap,identity_max_rounding_error=abs(gap-en+2*dot));lines.append(f'|{h}|{n}|{en:.6f}|{dot:.6f}|{gap:.6f}|')
 joint=json.loads((R/(name+'_joint_records.json')).read_text());js={f'E{e}_N{n}':{str(h):agg([r for r in joint if r['Euler']==e and r['N']==n and r['h']==h]) for h in H} for e,n in ((4,1),(16,16))}
 lines += [f'\n### {name} 同时查询固定四时域的额外诊断','训练实际为固定四时域再加一个随机连续偏移；此处只同时查询固定10/25/50/100，因此仍不是完全相同训练布局。联合查询在同一CPU seed上连续抽取全部token噪声，除第一个时域外并非与单查询逐token共同随机数配对；差异不能纯因果归因于attention布局。没有真未来latent作输入，缺失端点按原mask。','|+帧|joint 4Euler N1 skill %|joint 16Euler N16 skill %|','|---|---:|---:|']
 for h in H:lines.append(f'|{h}|{100*js["E4_N1"][str(h)]["values"]["skill_copy"]:.3f}|{100*js["E16_N16"][str(h)]["values"]["skill_copy"]:.3f}|')
 (R/(name+'_decomposition.json')).write_text(json.dumps(dict(groups=decomps,change_quantile_boundaries=bounds,energy=energy,joint=js,backend_max_MSE_difference=diff),indent=2))
 with (T/'notes/e93_dev_diag_20261114.md').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'.join(lines)+'\n')
 marker.write_text(json.dumps(dict(status='completed',backend_max_MSE_difference=diff,groups=len(decomps))))
 print(name,'reported',len(grid),'records',len(decomps),'groups', 'backend_diff',diff)
