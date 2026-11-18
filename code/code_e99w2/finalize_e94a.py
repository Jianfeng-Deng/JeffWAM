import csv,json,hashlib,fcntl,collections
from pathlib import Path
import numpy as np
from e94a.common import T,ART,OUT,sha,atomic,brt,report
E93=T/'results/e93_dev_diag_20261114_01';H=(10,25,50,100)
assert json.loads((ART/'pipeline_status.json').read_text())['status']=='completed'
assert json.loads((ART/'ridge_train64_status.json').read_text())['status']=='completed'
ready=json.loads((OUT/'READY_readout.json').read_text());assert all(sha(OUT/p)==h for p,h in ready['sha256'].items())
before=json.loads((ART/'protected_before.json').read_text());changes=[p for p,h in before.items() if sha(Path(p))!=h];assert not changes,changes
runs={a:T/'results'/f'e94a_{a.lower()}_p1_20261114_01' for a in ('M1','M2')};sums={};cps=[];curves=[]
for a,r in runs.items():
 s=json.loads((r/'train_summary.json').read_text());assert s['steps']==5000;assert s['sampler']==json.loads((T/'code_e90/artifacts/e90_p1_20261114_01/train_summary.json').read_text())['sampler'];sums[a]=s
 updates=[json.loads(l) for l in (r/'updates.jsonl').read_text().splitlines()];assert len(updates)==5000
 for u in updates:curves.append(dict(recipe=a,**{k:u[k] for k in ('step','loss','mean','fm','cosine_short','adapter','chain','chain_weight','seconds')}))
 for cp in sorted(r.glob('ckpt_*.pt')):cps.append(dict(recipe=a,path=str(cp),bytes=cp.stat().st_size,sha256=sha(cp)))
 assert len(list(r.glob('ckpt_*.pt')))==10
atomic(ART/'checkpoint_manifest.json',cps)
with (ART/'training_curves.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
ledger=json.loads((ART/'gpu_ledger.json').read_text());assert not ledger.get('active') and ledger['elapsed_s']<=7200
paths=[OUT,ART,*runs.values()];nbytes=sum(p.stat().st_size for root in paths for p in root.rglob('*') if p.is_file());assert nbytes+1050000000<=10e9
methods={};allresults={}
for pool in ('train64','dev'):
 d={}
 for a,r in runs.items():d.update(json.loads((r/(pool+'_summary.json')).read_text()))
 d['ckpt1_FM_N1']=json.loads((E93/(pool+'_v1_summary.json')).read_text())
 grid=json.loads((E93/(pool+'_grid_summary.json')).read_text());d['ckpt1_FM_N16']=grid['E4_N16']
 d['ridge_E93']=json.loads(((ART/'ridge_train64_summary.json') if pool=='train64' else (E93/'linear_summary.json')).read_text())
 allresults[pool]=d
checks=[]
for pool,d in allresults.items():
 for h in (10,25,50):
  m=d['M1_mean'][str(h)];ridge=d['ridge_E93'][str(h)]['values']['skill_copy'];v=m['values']['skill_copy'];ci=m['ci95']['skill_copy']
  checks.append(dict(pool=pool,h=h,skill=v,ci95=ci,ridge_skill=ridge,fraction_of_ridge=v/ridge,positive_ci=ci[0]>0,at_least_half_ridge=v>=.5*ridge,passed=ci[0]>0 and v>=.5*ridge))
passed=all(c['passed'] for c in checks);atomic(ART/'repair_decision.json',dict(preregistered=True,criterion='M1 train64 and dev +10/+25/+50: paired parent95CI lower>0 and point skill>=50% E93 ridge skill, all six cells',checks=checks,readout_repaired=passed))
receipt=dict(status='completed',brt=brt(),GPU_seconds=ledger['elapsed_s'],GPUh=ledger['elapsed_s']/3600,disk_bytes=nbytes,disk_GB=nbytes/1e9,removed_local_temporary_bytes=1046032857,local_write_upper_bound_bytes=1050000000,cumulative_disk_upper_bound_bytes=nbytes+1050000000,protected_files=len(before),protected_changes=changes,ready_sha256=sha(OUT/'READY_readout.json'),tool_sha256={n:sha(OUT/n) for n in ('eval_e94a.py','ridge_train64.py','pipeline_e94a.py','finalize_e94a.py')},original_tests=133,new_tests=16,tests=149,runs=sums,readout_repaired=passed)
atomic(ART/'final_resource_receipt.json',receipt);atomic(ART/'comparison_summary.json',allresults)
lines=['\n## 固定5000步训练完成',f'完成BRT：{brt()}。两配方P1均为107057窗/242训练parent；从相同seed90与相同Piper采样器起训，最终160000次抽样的采样器状态与E90逐字段一致，batch32、AdamW3e-4/1e-4、clip1，固定5000终点；每500保存，共20份完整checkpoint。无dev选型、无其他训练。',
'均值查询/MLP与主目标改变；其余短余弦.1、适配器.1、3000前无链、3000–4000链0→.1、最后1000保持.1、训练parent门槛与采样不变。M2两个主损失各1；M1没有FM前向。可学习查询为mean_query+连续时域MLP+空间位置。零初始化末层，随后可学习为超过256维的输出仿射跨度；共享因果表示宽度仍256，不声称消除一切信息压缩。',
'|配方|首32/末32总loss|链有效步|最后门检NMSE/cos|可训练参数|', '|---|---|---:|---|---:|']
for a,s in sums.items():
 g=s['adapter_last'];lines.append(f'|{a}|{s["first32"]:.5f}/{s["last32"]:.5f}|{s["chain_updates"]}|{g["nmse"]:.6f}/{g["cosine"]:.6f}|{s["trainable_parameters"]}|')
lines += ['\n## 统一评估与E93并列','同一E93固定窗口，dev211/27（100为208/26），train64 507/64（100为396/61）；每次单时域查询，与E93主表一致。训练统计归一化残差MSE，parent内先平均再parent等权；2000次配对parent bootstrap，seed90。确定性均值头直接预测；FM固定4Euler、CPU生成噪声seed9001+原窗index+100000×样本index，N1与同16样本池均值。dev仅固定人工短指令；train64也统一短指令。线性外推数值复用逐窗完全匹配的E93记录，copy逐窗误差断言<1e-5；旧ckpt数字为E93原记录，不重评/替换。',
'目标与归一化未改变，因此复用E93 B5原ridge：242train parent各最多12窗；内层48train parent/seed9305选alpha，PCA64仅内层fit；alpha候选1e-4/1e-3/.01/.1/1/10，各时域选.1，再全242训练parent拟合。dev只评过E93原那一次，本轮直接引用；新增train64评价只读取同一冻结系数，属于训练parent内参照。原方法/系数sha见ridge_train64_status.json。不同训练采样和输入组织使ridge不是纯头结构消融。']
for pool,d in allresults.items():
 lines += [f'\n### {pool}', '|方法|+帧|窗/parent|MSE|copy MSE|线性外推MSE|skill % [95% CI]|残差cos|','|---|---:|---|---:|---:|---:|---|---:|']
 for method in ('ckpt1_FM_N1','ckpt1_FM_N16','ridge_E93','M1_mean','M2_mean','M2_FM_N1','M2_FM_N16'):
  for h in H:
   x=d[method][str(h)];v=x['values'];ci=x['ci95']['skill_copy'];lin=v.get('linear',d['M1_mean'][str(h)]['values']['linear'])
   lines.append(f'|{method}|{h}|{x["windows"]}/{x["parents"]}|{v["model"]:.6f}|{v["copy"]:.6f}|{lin:.6f}|{100*v["skill_copy"]:.3f} [{100*ci[0]:.3f}, {100*ci[1]:.3f}]|{v["cosine"]:.5f}|')
lines += ['\n## 相机与真实变化量分解','以下为确定性头的skill点估计%；每组MSE/copy/配对95%区间及M2 FM N1/N16分解完整见各run的{train64,dev}_decomposition.json。真实变化量按各池/时域真实归一化残差能量四分位，仅作描述，不用于阈值、训练或checkpoint选择。','|池|方法|+帧|top|left_wrist|right_wrist|Q1|Q2|Q3|Q4|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for pool in ('train64','dev'):
 for a,r in runs.items():
  dc=json.loads((r/(pool+'_decomposition.json')).read_text())[a+'_mean']
  for h in H:
   z=dc[str(h)]['groups'];vals=[100*z[g]['values']['skill_copy'] for g in ('top','left_wrist','right_wrist','change_Q1','change_Q2','change_Q3','change_Q4')]
   lines.append('|'+ '|'.join([pool,a,str(h),*[f'{v:.3f}' for v in vals]])+'|')
lines += ['\n## 预注册修复判定','判定：**'+('读出问题已修复' if passed else '未达到“读出问题已修复”的预注册标准')+'**。必须M1在train64与dev的10/25/50六格同时CI下界>0且点估计≥对应ridge的50%。逐格结果如下；未依据结果追加训练或改配方。','|池|+帧|M1 skill %|95%下界 %|ridge skill %|达到ridge %|通过|','|---|---:|---:|---:|---:|---:|---|']
for c in checks:lines.append(f'|{c["pool"]}|{c["h"]}|{100*c["skill"]:.3f}|{100*c["ci95"][0]:.3f}|{100*c["ridge_skill"]:.3f}|{100*c["fraction_of_ridge"]:.1f}|{c["passed"]}|')
lines += ['\n## 资源、保护与最终交付',f'总GPU墙钟 {ledger["elapsed_s"]:.3f}s = {ledger["elapsed_s"]/3600:.6f}GPUh≤2，含加载/I/O与失败段；独立账本gpu_ledger.json，所有段已关闭。每段准入free≥8192MiB×1.5，监测共享余量与实测peak≤8192MiB，無MPS/他人进程操作。新增产物 {nbytes:,} B（{nbytes/1e9:.4f}GB）≤10GB；缓存原位只读，无新特征/数据集或预训练权重下载。初次取源码误带E90 artifacts，在本地临时目录产生1,046,032,857B（含12个大文件）的副本，已删除，服务器原件不变；本地源码/临时副本累计按1.05GB保守计入，连同服务器产物仍≤10GB。此为已纠正的复制规则偏离，不能记为完全未复制大文件。保护清单{len(before)}文件变化0，READY v1绑定哈希全部一致。',
'代码：code_e94a/；READY：code_e94a/READY_readout.json。结果：results/e94a_m1_p1_20261114_01、results/e94a_m2_p1_20261114_01；汇总/台账/曲线/20checkpoint清单：results/e94a_control_20261114_01。每配方ckpt_05000.pt是唯一评估终点；完整SHA256见checkpoint_manifest.json。命令：ssh -o BatchMode=yes <gpu-server>；cd T/code_e94a；setsid nohup env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u pipeline_e94a.py；CPU ridge_train64.py；finalize_e94a.py。',
'\n## 未完成项与需要用户决定','本轮指定实现、149合同、两配方5000步及统一评估已完成。冻结51parent中仅27有同合同可评图像/窗口，沿用E93边界，不补造另外24parent；单seed、固定目标指标不代表实机动作或分布校准。未训练天机、P3、ABC/yam或动作头，不启动其他训练。本轮执行无新增用户决定；后续是否继续优化，由本轮预注册判定及目标选择另行安排。']
report('\n'.join(lines));now=brt();skills={pool:{a:[100*allresults[pool][a+'_mean'][str(h)]['values']['skill_copy'] for h in H] for a in ('M1','M2')} for pool in ('train64','dev')}
entry=f'\n## E94a — {now} BRT — 完成\n目标：不变SigLIP2 2×4目标/归一化，修复读出并对照M1/M2。改动：新code_e94a，双MLP、确定性时域查询、均值链；READY v1冻结，原133+新增16=149合同通过。命令：setsid nohup e29/python -u code_e94a/pipeline_e94a.py；ridge_train64.py；finalize_e94a.py。目录：results/e94a_control_20261114_01、e94a_m1_p1_20261114_01、e94a_m2_p1_20261114_01；报告notes/e94a_readout_20261114.md。结果：两配方各5000步、20ckpt，统一E93 dev/train64、4Euler FM N1/N16、2000次parent CI及相机/四分位分解齐备；均值skill%={json.dumps(skills,ensure_ascii=False)}。结论：预注册读出修复通过={passed}。资源：{receipt["GPUh"]:.6f}GPUh、{receipt["disk_GB"]:.4f}GB，保护{len(before)}文件变化0。状态：COMPLETE；无其他训练，无待决定的执行事项。\n'
for p,s in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',entry),(T/'notes/worklog.md',f'\n- {now} BRT E94a完成：149合同、M1/M2各5000步，修复标准通过={passed}；{receipt["GPUh"]:.6f}GPUh/{receipt["disk_GB"]:.4f}GB，见notes/e94a_readout_20261114.md。\n')]:
 with p.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(s)
atomic(ART/'FINAL_STATUS.json',dict(status='completed',brt=brt(),readout_repaired=passed,skills=skills,receipt=str(ART/'final_resource_receipt.json')))
print(json.dumps(dict(status='completed',skills=skills,readout_repaired=passed,GPUh=receipt['GPUh'],disk_GB=receipt['disk_GB']),ensure_ascii=False))

text=(T/'notes/e94a_readout_20261114.md').read_text().replace('状态：检查与实现。','状态：完成；固定5000步与统一评估结果见下文最终章节。',1).replace('待完成。','已完成，见后续追加章节。')
(T/'notes/e94a_readout_20261114.md').write_text(text)
