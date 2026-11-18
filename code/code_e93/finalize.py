import os,sys,json,time,math,fcntl,subprocess,hashlib,datetime,traceback
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e93_dev_diag_20261114_01';sys.path.insert(0,str(T/'code_e90'))
from e86.common import atomic,sha,brt
import numpy as np,torch
from evaluate_cpu_tail import aggregate
H=(10,25,50,100)
try:
 assert not (R/'E93_FINAL_STATUS.json').exists()
 while True:
  p=R/'diag_status.json'
  if p.exists():
   d=json.loads(p.read_text())
   if d['status']=='completed':break
   if d['status']=='stopped':raise RuntimeError(d)
  time.sleep(5)
 with (R/'train64_grid_report.log').open('x') as f:subprocess.run([sys.executable,'-u','grid_report.py','train64'],stdout=f,stderr=subprocess.STDOUT,check=True)
 for name in ('dev','train64'):
  base=json.loads((R/(name+'_v1_records.json')).read_text());grid=json.loads((R/(name+'_grid_records.json')).read_text());assert len(grid)==len(base)*9
  assert json.loads((R/(name+'_grid_report_done.json')).read_text())['status']=='completed'
 fm=json.loads((R/'fm_diagnostic.json').read_text());summary={pool:{str(h):aggregate([r for r in fm if r['pool']==pool and r['h']==h],['model_fm','zero_velocity','zero_clean_head','oracle_subspace_FM_floor','oracle_subspace_mse','copy'],2000) for h in H} for pool in ('dev','train64')}
 maxt=max(float(torch.sigmoid(torch.randn(16,generator=torch.Generator().manual_seed(9393+i))).max()) for n in (211,507) for i in range(0,n,8));assert maxt<.99
 assert all(r['oracle_subspace_FM_floor']<=r['model_fm']+1e-4 for r in fm)
 atomic(R/'fm_summary.json',dict(pools=summary,max_sampled_tau=maxt,zero_clean_expected_factor=1+2*math.exp(.5)+math.exp(2)))
 lines=['\n## B6 FM损失下界、噪声与固定head约束',
'记真实归一化残差y、独立标准高斯e、w_t=(1−t)e+t y、FM目标v*=y−e。E90采用vθ=(rhatθ−w_t)/max(1−t,.01)。以下是同ckpt的固定窗口/固定四时域联合查询、每窗16个logistic-normal t/噪声的额外前向；不是重训练。训练原配方还抽连续时域，batch/语言/窗口分布也不同，所以不能把本表FM与整轮训练曲线逐点相等对照。',
'1. **速度全零**的FM期望为 E||y−e||²=E||y||²+1（按坐标平均）。这是一个参考函数的损失，不是不可约下界。此函数要求干净头输出w_t，当前256维仿射输出空间一般不能完整表达768维恒等映射。',
'2. **干净残差头全零/复制轨迹**不等于速度全零：v=−w_t/(1−t)，其FM误差为||y||²/(1−t)²。若t=sigmoid(N(0,1))且忽略极罕见的.01截断，倍率期望1+2exp(.5)+exp(2)=11.6865。实际有限MC值列下表。最终复制点预测的MSE则仅为E||y||²。',
'3. 无约束Bayes FM下界为 E Var(y−e | w_t,t,因果观测/文本)，没有通用的“噪声底=1”。若未来残差由观测完全确定，FM可为0；非零/平稳loss既不能直接证明没学会，也不能证明到了数据不可约上限。',
'4. **本checkpoint的可计算下界**：world_head是768×256矩阵，实测数值秩256。其干净残差输出在固定仿射空间b+col(W)内。对本轮4/8/16步Euler，末步dt=1−t（未触及.01floor），所以最终残差恰等于末步rhat，仍在该空间。令Q为col(W)正交基，oracle点误差下界为||(I−QQᵀ)(y−b)||²；同t下FM下界再除(1−t)²。它使用真实y只作结构诊断，不能部署、作checkpoint选择或当数据不可预测性。改变head权重会改变此下界；它不是所有模型/训练的普遍极限。',
f'本轮MC最大t={maxt:.8f}<.99，分母floor未激活，所有记录都满足固定子空间FM下界≤实测模型FM。',
'|池|+帧|诊断窗/parent|模型FM|速度零FM|速度零理论(copy+1)|干净头零FM|固定head oracle FM下界|oracle点MSE下界|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
 for pool,dd in summary.items():
  for h,r in dd.items():
   v=r['values'];lines.append(f'|{pool}|{h}|{r["windows"]}/{r["parents"]}|{v["model_fm"]:.5f}|{v["zero_velocity"]:.5f}|{v["copy"]+1:.5f}|{v["zero_clean_head"]:.5f}|{v["oracle_subspace_FM_floor"]:.5f}|{v["oracle_subspace_mse"]:.5f}|')
 lines += ['FM下界的存在提示满维目标/输出空间/时间权重都影响绝对loss；不能因其非零而只加训练步数，也不能用它替E90点预测均值差辩护。旧E33采用可学习256D目标，E90预测固定768D原生残差，难度和满维读出约束不同。',
'\n### 单样本为何输复制，均值为何仍不够',
'固定真实端点y，随机预测P的期望平方误差为||E[P]−y||²+Var(P)；N个独立样本均值的方差项缩为Var(P)/N。故“一个随机未来”对“确定性零残差”的MSE比较会同时测均值误差和采样方差。v1负skill是真实的该协议点误差，不是计算错；它不能单独判定分布预测完全无效。反过来，低FM也不能证明预测分布校准良好。',
'完整dev中，4Euler N1→N16把+10/25/50的额外MSE约去掉93.0%/97.0%/98.6%，+100转为小幅正skill；误差能量/真实内积表给出精确数值。增加Euler步数提高了输出分散，单样本MSE更差；不能简单把更多积分步当作MSE更优。N16均值的短时域改善仍远低于ridge，因此采样协议解释了大部分负值，未解释“为什么没有学到线性参照已有的均值预测能力”。',
'\n### B3 实际训练曝光核对',
'以只读原索引和原MultiSourceSampler在CPU精确重放5000×32次采样，末状态与ckpt训练摘要sampler逐字段完全一致；没有构造模型或执行训练。160000次抽样覆盖69922/107057个不同窗口（65.31%）；抽样次数相当于池大小1.4945倍，不等于真正遍历1.49个epoch。train64固定507窗中408个被主训练抽中过、99个未抽中过；适配器/链辅助暴露不额外算作主抽样。','|主抽样分组|+帧|窗/parent|v1 skill % [95% parent CI]|','|---|---:|---|---|']
 exposure=json.loads((R/'training_exposure.json').read_text())
 for group,dd in exposure['summary'].items():
  for h,r in dd.items():
   ci=r['ci95']['skill_copy'];lines.append(f'|{group}|{h}|{r["windows"]}/{r["parents"]}|{100*r["values"]["skill_copy"]:.3f} [{100*ci[0]:.3f}, {100*ci[1]:.3f}]|')
 lines += ['至少见过3次的窗口仍约−5%短时域skill；这使“仅dev泛化失败”更不符合证据，但它仍是随机单样本协议，不能把负值全等同于条件均值欠拟合。',
'\n## D 最终判断与最小下一步（没有开始训练）',
'**不建议把“原FM配方直接加长”作为下一步。也不能声称5000步已充分收敛、更多训练一定无效。** 末1000步FM OLS仍为−0.817/1000步，最后两个500步FM均值4.1166→3.7503，主池也只覆盖65.31%不同窗口；继续优化可能继续降FM。但训练parent、确实见过甚至见过≥3次窗口都未在v1赢复制，恢复原训练instruction也没修复；dev与train差距小，主要不是dev特有的泛化崩溃。',
'完整dev复现与缓存重放排除了本轮发现的帧/预处理漂移，v1扩大后仍为负；4Euler16均值把预测拉回复制附近，证明协议方差惩罚占主要负项。与此同时，纯线性模型+6.78/+9.28/+12.70/+22.61%，CI均>0，明确排除“该目标基本不可预测”的主要解释。固定head空间下界、FM时间加权、因果观测压缩与独立余弦头/采样头的目标差异，都是模型/训练路径需要核查的具体因素；本轮未做训练因果消融，不能挑一个当已证实唯一原因。',
'**最小下一步建议**：冻结本轮数据/缓存/归一化与ckpt-1，先锁定主目标是“准确预测条件均值”还是“生成多个可能未来”。若主目标仍是超越复制的MSE，下一轮仅做同输入、同固定768D残差目标的直接MSE读出/确定性均值对照，先以训练parent固定终点验证能否接近现有ridge，再看dev；保持神经预算和停止规则预先固定，不按dev/模型标注选checkpoint。保留本轮v1，并固定报告N1与N16均值，不能事后用最好网格替代旧基线。',
'若用户主要需要生成分布，先讨论条件均值读出与不确定性分支的分工，以及干净头的满维噪声通路；单样本MSE不足以验收分布质量，应另预注册适当分布评分/校准，不能因FM下降就宣称世界质量通过。现有证据不足以优先加大Transformer、换编码器或改归一化；最小对照应一次只改变目标/读出，避免同时换数据与结构。',
'\n### 未完成与需要用户决定',
'- **未完成的硬边界**：全部51个parent都有有效同合同缓存/指标无法实现；24个分别为18 metadata-only、5必要时钟字段缺失、1短录制。已完成51名单逐项审计及全部27可恢复parent的固定窗口缓存。不得把这27个泛化成全部51有观测。',
'- 历史表少数早期run的时域/步数/归一化在允许读取的已核查材料中未记，明确保留未记；没有加载旧权重或补造指标。分布校准、多seed、实机、动作条件新世界头均未在本轮运行。',
'- **用户决定1**：是否将27个有同合同录制的parent作为当前可评dev口径，另外24个列为当前合同下不可评；若要求覆盖短录制/旧时钟，须另定时间与短时域窗口合同，并保持原v1不变。metadata-only无法靠补缓存恢复图像。',
'- **用户决定2**：是否授权下一轮单一直接残差MSE对照，或优先保留FM并专门修正分布/均值读出。当前未继续训练、未启动任何新神经模型训练。']
 # Final immutable-input/resource audit precedes completion/log entries.
 protected=json.loads((R/'protected_source_manifest.json').read_text());changed=[p for p,h in protected.items() if sha(p)!=h];assert not changed,changed
 ledgers={p.name:json.loads(p.read_text()) for p in R.glob('gpu_ledger_*.json')};seconds=sum(v['seconds'] for v in ledgers.values());assert seconds<=7200
 features_bytes=sum(p.stat().st_size for p in (R/'features').rglob('*') if p.is_file());assert features_bytes<=30_000_000_000
 new_bytes=sum(p.stat().st_size for root in (R,T/'code_e93') for p in root.rglob('*') if p.is_file());assert new_bytes<60_000_000_000
 ckhash=sha(T/'code_e90/artifacts/e90_p1_20261114_01/ckpt_05000.pt');assert ckhash=='45164cfb86483709892d5c59a55fb830484cb5923eb7c018579595c826abbc49'
 receipt=dict(status='COMPLETE_DIAGNOSTICS_PARTIAL_51_PARENT_COVERAGE',brt=brt(),neural_training_updates=0,checkpoint_sha256=ckhash,protected_files=len(protected),protected_changes=changed,cache_bytes=features_bytes,new_project_bytes_before_final_receipt=new_bytes,GPU_seconds=seconds,GPUh=seconds/3600,gpu_ledgers=ledgers,frozen_parents=51,evaluable_parents=27,unevaluable_parents=24,dev_windows=211,dev100_windows=208,train64_windows=507,CPU_v1_reproduced_records=145,bootstrap_resamples=2000)
 atomic(R/'final_resource_receipt.json',receipt)
 lines += ['\n## 资源、命令与最终收据',f'完成 BRT={receipt["brt"]}；GPU进程墙钟合计={seconds:.3f}s={seconds/3600:.6f}GPUh≤2（含加载/I/O/全部GPU段，CPU失败0GPUh）；独立账本详见final_resource_receipt.json。新增dev特征={features_bytes:,}B={features_bytes/1e9:.6f}GB≤30；本项目结果/工具落盘约{new_bytes/1e9:.6f}GB（最终报告/日志另仅少量KB）≤60。',
f'284个受保护代码/固定输入文件哈希复核：实际清单{len(protected)}个，变化0；ckpt-1 SHA={ckhash}；E90原评估JSON不变。未碰训练parent缓存、同事写权限或他人进程，无MPS、OOM、下载或安装。每GPU段先free≥估算8GiB×1.5准入，逐窗/批检查显存余量与预算。编码器guard观察峰值约1796.5MiB（未单独保留完整峰值轨迹），诊断峰值见账本；均受8GiB硬上限。',
'主命令：`ssh -o BatchMode=yes <gpu-server>`；`cd T/code_e93`；`setsid nohup env PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u {extract.py,extract_supp.py,cpu_eval.py,gpu_diag.py,linear_03.py,...} > T/results/e93_dev_diag_20261114_01/<独立日志> 2>&1 < /dev/null &`。CPU复现收尾用cpu_validate.py；报告用early_report.py/linear_report.py/grid_report.py/extra_report.py/finalize.py；纯索引曝光重放exposure.py。完整阶段与失败恢复见code_e93/README_E93.md。',
'保留的失败：cpu_eval收尾的旧/新+100记录数断言（完整推理已完成，后续正确配对验证通过）；linear初次缺非必要threadpoolctl、随后等待旧CPU失败状态退出，均0GPUh。pipeline_status.json保留原中断；最终状态以E93_FINAL_STATUS.json及本节为准。']
 report=T/'notes/e93_dev_diag_20261114.md'
 with report.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'.join(lines)+'\n')
 with report.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);s=f.read();s=s.replace('状态：执行中。不训练，不更新 ckpt-1，不覆盖 E90 v1。','状态：诊断完成；51名单中27可评、24无法产生原合同窗口。不训练，不更新 ckpt-1，不覆盖 E90 v1。').replace('待完成。先在原 6 parent 验证；不一致停止扩展。','已通过原6parent严格重放，覆盖与不可恢复原因见下文A。').replace('待完成：完整 dev v1、Euler/样本均值、train64、分解、CPU 岭回归、FM 噪声底。','已完成所有可评窗口的v1、全采样网格、train64、分解、ridge及FM诊断，详见下文。').replace('## C 历史复制基线审计\n待完成。','## C 历史复制基线审计\n已完成指定来源全量检索与正负结果表，详见下文。').replace('待证据后填写。不启动训练。','结论与用户决定见末尾D。不启动训练。')
  f.seek(0);f.write(s);f.truncate()
 entry=f'\n### E93 — dev缓存与ckpt-1诊断（{receipt["brt"]} BRT）\n- ID/日期：E93 / 2026-11-14 BRT。目标：补dev合同特征，固定ckpt-1诊断，不训练。\n- 改动：新code_e93与results/e93_dev_diag_20261114_01；原6parent1134帧重放最大差0；51parent名单全审计，26发布parent+1原始失败录制恢复为27parent/211窗，24不可按原合同评；100为208窗/26parent。旧代码/训练缓存/原v1不写。\n- 命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup已有e29 Python执行prepare/extract/extract_supp/cpu_eval/cpu_validate/gpu_diag/linear_03/aux_cos/train_language/exposure与报告脚本，见code_e93/README_E93.md。\n- 结果：原145条v1最大MSE差4.77e-7；dev v1 skill10/25/50/100=−5.204/−4.982/−4.800/−14.256%；4Euler N16=−.362/−.148/−.069/+1.037%；ridge=+6.778/+9.279/+12.705/+22.615%，CI均>0；train64同样未胜copy。完整4/8/16×N1/8/16、分解、FM下界与历史正负比较见notes/e93_dev_diag_20261114.md。\n- 资源：{seconds/3600:.6f} GPUh含所有GPU段，新增特征{features_bytes/1e9:.6f}GB，总产物约{new_bytes/1e9:.6f}GB；保护清单{len(protected)}文件变化0。\n- 结论：采样方差解释大部分v1负项，条件均值仍远弱于线性可学习参照；不建议盲目加长原FM，先锁定均值/分布目标并做最小直接残差MSE对照，需另授权。\n- 未完成：24parent无同合同时钟/录制/窗口；非全51有效缓存。状态：COMPLETE_DIAGNOSTICS_PARTIAL_51_PARENT_COVERAGE；神经训练更新0。\n'
 for path,text in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',entry),(T/'notes/worklog.md',f'\n- {receipt["brt"]} BRT E93：27/51可评parent、211窗（100=208/26），cache重放差0；v1负skill主要受采样方差惩罚，N16近copy、ridge+6.78/+9.28/+12.70/+22.61%；不训练，{seconds/3600:.6f}GPUh，详见notes/e93_dev_diag_20261114.md；24parent原合同不可评。\n')]:
  with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text);f.flush();os.fsync(f.fileno())
 atomic(R/'E93_FINAL_STATUS.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
except BaseException as e:atomic(R/'finalize_stopped.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc(),brt=brt()));raise
