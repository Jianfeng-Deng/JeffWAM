from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e108g_diag_20261116_01';R=T/'notes/e108g_gripper_diagnosis_20261116.md'
assert (O/'FINAL_STATUS.json').exists();assert not (O/'FINAL_VERIFICATION.json').exists()
g=json.loads((O/'G1_RESULTS.json').read_text());base=g['all27']['baseline']['10'];n=sum(v['true']>0 for v in base['parent_raw'].values());zero=[v for v in base['parent_raw'].values() if not v['true']]
lines=['## 补充：指标上限与偏移配对区间',f'沿用旧口径，零真实事件parent的F1记0，即使无假检也记0。本主集{n}/27个parent有真实事件，理想onset事件预测的宏平均F1上限为{n/27:.6f}；这来自指标定义，不能归为标签噪声或模型错误。6个零事件parent共有'+str(sum(v['pred'] for v in zero))+'个假检，这些假检影响假切换率，但消除它们本身不会提高该F1。此处只解释口径，未修改任何旧指标。','以下偏移为所有已匹配事件的全体有符号中位数；每次parent成对重采样后重新计算两个中位数之差，非将逐parent中位数平均。缺匹配的bootstrap样本不定义偏移差，单列有效次数。','|集合|规则|tol|Δ偏移median帧|95%CI|有效bootstrap次数|','|---|---|---|---|---|---|']
for subset,x in g.items():
 for fam,v in x.items():
  if fam=='baseline':continue
  for tol,d in v['paired'].items():
   v=d['offset_median'];lines.append(f"|{subset}|{fam}|{tol}|{v['delta']}|{v['ci95']}|{v['valid_bootstraps']}|")
lines+=['滞回状态规则：closed→open需p>high，open→closed需p≤low；最小停留是上次输出切换后d帧内禁止再切换（不是等待未来帧后回填事件）；EMA严格使用ema_t=alpha*p_t+(1−alpha)*ema_(t−1)。每窗从当前实测二值状态重置；全部23配置通过截断及未来扰动测试。','标签定义差尤其影响±5：实测阈值crossing对原onset的F1@±5为0.280511、@±10为0.511376，匹配crossing−onset中位数5帧、四分位4/8帧。这只是两种反馈定义的差，不说明模型照搬crossing，也不允许平移预测或标签来改善本轮成绩。','预注册结论保持：瓶颈不在因果解码，需要在原架构内通过目标/数据层面解决。本轮无待决定事项；后续目标或采样改动需另轮明确授权，未实施。']
s=R.read_text().replace('偏移Q25/median/Q75帧','偏移[Q25,Q75]/median帧')
R.write_text(s+'\n'+'\n'.join(lines)+'\n')
files={}
for p in sorted((T/'code_e108g').glob('*.py')):files[str(p.relative_to(T))]=hashlib.sha256(p.read_bytes()).hexdigest()
size=sum(p.stat().st_size for root in (T/'code_e108g',O) for p in root.rglob('*') if p.is_file())+R.stat().st_size
assert size<2*1024**3
v=dict(status='complete',brt=datetime.now(timezone(timedelta(hours=-3))).isoformat(),GPUh=0,new_bytes_before_this_receipt=size,files_sha256=files,report_sha256=hashlib.sha256(R.read_bytes()).hexdigest(),plots_visual_checked=True,no_training=True,bootstrap_seed=90,bootstrap_replicates=2000)
(O/'FINAL_VERIFICATION.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));print(json.dumps(v,ensure_ascii=False))
