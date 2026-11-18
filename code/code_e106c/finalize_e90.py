"""CPU-only final delivery, outside the immutable training-code hash namespace."""
import csv,fcntl,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e90.common import T,ART,OUT,REPORT,atomic,brt,sha,source_hash
from e90.model import Config

RUN='e90_p1_20261114_01'
run=ART/RUN;result=T/'results'/RUN
status=json.loads((run/'status.json').read_text())
assert status['status']=='completed' and status['step']==5000
ledger=json.loads((ART/'gpu_ledger.json').read_text())
assert not ledger.get('active') and ledger['elapsed_s']<=21600
train=json.loads((run/'train_summary.json').read_text())
manifest=json.loads((ART/'preflight_01/manifest.json').read_text())
evaluation=json.loads((result/'ckpt1_dev_eval.json').read_text())
config=json.loads((run/'config.json').read_text())
assert source_hash()==train['code_hash'] and train['manifest_hash']==manifest['manifest_hash']
assert sha(train['checkpoint'])==train['checkpoint_sha256']
records=[json.loads(line) for line in (run/'updates.jsonl').read_text().splitlines()]
assert len(records)==5000 and [r['step'] for r in records]==list(range(1,5001))
assert all(r['mode']=='world_only' and r['action']==0 and r['phase']==0 and not r['clean_aux'] for r in records)
assert all(r['chain_weight']==0 for r in records[:3000])
checkpoints=[]
for p in sorted(run.glob('ckpt_*.pt')):
    checkpoints.append(dict(path=str(p),step=int(p.stem.split('_')[-1]),sha256=sha(p),bytes=p.stat().st_size))
assert [c['step'] for c in checkpoints]==list(range(500,5001,500))
atomic(result/'resolved_config.json',dict(training=config,architecture=vars(Config()),source_token=['piper','yam','abc'],world_fixed_horizons=[10,25,50,100],short_target_grid=[2,4],long_target_grid=[1,1],original_config_hash=train['config_hash'],code_hash=train['code_hash']))
atomic(result/'checkpoint_manifest.json',dict(checkpoints=checkpoints,final_ckpt1=checkpoints[-1],config=config,**{k:train[k] for k in ('code_hash','config_hash','manifest_hash')}))

def smooth(x,n=100):
    x=np.asarray(x,float);finite=np.isfinite(x);a=np.convolve(np.where(finite,x,0),np.ones(n),'same');b=np.convolve(finite.astype(float),np.ones(n),'same')
    return a/np.maximum(b,1)

keys=['step','loss','world','cosine_short','adapter','chain','chain_weight','chain_supervised','chain_gap','seconds']
with (result/'training_curves.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=keys+['h10','h25','h50','h100']);writer.writeheader()
    for r in records:writer.writerow({**{k:r.get(k,'') for k in keys},**{'h'+str(h):r['per_horizon'].get(str(h),'') for h in (10,25,50,100)}})
x=np.arange(1,5001);fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
axes[0,0].plot(x,[r['loss'] for r in records],alpha=.13,color='gray',linewidth=.5)
axes[0,0].plot(x,smooth([r['loss'] for r in records]),label='Total')
axes[0,0].plot(x,smooth([r['world'] for r in records]),label='World FM');axes[0,0].set_title('Training losses (100-step mean)')
for h in (10,25,50,100):axes[0,1].plot(x,smooth([r['per_horizon'].get(str(h),np.nan) for r in records]),label=f'+{h}')
axes[0,1].set_title('FM by horizon')
for key,label in [('chain','Chain total'),('chain_supervised','Chain supervision'),('chain_gap','Chain-direct gap')]:
    axes[1,0].plot(x,smooth([r.get(key,np.nan) if r['chain_weight']>0 else np.nan for r in records]),label=label)
axes[1,0].set_title('Chain losses (active updates)')
axes[1,1].plot(x,[r['chain_weight'] for r in records],label='Applied chain weight')
axes[1,1].plot(x,smooth([r['adapter'] for r in records]),label='Adapter NMSE');axes[1,1].set_title('Schedule and adapter distillation')
for ax in axes.flat:
    ax.axvline(3000,color='gray',linestyle='--',linewidth=.7);ax.axvline(4000,color='gray',linestyle='--',linewidth=.7)
    ax.set_xlabel('Optimizer steps');ax.grid(alpha=.2);ax.legend()
fig.savefig(result/'training_curves.png',dpi=160);plt.close(fig)

baseline=json.loads((ART/'old_source_baseline.json').read_text())
changed=[p for p,h in baseline.items() if sha(T/p)!=h]
assert not changed,changed
newbytes=sum(p.stat().st_size for root in (OUT,result) for p in root.rglob('*') if p.is_file() and not p.is_symlink())
assert newbytes<=60*10**9
curve_summary=[]
for lo,hi in ((1,500),(2501,3000),(3001,4000),(4001,5000)):
    selected=[r for r in records if lo<=r['step']<=hi]
    curve_summary.append(dict(start=lo,end=hi,loss=float(np.mean([r['loss'] for r in selected])),world=float(np.mean([r['world'] for r in selected])),chain_updates=sum(r['chain_weight']>0 for r in selected),mean_seconds=float(np.mean([r['seconds'] for r in selected]))))
delivery=dict(status='COMPLETE_P1_CACHED_DEV_LIMITED',completed_brt=brt(),train=train,
              gpu_seconds=ledger['elapsed_s'],GPUh=ledger['elapsed_s']/3600,new_bytes=newbytes,
              old_sources_unchanged=len(baseline),feature_cache_new_bytes=0,curve_summary=curve_summary,
              evaluation_backend=evaluation.get('evaluation_device'),evaluator_sha256=sha(OUT/'evaluate_cpu_tail.py'),dev_eval=str(result/'ckpt1_dev_eval.json'),checkpoint_manifest=str(result/'checkpoint_manifest.json'),
              limitations=['Only existing 6/51 dev parents cached; +100 only 1 window/parent.','No P3/actions/ABC/yam training; action/alignment heads not trained by P1.','Single seed; no physical-robot or language-effect conclusion.'])
atomic(result/'FINAL_DELIVERY.json',delivery)

def interval(d,key):
    if not d or d.get('values') is None:return 'NA'
    value=d['values'][key];ci=d['ci95'][key]
    return f'{value:.5f} [{ci[0]:.5f}, {ci[1]:.5f}]'

lines=['## E90 最终交付（以本节为最终口径）',f'完成：{brt()} BRT。状态：COMPLETE_P1_CACHED_DEV_LIMITED。步骤0与Piper P1固定5000步已完成；未启动步骤2。',
       '### 实现、测试与恢复','133项CPU合同通过（原108+新增25）；日志code_e90/artifacts/contracts_02.log。GPU smoke训练/评估/内部恢复与独立进程恢复均通过：下一步loss与全模型参数逐位一致。',
       f'策略可训练参数={train["trainable_parameters"]:,}；共享Transformer含最终LN={train["shared_core_parameters"]+512:,}。冻结SigLIP2图文双塔通过原缓存和真实encoder合同复用，无LoRA、无新特征、无下载。来源token为只读O；源采样硬上限按逐窗口前缀。阶段权重0无梯度、动作默认action_only。',
       '### 数据与训练口径',
       'rev4 parent划分未变。P1 107057窗/242parent，10/25/50各全量；100有效98655窗/237parent，屏蔽8402窗（end4374、cache3751、camera/time/seam174、source_mask103）。按时域mask保留同窗有效短时域，不移动标签。episode142永久排除。',
       '训练：77173窗既有E83任务指令（原有效范围），29884窗固定短指令；无过程叙述。固定短指令：Pick up the zip tie and thread its pointed end through its own locking head to form a loop. dev仅此固定人工短指令，无模型语言标注。',
       '5000步固定终点、batch32、AdamW lr3e-4/weight_decay1e-4、clip1、恒定LR；每500步保存。总损失=世界FM+.1短余弦+.1世界重入蒸馏辅助+计划链。P1不装配状态/动作输入，动作/对齐/阶段均不训练；完整三头实现保留。前3000步无链；3000边界再次蒸馏门检；3000..4000权重0→.1，最后1000步.1。每个合格更新执行一条25/50链（最多2窗、2Euler），无四分之一频率折减；计划预测重入.2→.8。3000步为用户锁定日程，非模型标签或dev推导的收敛判据。',
       '### 适配器门槛']
for label in ('initial','chain_boundary'):
    gate=json.loads((run/f'adapter_{label}.json').read_text())
    lines.append(f'{label}：NMSE={gate["normalized_mse"]:.6f}，cos={gate["cosine"]:.6f}，{gate["steps"]}蒸馏步，门开={gate["enabled"]}；64训练parent拟合/16训练parent门检，未用dev。')
lines += ['最后门检：'+json.dumps(train['adapter_last'],ensure_ascii=False),
          f'链有效更新={train["chain_updates"]}；门槛整轮关闭={train["adapter_gate_disabled"]}。每16步复查；门未过的更新不执行链。',
          '### 训练曲线','总损失/各时域FM/链项及权重见 `results/'+RUN+'/training_curves.csv` 与 `training_curves.png`。',
          '|步段|总loss均值|FM均值|链更新|秒/步|','|---|---:|---:|---:|---:|']
for row in curve_summary:lines.append(f'|{row["start"]}–{row["end"]}|{row["loss"]:.5f}|{row["world"]:.5f}|{row["chain_updates"]}|{row["mean_seconds"]:.4f}|')
lines+=['### ckpt-1 与清单',f'ckpt-1：`{train["checkpoint"]}`；SHA256=`{train["checkpoint_sha256"]}`。',
        f'配置hash=`{train["config_hash"]}`；数据清单hash=`{train["manifest_hash"]}`；代码hash=`{train["code_hash"]}`。T不是git仓库，使用全部e86/e88/e90 Python源的路径+字节SHA256，未捏造git commit。',
        '配置：`code_e90/artifacts/'+RUN+'/config.json`；数据：`code_e90/artifacts/preflight_01/manifest.json`；完整展开配置：`results/'+RUN+'/resolved_config.json`；全量checkpoint路径/sha/大小：`results/'+RUN+'/checkpoint_manifest.json`。',
        '|步数|文件|SHA256|','|---:|---|---|']
for cp in checkpoints:lines.append(f'|{cp["step"]}|{Path(cp["path"]).name}|{cp["sha256"]}|')
lines+=['### ckpt-1 dev 基线表（均值 [parent bootstrap 95%区间]）',
        'CPU float32、固定4Euler、W-only、无状态/动作，2000次parent有放回重采样；先parent内平均再parent等权；skill=1−模型MSE/基线MSE，配对重采样比值。误差为训练统计归一化残差坐标MSE。',
        '**缓存限制**：冻结dev共有51parent，既有缓存仅6parent/48窗；100仅1窗/1parent，其他47缺缓存。没有补特征/模型语言。单parent区间会退化，不构成可靠泛化CI；缺层记NA。此dev表未用于阈值或checkpoint选择。',
        'CPU evaluator为code_e90/evaluate_cpu_tail.py，SHA256绑定在evaluation_receipt.json；步骤2必须复用相同CPU后端、每窗seed和4Euler，CPU与CUDA随机数序列不同。自动收尾的目录缺失异常与util0准入拒绝记录均保留；未重训。',
        'JSON基线：`results/'+RUN+'/ckpt1_dev_eval.json`；逐窗记录：同目录`dev_records.json`。',
        '|+帧|分层|窗/parent|模型MSE|复制MSE|线性MSE|skill复制|skill线性|','|---|---|---|---|---|---|---|---|']
for h,info in evaluation['horizons'].items():
    for group in ('overall','demo3','demo4','demo6','demo6early','demo6native'):
        d=info['overall'] if group=='overall' else info['demos'].get(group,info['strata'].get(group))
        count=f'{d["windows"]}/{d["parents"]}' if d else '0/0'
        lines.append('|'+ '|'.join([h,group,count]+[interval(d,k) for k in ('model','copy','linear','skill_copy','skill_linear')])+'|')
lines+=['### 链与真实增量可加性','链gap与直接+2k预测之差、以及链对真实+2k监督，使用同一归一化目标；链只在真实缓存端点可用时计算。',
        '|链|窗/parent|对真实端点MSE|对直接预测gap|','|---|---|---|---|']
for k,info in evaluation['chains'].items():
    d=info['overall'];lines.append(f'|{k}+{k}→{2*int(k)}|{d["windows"]}/{d["parents"]}|{interval(d,"chain_model")}|{interval(d,"chain_gap")}|')
add=evaluation['additivity']['overall'];lines.append(f'真实Δ(0→50)与Δ(0→25)+Δ(25→50)：{add["windows"]}窗/{add["parents"]}parent，相对误差{interval(add,"relative_error")}，余弦{interval(add,"cosine")}。共享同一固定编码中间端点，属于代数与帧映射自检，不能解释为预测链准确性。')
lines+=['### 资源与旧结果保护',f'总GPU墙钟={ledger["elapsed_s"]:.3f}s={ledger["elapsed_s"]/3600:.6f}GPUh≤6；失败段计入，台账无active worker。每段准入free≥8192MiB×1.5、util0，烟雾实测峰值1082.29MiB；正式训练每步受≤8192MiB guard约束，但退出发生在汇总前，未单独落盘其精确峰值。没有MPS、操作他人进程或写同事目录。',
        f'本轮code_e90+results新增落盘={newbytes:,}B（{newbytes/1e9:.3f}GB）≤60GB，新增特征缓存0。旧code_e82/e86/e88共{len(baseline)}个Python源hash全部不变。',
        '### 未完成与需要用户决定的事','未完成：全51parent dev与充分+100分层覆盖，现有缓存无法支持；不扩缓存的限制下以6parent现有缓存结果交付并保留NA。没有动作/P3/ABC/yam训练、消融、多种子、实机或语言效能结论。完整动作/对齐头在本轮P1未受监督，不能视为已训练策略。',
        '本轮执行无需新增决定。后续若要充分的dev/+100评估，需要另行安排已有冻结dev图像的特征缓存补齐；步骤2等待用户下一轮指令，本轮不启动。']
blocks=[];table=[]
for line in lines:
    if line.startswith('|'):table.append(line);continue
    if table:blocks.append('\n'.join(table));table=[]
    blocks.append(line)
if table:blocks.append('\n'.join(table))
with REPORT.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n'+'\n\n'.join(blocks)+'\n')
entry=f'\n### E90 最终收据 — {brt()} BRT\n- 目标/改动：完整v3.1实现与Piper P1固定5000步；133项CPU合同，GPU烟雾/独立恢复逐位一致，阶段0、来源上限、口径B、每步链/ramp、全checkpoint与init-ckpt。\n- 命令：setsid nohup e29/bin/python -m e90.train e90_p1_20261114_01 --mode-probabilities .25 .25 .25 .25 --clean-aux 0；CPU finalize_e90.py。\n- 目录：code_e90/artifacts/{RUN}；results/{RUN}；报告notes/e90_full_v31_p1_20261114.md。\n- 结果：5000步完成、10份500间隔ckpt；ckpt-1 SHA={train["checkpoint_sha256"]}；GPUh={ledger["elapsed_s"]/3600:.6f}，新增{newbytes/1e9:.3f}GB，新增特征0；dev 4Euler/2000 bootstrap，6/51缓存parent，100仅1窗。\n- 结论：步骤0/1完成，dev缓存覆盖受限，完整指标及区间见ckpt1_dev_eval.json；不以dev选型，不作动作/语言效果结论。\n- 状态：COMPLETE_P1_CACHED_DEV_LIMITED；步骤2未启动。\n'
for path,text in ((T/'docs/experiment_log/EXPERIMENT_LOG.md',entry),(T/'notes/worklog.md',f'\n- {brt()} BRT E90完成：133合同、Piper P1 5000步/10ckpt、{ledger["elapsed_s"]/3600:.6f}GPUh；dev缓存6/51parent、+100仅1窗；results/{RUN}/ckpt1_dev_eval.json；步骤2未启动。\n')):
    with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text)
print(json.dumps({k:delivery[k] for k in ('status','GPUh','new_bytes','old_sources_unchanged','dev_eval')},ensure_ascii=False))
