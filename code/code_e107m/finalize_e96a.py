"""CPU receipt and append-only final report; never chooses a checkpoint."""
import os,json,hashlib,fcntl,collections,time,datetime
from pathlib import Path
import numpy as np
import torch
from e96a.common import T,ART,REPORT,atomic,brt,sha,source_hash,report
from e96a.model import WorldModel
from eval_e96a import get_data,H,E93,RUN,M1,M2

def read(p):return json.loads(p.read_text())
def ci(s):
 v=s['values']['skill_copy']*100;lo,hi=np.array(s['ci95']['skill_copy'])*100
 return f'{v:.3f} [{lo:.3f}, {hi:.3f}]'
def dci(s):
 v=s['delta_skill']*100;lo,hi=np.array(s['ci95'])*100
 return f'{v:+.3f} [{lo:+.3f}, {hi:+.3f}]'
def slot(x):return float((x-x.mean(-2,keepdim=True)).square().mean()/x.square().mean())

def main():
 assert read(ART/'evaluation_status.json')['status']=='completed'
 assert not (ART/'FINAL_STATUS.json').exists()
 torch.set_num_threads(2);torch.manual_seed(90)
 train=read(RUN/'train_summary.json');old=read(M1/'train_summary.json');cfg=read(RUN/'config.json');oldcfg=read(M1/'config.json')
 assert train['steps']==5000 and train['sampler']==old['sampler']
 configdiff={k:[oldcfg.get(k),cfg.get(k)] for k in set(cfg)|set(oldcfg) if oldcfg.get(k)!=cfg.get(k)}
 assert set(configdiff)=={'schema','spatial_bypass'},configdiff
 updates=[json.loads(s) for s in (RUN/'updates.jsonl').read_text().splitlines()]
 baseupdates=[json.loads(s) for s in (M1/'updates.jsonl').read_text().splitlines()]
 assert len(updates)==len(baseupdates)==5000
 assert all(a['horizons']==b['horizons'] for a,b in zip(updates,baseupdates))
 hist=collections.Counter(tuple(x['horizons']) for x in updates)
 stats=dict(sampler_equals_M1=True,horizon_schedule_equals_M1=True,config_differences=configdiff,
            train_pool_windows=107057,train_parents=242,draws=160000,short_parameters_added=768*256,
            chain_updates=train['chain_updates'],chain_weight_sum=sum(x['chain_weight'] for x in updates),
            initial_adapter=read(RUN/'adapter_initial.json'),boundary_adapter=read(RUN/'adapter_chain_boundary.json'),
            final_adapter=train['adapter_last'],effective_chain_by_type={})
 for k in (25,50):
  for pred in (False,True):
   xx=[x for x in updates if x['chain_weight']>0 and x['chain_k']==k and x['predicted_reentry']==pred]
   stats['effective_chain_by_type'][str(k)+'_'+('predicted' if pred else 'teacher')]=dict(steps=len(xx),endpoint_mse_mean=float(np.mean([x['chain_supervised'] for x in xx])) if xx else None,gap_mean=float(np.mean([x['chain_gap'] for x in xx])) if xx else None)
 atomic(ART/'training_contract_audit.json',stats)
 # Small train-only structural probe; no optimization and no new acceptance threshold.
 tr,_=get_data();by=collections.defaultdict(list)
 for i,r in enumerate(tr.rows):by[r['parent_component']].append(i)
 ids=[v[len(v)//2] for _,v in sorted(by.items())[:16]]
 raw=torch.stack([tr.base[i]['obs'][4] for i in ids])
 d=torch.load(RUN/'ckpt_05000.pt',map_location='cpu',weights_only=False);m=WorldModel().eval();m.load_state_dict(d['model']);del d
 with torch.inference_mode():
  cur=m.current(raw);reverse=m.current(raw.flip(-2));plain=cur-m.current.bypass(raw);p=m.current.bypass(raw);ad=m.reentry(raw)
  cent=lambda x:x-x.mean(-2,keepdim=True)
  probe=dict(train_only=True,parents=[tr.rows[i]['parent_component'] for i in ids],selection='train64 sorted parent first16; median selected window',
   P_weight_norm=float(m.current.bypass.weight.norm()),current_reverse_max_abs=float((reverse-cur).abs().max()),
   pool_reverse_max_abs=float(((reverse-m.current.bypass(raw.flip(-2)))-plain).abs().max()),
   permutation_delta_identity_max_abs=float(((reverse-cur)-m.current.bypass(raw.flip(-2)-raw)).abs().max()),
   raw_slot_energy=slot(raw),pooled_slot_energy=slot(plain),current_slot_energy=slot(cur),bypass_rms=float(p.square().mean().sqrt()),current_rms=float(cur.square().mean().sqrt()),
   centered_adapter_nmse=float((cent(ad)-cent(cur)).square().mean()/cent(cur).square().mean().clamp_min(1e-12)),
   centered_adapter_note='descriptive train-only centered error, no new gate; reentry remains invariant pooling')
 atomic(ART/'trained_spatial_probe.json',probe)
 report('\n## 固定训练合同核对与空间诊断\n'
  f"5000步完成；160000抽样，sampler与M1逐字段相同，5000步horizons布局逐步相同；仅config schema和旁路声明不同。可训练参数{train['trainable_parameters']}（比M1增加196608）。首32/末32loss={train['first32']:.6f}/{train['last32']:.6f}。\n"
  f"初始adapter：NMSE={stats['initial_adapter']['normalized_mse']:.6f}, cos={stats['initial_adapter']['cosine']:.6f}, {stats['initial_adapter']['steps']}步；3000边界：NMSE={stats['boundary_adapter']['normalized_mse']:.6f}, cos={stats['boundary_adapter']['cosine']:.6f}, {stats['boundary_adapter']['steps']}步；最后门检：{json.dumps(train['adapter_last'])}。有效链{train['chain_updates']}/2000，链权重和{stats['chain_weight_sum']:.6f}（M1=150.05）；门和链调度未调整。详情training_contract_audit.json。\n"
  f"训练侧16parent固定诊断：P范数{probe['P_weight_norm']:.6f}；格子逆序current最大变化{probe['current_reverse_max_abs']:.6f}，旧pool部分逆序差{probe['pool_reverse_max_abs']:.3g}；旁路置换差恒等式误差{probe['permutation_delta_identity_max_abs']:.3g}。raw/pool/加旁路后的slot能量={100*probe['raw_slot_energy']:.4f}%/{100*probe['pooled_slot_energy']:.4f}%/{100*probe['current_slot_energy']:.4f}%；这不是互信息或任务定位精度。adapter去camera槽均值NMSE={probe['centered_adapter_nmse']:.5f}，仅诊断，不改原门。")
 methods=['E96a','M1','M2','ridge_E93'];pools=['train64','dev'];summaries={(m,p):read(ART/(m+'_'+p+'_summaries.json')) for m in methods for p in pools}
 text='\n## 单h主表：与M1/M2均值头及E93冻结岭并列\n'
 text+='训练RMS归一化残差；skill=1−parent等权model MSE/copy MSE；95%CI为2000次parent配对bootstrap/seed90，parent内先平均固定窗口。不是先算各窗skill再平均。每个h单独查询。\n'
 text+='|池|方法|h|窗/parent|MSE|copy MSE|skill % [95%CI]|残差方向cos|\n|---|---|---:|---|---:|---:|---|---:|\n'
 for p in pools:
  for method in methods:
   for h in H:
    x=summaries[method,p]['normalized'][str(h)];v=x['values']
    text+=f"|{p}|{method}|{h}|{x['windows']}/{x['parents']}|{v['model']:.6f}|{v['copy']:.6f}|{ci(x)}|{v['cosine']:.5f}|\n"
 text+='M1/M2为E94a固定5000步确定性均值分支；M2的FM N1/N16旧主表仍见E94a原报告，不混作M2均值。岭只读取E93冻结PCA/系数，未重拟合或重选alpha；原输入组织/训练采样与神经网络不同，不当纯W头消融。复推M1/M2/岭与原逐窗normalized MSE差在evaluation_audit.json，全部<2e−5。\n'
 report(text)
 paired=read(ART/'paired_E96a_minus_M1.json')
 text='\n## E96a−M1同parent配对差值\n正值表示E96a优于M1；单位为skill百分点，按同一bootstrap parent索引同时重抽两模型和copy分母，不相减独立CI。\n'
 text+='|池|h|原normalized细格 Δpp [95%CI]|同单位fine raw Δpp [95%CI]|同h camera-global raw Δpp [95%CI]|\n|---|---:|---|---|---|\n'
 for p in pools:
  for h in H:text+=f"|{p}|{h}|{dci(paired[p]['normalized'][str(h)])}|{dci(paired[p]['fine_raw'][str(h)])}|{dci(paired[p]['global_raw'][str(h)])}|\n"
 report(text)
 text='\n## 同h细格与按相机平均global并列\n'
 text+='先恢复raw残差，再分别对每相机8格求误差（fine）或先平均预测/真实向量后求误差（global）；两栏均为raw单位，避免把空间聚合与重标度混为同一效果。+100原本即每相机单global，两栏相同；不存在本模型+100细格预测，不能伪称100细粒度已评。下面同一固定单h输出，未按global重训或换scale。\n'
 text+='|池|方法|h|fine raw MSE / copy|fine raw skill % [95%CI]|global raw MSE / copy|global raw skill % [95%CI]|\n|---|---|---:|---|---|---|---|\n'
 for p in pools:
  for method in methods:
   for h in H:
    a=summaries[method,p]['fine_raw'][str(h)];b=summaries[method,p]['global_raw'][str(h)]
    text+=f"|{p}|{method}|{h}|{a['values']['model']:.6f}/{a['values']['copy']:.6f}|{ci(a)}|{b['values']['model']:.6f}/{b['values']['copy']:.6f}|{ci(b)}|\n"
 report(text)
 text='\n## 分相机和变化量四分位（E96a单h）\n'
 text+='skill点估计%；所有方法每组MSE/copy/CI见各*_decomposition.json。Q1–Q4按原真实normalized残差能量的每池/每h四分位，仅描述，未用作训练门、选点或阈值。raw细/global分解亦完整落盘。\n'
 text+='|池|h|top|left_wrist|right_wrist|Q1|Q2|Q3|Q4|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for p in pools:
  d=read(ART/('E96a_'+p+'_decomposition.json'))['normalized']
  for h in H:
   g=d[str(h)]['groups'];vals=[g[k]['values']['skill_copy']*100 for k in ('top','left_wrist','right_wrist','change_Q1','change_Q2','change_Q3','change_Q4')]
   text+='|'+p+'|'+str(h)+'|'+'|'.join(f'{x:.3f}' for x in vals)+'|\n'
 report(text)
 text='\n## 四h联合查询补充（不替代单h部署主表）\n无FM噪声；同固定输入/窗口和均值头。短h可以读长h查询，+100不读短h；联合布局未加训练随机h，不等同完整训练布局。表为normalized skill % [95%CI]。\n'
 text+='|池|方法|+10|+25|+50|+100|\n|---|---|---|---|---|---|\n'
 for p in pools:
  for mth in ('E96a','M1','M2'):
   x=read(ART/(mth+'_'+p+'_joint_summaries.json'))['normalized']
   text+='|'+p+'|'+mth+'|'+'|'.join(ci(x[str(h)]) for h in H)+'|\n'
 report(text)
 verdict=read(ART/'preregistered_verdict.json')
 text='\n## 预注册判定（未更改）\n'
 text+=f"总体：{'通过' if verdict['passed'] else '不通过'}。dev +10对M1配对改善CI下界>0：{verdict['dev10_paired_improvement_ci_lower_positive']}；同h fine raw点改善：{verdict['local_fine_raw_improves']}；六格全部通过：{verdict['six_short_h_all_pass']}。\n"
 text+='|池|h|E96a skill %|CI下界 %|ridge skill %|达到ridge %|通过|\n|---|---:|---:|---:|---:|---:|---|\n'
 for r in verdict['six_short_h_checks']:text+=f"|{r['pool']}|{r['h']}|{r['skill']*100:.3f}|{r['ci_lower']*100:.3f}|{r['ridge_skill']*100:.3f}|{r['ridge_fraction']*100:.1f}|{r['passed']}|\n"
 text+='只要只有global改善而原细格无改善，即不得通过。上述判定不由+100或四h补充结果抵消；没有延训、挑checkpoint或按结果重设门。旁路也改变adapter的停梯度监督目标，门/有效链暴露的变化是这次token化改变的必要连带影响，不能将最终差异全归因于W读出或空间信息单一路径。\n'
 report(text)
 # Read-only protection receipt and storage budget.
 initial=read(ART/'protected_hashes.json');changed=[p for p,h in initial.items() if sha(T/p)!=h];assert not changed,changed
 ready=read(T/'code_e94a/READY_readout.json');assert all(sha(T/'code_e94a'/p)==h for p,h in ready['sha256'].items())
 ledger=read(ART/'gpu_ledger.json');assert 'active' not in ledger and ledger['elapsed_s']<=3600
 folders=[T/'code_e96a',ART,RUN];size=sum(p.stat().st_size for root in folders for p in root.rglob('*') if p.is_file())+REPORT.stat().st_size
 assert size<5_000_000_000
 inputs=[T/'code_e88/artifacts/cpu_audit_01/residual_stats.json',T/'code_e90/artifacts/preflight_01/manifest.json',E93/'train64_selection.json',E93/'dev/windows_full.json',E93/'dev/captures_full.json',*[E93/f'ridge_h{h}_coefficients.npz' for h in H]]
 receipt=dict(status='COMPLETE',finished_brt=brt(),passed=verdict['passed'],source_hash=source_hash(),checkpoint=str(RUN/'ckpt_05000.pt'),checkpoint_sha256=sha(RUN/'ckpt_05000.pt'),GPU_seconds=ledger['elapsed_s'],GPUh=ledger['elapsed_s']/3600,server_new_bytes_before_receipt=size,local_small_source_copy_bound_bytes=2_000_000,storage_bound_bytes=size+3_000_000,protected_files=len(initial),protected_changes=changed,ready_unchanged=True,input_hashes={str(p):sha(p) for p in inputs},contract_tests='193 passed + 2 evaluation numerical contracts',no_other_process_actions=True,no_downloads=True,no_new_feature_cache=True,uncompleted_in_scope=[],limitations=['single seed','fixed frozen encoder proxy','dev 27 eligible parents only; original 24 not evaluable','not real robot task success'])
 atomic(ART/'FINAL_STATUS.json',receipt)
 report('\n## 资源、保护与完成收据\n'
  f"完成：{receipt['finished_brt']} BRT。GPU墙钟（含加载/I/O/全部段）={receipt['GPU_seconds']:.3f}秒={receipt['GPUh']:.6f}GPUh≤1，全部段关闭；峰值与准入见gpu_ledger.json及status/evaluation_status。无OOM、MPS、他人进程操作。服务器新增约{size/1e9:.6f}GB；本地仅小型源码与工具副本，保守额外2MB，合计上界{receipt['storage_bound_bytes']/1e9:.6f}GB≤5。没有复制大文件、缓存/数据集/新预训练权重下载，原位只读mmap。193+2=195项合同通过；保护清单{len(initial)}文件变化0，E94a READY绑定全部一致。\n"
  '目录：code_e96a/；results/e96a_m1_p1_20261114_01/（每500步保存，共10个完整ckpt，只评05000）；results/e96a_control_20261114_01/（合同/账本/统一评估/逐窗记录/差值/判定/审计）。\n'
  '命令：ssh -o BatchMode=yes <gpu-server>；cd T/code_e96a；setsid nohup env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u pipeline_e96a.py；evaluation_waiter_e96a.py 在固定5000步后调用 eval_e96a.py；CPU finalize_e96a.py。全任务日志和状态文件保留。\n'
  '已完成本轮全部指定实现、训练与评估；没有执行额外训练。单seed、冻结编码器代理评分不代表实机穿扎带成功率；原51dev parent中仅27具备同合同合法窗口，沿用E93既定口径，不补造余24。后续多seed或另项结构实验需另轮决定；本轮交付无需新增用户决定。')
 for p,txt in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',f"\n### E96a 完成 | {receipt['finished_brt']} BRT\n目标：E95 P1 current_j=pooled_j+P(raw_j)单因素；P共享无偏置零初始化，history/W/目标/归一化/优化器/seed90不变，adapter监督目标随current改变（必要配套，门不变）。195合同通过；Piper固定5000步/batch32/160000抽样，采样器及h布局与M1完全相同；单h主布局，27dev/64train parent、bootstrap2000 seed90，分camera/变化量Q、同h fine/global与四h补充完成。\n结果：dev+10 E96a−M1={dci(paired['dev']['normalized']['10'])}pp；预注册总体{'通过' if verdict['passed'] else '不通过'}。GPUh={receipt['GPUh']:.6f}≤1；新增上界{receipt['storage_bound_bytes']/1e9:.6f}GB≤5；保护文件变化0。目录/命令：code_e96a，results/e96a_control_20261114_01，results/e96a_m1_p1_20261114_01；setsid nohup pipeline_e96a.py/evaluation_waiter_e96a.py；报告notes/e96a_spatial_bypass_20261114.md。结论以固定终点预注册为准，不延训/挑点。状态 COMPLETE。\n"),(T/'notes/worklog.md',f"\n- {receipt['finished_brt']} BRT E96a 完成：195合同、Piper5000步，dev+10对M1 {dci(paired['dev']['normalized']['10'])}pp，预注册{'通过' if verdict['passed'] else '不通过'}；GPUh={receipt['GPUh']:.6f}；报告notes/e96a_spatial_bypass_20261114.md。\n")]:
  with p.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(txt);f.flush()
 print(json.dumps(dict(status='COMPLETE',passed=verdict['passed'],GPUh=receipt['GPUh'],storage_bound_GB=receipt['storage_bound_bytes']/1e9,dev10_delta=paired['dev']['normalized']['10']),ensure_ascii=False))

if __name__=='__main__':main()
