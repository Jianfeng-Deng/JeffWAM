import sys
sys.dont_write_bytecode=True
import json,collections,fcntl,os
from e99w2_common import *
def main():
 lock=json.loads((ART/'IMPLEMENTATION_LOCK.json').read_text());assert all(sha(ROOT/p)==v for p,v in lock['training_sources'].items())
 protected=json.loads((ART/'protected_sources_lock.json').read_text());assert all(sha(r['path'])==r['sha256'] for r in protected);atomic(ART/'protection_final.json',dict(status='passed',own_training_sources_unchanged=len(lock['training_sources']),read_only_files_unchanged_since_lock=len(protected),brt=brt()))
 ev=ART/'evaluation_01';assert (ev/'COMPLETE.json').exists();decision=json.loads((ev/'decision.json').read_text());summary=json.loads((ev/'world_summary.json').read_text());paired=json.loads((ev/'world_paired.json').read_text());ret=json.loads((ev/'retention.json').read_text());actions=json.loads((ev/'action_summary.json').read_text());lat=json.loads((ev/'latency.json').read_text());coverage=json.loads((ev/'coverage.json').read_text());train=json.loads((RUN/'train_summary.json').read_text());ledger=json.loads((ART/'gpu_ledger.json').read_text());assert not ledger.get('active');gpu_upper=ledger['elapsed_s']+30*len(ledger['segments']);assert gpu_upper<=14400
 # Only own E99-W2 paths count; no symlinks or read-only referenced files are charged as copies.
 roots=[ROOT,*sorted((T/'results').glob('e99w2_*')),REPORT];files=[p for root in roots for p in ([root] if root.is_file() else root.rglob('*')) if p.is_file() and not p.is_symlink()];disk=sum(p.stat().st_size for p in files);assert disk<10*1024**3
 lines=['## 最终统一评估（唯一5000步终点）','skill=1−parent等权MSE/parent等权copy能量；固定窗内平均后parent等权，2000次同parent重抽，seed90；+10/+25/+50原2×4，+100原global。所有输入输出与指标有限；MHA fastpath关闭。N8为8次动作采样各自条件世界预测的均值，不是先平均动作再跑世界头。','|h|方法|窗口/parent|skill % [95%CI]|方向cos|','|---:|---|---|---|---:|']
 for s in summary:
  if s['camera']!='all':continue
  ci=s['ci95'];lines.append(f"|{s['h']}|{s['method']}|{s['windows']}/{s['parents']}|{100*s['skill']:.3f} [{100*ci[0]:.3f}, {100*ci[1]:.3f}]|{s['cosine']:.5f}|")
 lines+=['','|h|pred_N1−no Δpp [95%CI]|N1保留oracle增益 %|N8保留 %|','|---:|---|---:|---:|']
 for h in (10,25,50,100):
  rows=[r for r in paired if r['h']==h and r['camera']=='all' and r['method']=='pred_N1']
  if not rows:continue
  p=rows[0];r={v['method']:v for v in ret if v['h']==h and v['camera']=='all'};fmt=lambda x:'NA' if x is None else f'{x:.2f}';lines.append(f"|{h}|{100*p['delta_skill']:.3f} [{100*p['ci95'][0]:.3f}, {100*p['ci95'][1]:.3f}]|{fmt(r['pred_N1']['percent'])}|{fmt(r['pred_N8']['percent'])}|")
 lines+=['','### 分相机（skill % / 方向cos）','|h|方法|top|left_wrist|right_wrist|','|---:|---|---|---|---|']
 for h in (10,25,50,100):
  for method in ('oracle','gt_action','pred_N1','pred_N8','no_traj','noisy_gt'):
   ss={s['camera']:s for s in summary if s['h']==h and s['method']==method}
   if not ss:continue
   lines.append('|'+ '|'.join([str(h),method,*[f"{100*ss[c]['skill']:.3f} / {ss[c]['cosine']:.5f}" for c in ('top','left_wrist','right_wrist')]])+'|')
 lines+=['','### 动作误差（next-frame反馈标签）','前8/H50为前缀平均；复制当前测量状态为静态基线；不是控制命令误差或执行成功率。','|推理|前缀|关节L1 rad|夹爪L1 mm|','|---|---:|---:|---:|']
 for a in actions:lines.append(f"|{a['method']}|{a['h']}|{a['joint_L1_rad']:.6f}|{a['gripper_L1_mm']:.3f}|")
 lines+=['','### 推理时延与噪声敏感度',json.dumps(lat,ensure_ascii=False),'误差同幅高斯噪声的每步/每维RMS仅由预先固定的16个训练parent估计，非dev拟合；独立高斯只匹配幅度、不匹配预测误差的偏置和时序相关性。noisy_gt表据此解释。','', '### 预注册判定与范围',f"机械判定 deployable_condition_effective={decision['deployable_condition_effective']}。coverage={json.dumps(coverage,ensure_ascii=False)}。",'当某h仅1个parent时bootstrap退化，不支持总体泛化推断；即使机械CI通过也不能把退化CI当作可信的+100证据。oracle是同一个动作条件训练终点上的oracle输入参照，并非保证有序的数学上界；保留率分母非正时NA、不裁剪负差或超过100%的比率。','确定性W均值头不读W噪声/时钟，因此sync与independent在本均值W侧分布等价；四个模式仍按原25%概率抽样。no-traj是条件置空干预，训练没有额外添加trajectory dropout。','训练采用P3而E98-W采用P1，初始化也不同；本轮只对同checkpoint的条件来源作配对，不能把跨实验差当纯条件消融。W-only主评分去掉A token以隔离轨迹源；部署先action_only采样，再只把计划特征送W，没有把预测状态冒充观测。短W可经长W间接读取更长期计划，沿用E98联合训练限制；主评估均单h。','', '### 资源与完整收据',f"GPU精确分段总秒={ledger['elapsed_s']:.6f}；每段另加30秒保守清理上界={gpu_upper:.6f}秒={gpu_upper/3600:.6f}GPUh≤4。新增文件逻辑字节={disk}（报告最终追加及小型收据另计，远低于10GiB）；训练峰值={train['peak_mib']:.2f}MiB。",f"有效链更新={train['chain_updates']}；四模式计数={train['modes']}；适配器={json.dumps(train['extra'],ensure_ascii=False)}。",f"初始化SHA={sha(INIT)}；终点SHA={sha(RUN/'ckpt_05000.pt')}。",'命令：env CUDA_VISIBLE_DEVICES=GPU-56f697ae-6017-3e7b-5d9b-1b612543fe40 E99_GPU_UUID=同UUID /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u code_e99w2/e99w2_driver04.py（以code_e99w2为cwd；setsid nohup分离）。driver顺序完成训练、e99w2_eval.py、e99w2_finalize.py。失败与早停保留在control目录及前三个run目录；不覆盖旧实验包。','', '完整逐窗/分相机/CI与噪声结果：results/e99w2_control_20261115_01/evaluation_01/world_records.json、world_summary.json、world_paired.json、retention.json、action_summary.json、latency.json、vectors.npz。实现与378项只读文件锁定后哈希核验见protection_final.json。','### 完成状态、未完成与用户决定','实现、固定训练和本轮冻结缓存内评估全部完成。未完成的是冻结P3 dev既有缓存之外的广覆盖长期验证；未扩窗/加模型语言/按dev调参。需要用户后续决定是否单独授权更广但仍冻结parent的长期评估覆盖；本轮无需为已授权工作追加批准。']
 report('\n'.join(lines));receipt=dict(status='COMPLETE_FIXED_P3_DEV_CACHE_LIMITED',brt=brt(),decision=decision,GPUh_upper=gpu_upper/3600,new_bytes_before_final_report=disk,checkpoint=train['checkpoint'],report=str(REPORT),code_hash_final=source_hash(),coverage=coverage)
 atomic(ART/'FINAL_STATUS.json',receipt);atomic(ART/'STATUS.json',receipt)
 for path,line in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',f"\n### E99-W2 完成 | {brt()} BRT\n目标：P3完整v3.1计划轨迹部署条件收益。改动：code_e99w2；初始化E98-W A；5000步/batch32/lr3e-4。命令：setsid nohup env CUDA_VISIBLE_DEVICES=GPU-56f697ae-6017-3e7b-5d9b-1b612543fe40 E99_GPU_UUID=同UUID /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e99w2_driver04.py。目录：results/e99w2_p3_20261115_04、results/e99w2_control_20261115_01/evaluation_01。结果：预注册有效={decision['deployable_condition_effective']}，GPUh保守上界={gpu_upper/3600:.6f}，新增约{disk/1024**3:.3f}GiB；结论/覆盖限制见notes/e99w2_planned_traj_20261115.md；状态COMPLETE_FIXED_P3_DEV_CACHE_LIMITED。\n"),(T/'notes/worklog.md',f"\n- {brt()} BRT E99-W2完成：5000步，部署条件机械判定={decision['deployable_condition_effective']}，GPUh≤{gpu_upper/3600:.6f}；P3冻结dev缓存覆盖受限，详见notes/e99w2_planned_traj_20261115.md。\n")]:
  with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(line)
 print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
