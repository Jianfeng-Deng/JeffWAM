import os,sys,json,hashlib,fcntl
from pathlib import Path
from e103d_runtime import *
import numpy as np

def read(n):return json.loads((CONTROL/n).read_text())
def fmt(a,scale=1):
 if a is None:return 'NA'
 if a.get('ci95') is None:return str(a.get('value'))
 return f"{a['value']*scale:.5f} [{a['ci95'][0]*scale:.5f},{a['ci95'][1]*scale:.5f}]"
def main():
 lock=read('PRETRAIN_LOCK.json');audit=read('FINAL_AUDIT.json');assert read('PIPELINE03.json')['status']=='complete';assert read('OPENLOOP_DTYPE_TEST02.json')['status']=='passed'
 def checkfinite(v):
  if isinstance(v,float):assert np.isfinite(v)
  elif isinstance(v,dict):
   for a in v.values():checkfinite(a)
  elif isinstance(v,list):
   for a in v:checkfinite(a)
 for path in CONTROL.rglob('*.json'):checkfinite(json.loads(path.read_text()))
 changed=[k for k,v in lock['source_sha256'].items() if sha(ROOT/k)!=v];assert not changed,changed
 for n,v in lock['e107p_input_sha256'].items():assert sha(T/'results/e107p_diagnosis_20261116_02'/n)==v
 a=read('SKILL_ACTION_SUMMARY.json');p=json.loads((T/'results/e106c_control_20261115_02/SUMMARY.json').read_text())
 errors=[abs(a['BASE'][sub]['mean_world'][h]['skill'][method]-p['S20000'][sub]['world'][h]['skill'][method]) for sub in ('all27','original48') for h in ('10','25','50','100') for method in ('gt_action','pred_N1','no_traj','oracle')];assert max(errors)<1e-6;report('## 基线重放一致性\nBASE/M1、k25/k50的float64原计划CPU回归测试通过（OPENLOOP_DTYPE_TEST02.json），只转换第二跳robot模型输入dtype，不移动或重标注数据。固定20000的两集合/四条件/四时域skill与E106C原汇总最大绝对差='+str(max(errors))+'；无重新选点。')
 # Exact old checkpoint and training stream checks independent of outcome metrics.
 train={g:json.loads((T/f'results/e107m_{g.lower()}_s90_20261116_01/summary.json').read_text()) for g in ('M0','M1')}
 assert train['M0']['flow_sha']==train['M1']['flow_sha']==lock['flow_sha'];assert train['M1']['frozen_unchanged'];assert all(r['step']==23000 and r['updates']==3000 for r in train.values())
 for g in ('M0','M1'):
  import torch
  cp=torch.load(T/f'results/e107m_{g.lower()}_s90_20261116_01/ckpt_23000.pt',map_location='cpu',weights_only=False)
  assert all(torch.isfinite(v).all() for v in cp['model'].values());assert all(torch.isfinite(v).all() for st in cp['optimizer']['state'].values() for v in st.values() if torch.is_tensor(v));assert cp['sampler']['count']==736000 and cp['p1sampler']['count']==736000
  if g=='M1':
   base=torch.load(Path(lock['checkpoints']['BASE']),map_location='cpu',weights_only=False)
   assert all(torch.equal(v,cp['model'][n]) for n,v in base['model'].items());del base
  del cp
 rows=read('SHRINKAGE.json');dist=read('DISTRIBUTION.json');lines=['## 分布指标可读主表（主27parent）','完整48窗附表及每个样本成员的同口径指标CI均在JSON。方差比在原始token上，sample_population按窗内K4等权（实际211窗/27parent，+100为208窗/26parent），分布指标在E107P训练PCA32上。','|h|条件|组/读法|方差比 [CI]|位移方差比 [CI]|MMD² [CI]|精确率 / 召回率|C2ST [CI]|Fréchet [CI]|熵代理 [CI]|','|---:|---|---|---|---|---|---|---|---|---|']
 for r in rows:
  if r['subset']!='all27':continue
  d=next((x['mean'] for x in dist if all(x[k]==r[k] for k in ('subset','h','condition','group','reading'))),None)
  if d is None:
   lines.append(f"|{r['h']}|{r['condition']}|{r['group']}/{r['reading']}|{fmt(r['variance_ratio'])}|{fmt(r['delta_variance_ratio'])}|见四个sample成员|见JSON|见JSON|见JSON|见JSON|");continue
  lines.append(f"|{r['h']}|{r['condition']}|{r['group']}/{r['reading']}|{fmt(r['variance_ratio'])}|{fmt(r['delta_variance_ratio'])}|{fmt(d['mmd'])}|{fmt(d['precision'])} / {fmt(d['recall'])}|{fmt(d['c2st'])}|{fmt(d['frechet'])}|{fmt(d['entropy'])}|")
 report('\n'.join(lines))
 lines=['## 关节与夹爪简表','|集|组|前8/H50关节L1 rad|状态准确率|F1±5/±10|假切换/分钟±10|','|---|---|---|---|---|---|']
 for g in a:
  for subset,s in a[g].items():
   v=s['actions'];lines.append(f"|{subset}|{g}|{v['joints']['8']['l1']:.6f}/{v['joints']['50']['l1']:.6f}|{v['state_accuracy']['mean']:.5f}|{v['events']['5']['f1']:.5f}/{v['events']['10']['f1']:.5f}|{v['events']['10']['false_switches_per_minute']:.4f}|")
 report('\n'.join(lines))
 lines=['## 分叉样本开概率与两簇代理','两簇代理为同窗K4同时出现冻结探针开概率<.3和>.7；不是混合模型拟合或统计双峰检验。真开比例为原E102A onset状态；不足格直接标注，不调整阈值。','|h|条件|窗/parent|不足30窗或6parent|爪|两簇代理比例 [CI]|样本开比例 [CI]|真开比例 [CI]|差 [CI]|','|---:|---|---|---|---:|---|---|---|---|']
 for r in read('BRANCH.json'):
  if r['subset']!='all27' or r['group']!='mixed':continue
  if not r['windows']:lines.append(f"|{r['h']}|{r['condition']}|0/0|是|NA|NA|NA|NA|NA|")
  for g in r['per_gripper']:lines.append(f"|{r['h']}|{r['condition']}|{r['windows']}/{r['parents']}|{'是' if r['insufficient'] else '否'}|{g['gripper']}|{fmt(g['two_cluster_proxy'])}|{fmt(g['sample_open_fraction'])}|{fmt(g['true_open_fraction'])}|{fmt(g['open_fraction_gap'])}|")
 report('\n'.join(lines))
 account=[]
 for path in CONTROL.glob('gpu_*.json'):
  meta=json.loads(path.read_text());upper=meta['elapsed_s'];phase=meta['phase']
  if phase.startswith('eval_'):
   import datetime
   start=datetime.datetime.fromisoformat(meta['started_brt']).timestamp();art=CONTROL/phase;end=max((f.stat().st_mtime for f in art.rglob('*') if f.is_file()),default=start);upper=max(upper,end-start+30)
  account.append(dict(phase=phase,guard_seconds=meta['elapsed_s'],whole_phase_upper_seconds=upper,status=meta['status']))
 assert all(r['status']=='closed' for r in account);seconds=sum(r['whole_phase_upper_seconds'] for r in account);assert seconds<=7200;atomic(CONTROL/'GPU_PROCESS_ACCOUNTING.json',dict(phases=account,GPUh_upper=seconds/3600,failed_eval01_CPU_only=True));audit['GPUh_guard_only']=audit['GPUh_upper'];audit['GPUh_upper']=seconds/3600
 allbytes=sum(f.stat().st_size for root in (ROOT,CONTROL,T/'results/e107m_m0_s90_20261116_01',T/'results/e107m_m1_s90_20261116_01') for f in root.rglob('*') if f.is_file());audit.update(valid_evaluation_dirs='eval_BASE_03/eval_M0_03/eval_M1_03',valid_pipeline='PIPELINE03.json',final_review='passed',source_effective_lock_unchanged=True,training_sources_unchanged_since_training=True,execution_corrections=lock.get('execution_corrections',[]),recording_correction=lock.get('recording_correction'),E107P_inputs_unchanged=True,old_model_all_tensors_bitwise_unchanged_M1=True,sampler_draws_each=736000,new_bytes=allbytes,conservative_bytes_including_local_and_logs=allbytes+50_000_000,brt=brt());assert audit['conservative_bytes_including_local_and_logs']<10_000_000_000
 atomic(CONTROL/'FINAL_AUDIT.json',audit);report('## 交付终审\n'+json.dumps(audit,ensure_ascii=False)+'\n命令：cwd=T/code_e107m；nohup e29/bin/python -u e107m_pipeline.py，start_new_session=True（setsid等价），训练管线PID3429168；有效评估/汇总管线e107m_pipeline03.py（最终LAUNCH03.json）。GPU终审按guard开始至最终评估文件落盘再加30秒保守计费，覆盖压缩阶段，GPU_PROCESS_ACCOUNTING.json为最终资源口径。CPU测试失败均0GPU秒；训练与评估GPU段均计入本轮上界。关节真值为next-frame follower反馈，夹爪事件为反馈运动起点代理；原计划条件属于诊断，不是实发命令或实机成功真值。CI条件于seed90固定K4，不计噪声样本或seed间变异。spread2使用K分母，未作K−1修正；CRPS为坐标边际而非联合分数。没有模拟/像素损失/像素解码器/新增模型语言/新数据或权重下载；旧缓存原位只读；本轮用户Piper合同覆盖AGENTS旧天机/仿真说明。\n')
 s=REPORT.read_text();s=s.replace('状态：代码审查与预检；','状态：COMPLETE（有效训练run01、评估run03；以PIPELINE03.json与FINAL_AUDIT.json为准，PIPELINE/PIPELINE02为保留失败记录）；',1);REPORT.write_text(s)
 for path in (T/'docs/experiment_log/EXPERIMENT_LOG.md',T/'notes/worklog.md'):
  with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n- '+brt()+' BRT E107-M 交付终审通过；训练run01+评估run03；最终GPUh上界='+str(audit['GPUh_upper'])+'（含压缩尾段与全部失败），新增保守字节='+str(audit['conservative_bytes_including_local_and_logs'])+'；FINAL_AUDIT.json/GPU_PROCESS_ACCOUNTING.json；报告notes/e107m_multimodal_head_20261116.md。\n')
 print(json.dumps(audit,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
