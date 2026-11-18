import sys,json,collections,fcntl
sys.dont_write_bytecode=True
from e102x_setup import *
import numpy as np
from e102a_finalize import action_summary,quant
GROUPS=('A0','A1','A2');METHODS=('gt_action','pred_N1','no_traj','oracle')
def bootstrap(n):return np.random.default_rng(90).integers(n,size=(2000,n))
def worlds(rows):
 out={}
 for h in (10,25,50,100):
  rr=[r for r in rows if r['h']==h];parents=sorted({r['parent'] for r in rr});idx=bootstrap(len(parents));a={m:np.array([[np.mean([r[k] for r in rr if r['parent']==p and r['method']==m]) for k in ('model','copy')] for p in parents]) for m in METHODS};copy=a['no_traj'][:,1];den=copy.mean();delta=a['no_traj'][:,0]-a['pred_N1'][:,0];od=a['no_traj'][:,0]-a['oracle'][:,0]
  with np.errstate(divide='ignore',invalid='ignore'):ret=delta[idx].mean(1)/od[idx].mean(1)
  out[str(h)]=dict(parents=len(parents),windows=len(rr)//4,skill={m:float(1-v[:,0].mean()/den) for m,v in a.items()},skill_ci95={m:quant(1-v[idx,0].mean(1)/copy[idx].mean(1)) for m,v in a.items()},delta_skill=float(delta.mean()/den),delta_ci95=quant(delta[idx].mean(1)/copy[idx].mean(1)),retention=float(delta.mean()/od.mean()) if od.mean()>0 else None,retention_ci95=quant(ret[np.isfinite(ret)]) if np.isfinite(ret).any() else None)
 return out

def paired(allout,raw):
 out=dict(events={},gt_action_A2_minus_A0={})
 for g in ('A1','A2'):
  out['events'][g]={}
  for tol in ('5','10'):
   a=np.array(allout[g]['actions']['events'][tol]['parent_metrics']);b=np.array(allout['A0']['actions']['events'][tol]['parent_metrics']);idx=bootstrap(len(a));d=a[:,2]-b[:,2];f=a[:,3]-b[:,3];ci=quant(d[idx].mean(1));out['events'][g][tol]=dict(delta_f1=float(d.mean()),ci95=ci,delta_false_switches_per_minute=float(f.mean()),false_switch_ci95=quant(f[idx].mean(1)),effective=bool(ci[0]>0 and f.mean()<0))
 for h in (10,25,50,100):
  by={g:[r for r in raw[g]['world'] if r['h']==h and r['method']=='gt_action'] for g in ('A0','A2')};ps=sorted({r['parent'] for r in by['A0']});idx=bootstrap(len(ps));a={g:np.array([[np.mean([r[k] for r in by[g] if r['parent']==p]) for k in ('model','copy')] for p in ps]) for g in by};assert np.array_equal(a['A0'][:,1],a['A2'][:,1]);copy=a['A0'][:,1];d=a['A0'][:,0]-a['A2'][:,0];out['gt_action_A2_minus_A0'][str(h)]=dict(delta_skill=float(d.mean()/copy.mean()),ci95=quant(d[idx].mean(1)/copy[idx].mean(1)),parents=len(ps))
 return out

def main():
 status=json.loads((OUT/'STATUS.json').read_text());assert status['status']=='inference_complete'
 lock=json.loads((OUT/'BEFORE_FORWARD_LOCK.json').read_text());mapping={e['index']:e for e in lock['entries']};raw={g:json.loads((OUT/(g if g=='A0' else g+'_retry02')/'evaluation.json').read_text()) for g in GROUPS};out={};replay={}
 for g in GROUPS:
  x=raw[g];assert len(x['actions'])==211;assert len(x['world'])==(211*3+208)*4;assert x['checkpoint_sha']==lock['checkpoints'][g]['sha'];assert sha(Path(lock['checkpoints'][g]['path']))==x['checkpoint_sha']
  out[g]={}
  for scope in ('all27','original48','demo3','demo4','demo6'):
   keep=lambda r:scope=='all27' or (mapping[r['index']]['original48'] if scope=='original48' else mapping[r['index']]['stratum']==scope)
   aa=[r for r in x['actions'] if keep(r)];ww=[r for r in x['world'] if keep(r)];ac=action_summary(aa);ac['event_window_count']=sum(r['events']['10']['true'] for r in aa);ac['unique_physical_events']=len({tuple(e) for r in aa for e in r['true_event_ids']});out[g][scope]=dict(actions=ac,world=worlds(ww))
  old=json.loads((CONTROL/(g+'_s90')/'evaluation.json').read_text());oldact={r['index']:r for r in old['actions'] if r['split']=='dev'};oldwr={(r['index'],r['h'],r['method']):r for r in old['world']};amax=0.;wmax=0.;event_same=True
  for r in x['actions']:
   if not mapping[r['index']]['original48']:continue
   o=oldact[r['index']]
   for h in ('8','50'):
    for k in ('l1','mse','copy_l1','copy_mse'):amax=max(amax,float(np.max(abs(np.asarray(r['joints'][h][k])-o['joints'][h][k]))))
   event_same &= r['events']==o['events']
  for r in x['world']:
   if mapping[r['index']]['original48'] and r['h']!=10:wmax=max(wmax,abs(r['model']-oldwr[(r['index'],r['h'],r['method'])]['model']))
  replay[g]=dict(max_action_metric_abs_difference=amax,max_world_mse_abs_difference=wmax,events_identical=event_same)
 pair=paired({g:out[g]['all27'] for g in GROUPS},raw);decision=dict(joint_by_group={g:all(out[g]['all27']['actions']['joints'][h]['improvement_ci95'][0]>0 for h in ('8','50')) for g in GROUPS},world_by_group={g:all(out[g]['all27']['world'][h]['delta_ci95'][0]>0 for h in ('25','50','100')) for g in GROUPS},gripper_by_group={g:pair['events'][g]['10']['effective'] for g in ('A1','A2')});decision['gripper_fix_effective']=all(decision['gripper_by_group'].values())
 atomic(OUT/'SUMMARY.json',out);atomic(OUT/'PAIRED.json',pair);atomic(OUT/'DECISION.json',decision);atomic(OUT/'ORIGINAL48_REPLAY.json',replay)
 lines=['## 扩展评估最终结果（以下为最终有效口径）',f'完成：{brt()} BRT。三组固定2000终点，无训练；parent等权bootstrap2000 seed90。211窗/27parent，+10/+25/+50各211窗/27parent，+100为208窗/26parent（demo4短episode 3窗缺+100）；不填补缺失。原48窗/6parent完整子集，单独结果见SUMMARY.json original48。', '事件真值为反馈运动起点代理，沿用训练拟合阈值/起点规则。事件按预测窗口暴露计数，重叠可能重复；无事件parent的P/R/F1按原合同为0。gt_action为下一帧反馈关节+起点两态夹爪；oracle为未来实测阈值两态，二者夹爪时机不同。所有夹爪固定开度沿用训练合同。','\n### 全27parent关节与夹爪','|组|前8 L1/复制 [改善CI]|H50 L1/复制 [改善CI]|前8/H50 MSE skill %|状态准确率/保持|F1 ±5/±10|真事件窗口数/物理去重|','|---|---|---|---|---|---|---|']
 for g in GROUPS:
  a=out[g]['all27']['actions'];j=a['joints'];e=a['events'];fmt=lambda h:f"{j[h]['l1']:.6f}/{j[h]['copy_l1']:.6f} {j[h]['improvement_ci95']}";lines.append(f"|{g}|{fmt('8')}|{fmt('50')}|{100*j['8']['mse_skill']:.3f}/{100*j['50']['mse_skill']:.3f}|{a['state_accuracy']['mean']:.4f}/{a['copy_accuracy']['mean']:.4f}|{e['5']['f1']:.4f}/{e['10']['f1']:.4f}|{a['event_window_count']}/{a['unique_physical_events']}|")
 lines+=['\n|组|容差|P/R/F1|F1 CI|假切换/分钟 [CI]|时误差abs中位/P90、signed中位|匹配数/预测数|','|---|---|---|---|---|---|---|']
 for g in GROUPS:
  for tol,e in out[g]['all27']['actions']['events'].items():lines.append(f"|{g}|±{tol}|{e['precision']:.4f}/{e['recall']:.4f}/{e['f1']:.4f}|{e['f1_ci95']}|{e['false_switches_per_minute']:.3f} {e['false_switches_ci95']}|{e['timing_abs_median']}/{e['timing_abs_p90']}、{e['timing_signed_median']}|{e['matched_events']}/{sum(r['pred'] for r in e['raw'])}|")
 lines+=['\n### 世界头（skill及配对差用百分点pp；保留率%）','|组|h|GT / pred N1 / no-traj / oracle %|N1−no [CI] pp|保留率 %|','|---|---:|---|---|---|']
 for g in GROUPS:
  for h,w in out[g]['all27']['world'].items():lines.append(f"|{g}|{h}|"+'/'.join(f"{100*w['skill'][m]:.3f}" for m in METHODS)+f"|{100*w['delta_skill']:.3f} [{100*w['delta_ci95'][0]:.3f},{100*w['delta_ci95'][1]:.3f}]|{100*w['retention']:.2f}|" if w['retention'] is not None else f'|{g}|{h}|retention unavailable|')
 lines+=['\n### 重点(a)：夹爪配对差','|组−A0|容差|F1差 [CI]|假切换差/分钟 [CI]|预注册有效|','|---|---|---|---|---|']
 for g,v in pair['events'].items():
  for tol,e in v.items():lines.append(f"|{g}−A0|±{tol}|{e['delta_f1']:.5f} {e['ci95']}|{e['delta_false_switches_per_minute']:.3f} {e['false_switch_ci95']}|{e['effective']}|")
 lines+=['\n### 重点(b)：A2−A0 gt_action 世界skill配对差','|h|差 pp|95%CI pp|parent|','|---|---:|---|---:|']
 for h,w in pair['gt_action_A2_minus_A0'].items():lines.append(f"|{h}|{100*w['delta_skill']:.3f}|[{100*w['ci95'][0]:.3f},{100*w['ci95'][1]:.3f}]|{w['parents']}|")
 lines+=['\n### 重点(c)：沿用E102-A预注册机械判定',json.dumps(decision,ensure_ascii=False),'\n### 按demo3/4/6分层附表（不用于增设判据）','|层|组|parent/窗|前8/H50 L1|前8/H50 MSE skill %|状态准确率|F1±5/±10|P/R±10|假切换/分钟±10|事件窗口/去重|','|---|---|---|---|---|---|---|---|---|']
 for scope in ('demo3','demo4','demo6'):
  for g in GROUPS:
   a=out[g][scope]['actions'];j=a['joints'];e=a['events'];lines.append(f"|{scope}|{g}|{a['parents']}/{a['windows']}|{j['8']['l1']:.6f}/{j['50']['l1']:.6f}|{100*j['8']['mse_skill']:.3f}/{100*j['50']['mse_skill']:.3f}|{a['state_accuracy']['mean']:.4f}|{e['5']['f1']:.4f}/{e['10']['f1']:.4f}|{e['10']['precision']:.4f}/{e['10']['recall']:.4f}|{e['10']['false_switches_per_minute']:.3f}|{a['event_window_count']}/{a['unique_physical_events']}|")
 lines+=['\n|层|组|h|GT / pred N1 / no / oracle skill %|N1−no pp [CI]|保留率%|','|---|---|---|---|---|---|']
 for scope in ('demo3','demo4','demo6'):
  for g in GROUPS:
   for h,w in out[g][scope]['world'].items():
    rt='NA' if w['retention'] is None else f"{100*w['retention']:.2f}";lines.append(f'|{scope}|{g}|{h}|'+'/'.join(f"{100*w['skill'][m]:.3f}" for m in METHODS)+f"|{100*w['delta_skill']:.3f} [{100*w['delta_ci95'][0]:.3f},{100*w['delta_ci95'][1]:.3f}]|{rt}|")
 lines+=['\n### 原48窗/6parent单独对照','|组|前8/H50 L1|F1±5/±10|状态准确率|真事件窗口数/物理去重|','|---|---|---|---|---|']
 for g in GROUPS:
  a=out[g]['original48']['actions'];j=a['joints'];e=a['events'];lines.append(f"|{g}|{j['8']['l1']:.6f}/{j['50']['l1']:.6f}|{e['5']['f1']:.4f}/{e['10']['f1']:.4f}|{a['state_accuracy']['mean']:.4f}|{a['event_window_count']}/{a['unique_physical_events']}|")
 lines+=['\n|组|h|GT / pred N1 / no-traj / oracle skill %|N1−no pp [CI]|','|---|---|---|---|']
 for g in GROUPS:
  for h,w in out[g]['original48']['world'].items():lines.append(f'|{g}|{h}|'+ '/'.join(f"{100*w['skill'][m]:.3f}" for m in METHODS)+f"|{100*w['delta_skill']:.3f} [{100*w['delta_ci95'][0]:.3f},{100*w['delta_ci95'][1]:.3f}]|")
 lines+=['\n### 原48重放核对',json.dumps(replay,ensure_ascii=False),'SUMMARY.json另含原48完整指标、各层完整CI、逐关节/逐步夹爪准确率、全部世界skill CI与保留率CI。A0/A1/A2/evaluation.json为逐窗结果与事件标识，evaluator.py为实际执行函数。', '\n### 资源与审计']
 roots=[T/'code_e102x',OUT,REPORT];disk=sum(p.stat().st_size for root in roots for p in ([root] if root.is_file() else root.rglob('*')) if p.is_file());assert disk<=10*1024**3;assert status['GPUh_upper']<=.5
 receipt=dict(status='complete',brt=brt(),GPUh_upper=status['GPUh_upper'],new_bytes=disk,checkpoint_unchanged=True,feature_extraction_bytes=0,decision=decision,report=str(REPORT),segments=status['segments'],eval_source_unchanged=sha(SRC/'e102a_eval.py')==json.loads((OUT/'SOURCE_LOCK.json').read_text())['eval_sha']);assert receipt['eval_source_unchanged'];atomic(OUT/'FINAL_STATUS.json',receipt)
 lines += [json.dumps(receipt,ensure_ascii=False), '本轮仅推理，无训练、无阈值拟合、无选点，无dev模型语言。E93 24个不可评parent仍不可评：18 metadata-only、5时钟合同缺失、1仅74帧不足既有窗口；不扩大为51parent结论。+100缺1parent已明确分母。未完成：无；单seed不估seed变异。世界条件判据与关节判据通过也不等于整套策略/夹爪或实机部署已被验证。需要用户决定：无，本轮不自动部署。']
 with REPORT.open('a') as f:f.write('\n'+'\n'.join(lines)+'\n')
 text=REPORT.read_text().replace('状态：准备中；','状态：COMPLETE；').replace('待核对 E102-A/E93，模型输出前锁定。','已核对并锁定，见最终评分前锁。').replace('待完成。','已完成，见末尾最终有效结果。');REPORT.write_text(text)
 for path,line in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',f"\n### E102-X 完成 | {brt()} BRT\n目标：E102-A三组固定终点扩展dev，只读推理。改动：code_e102x复用E102-A评估器；E93 211窗/27parent含原48，+100为208窗/26parent；下一帧反馈仅RAM读取，0新特征。命令：setsid nohup .../e29/bin/python -u e102x_launch.py；随后python e102x_finalize.py。目录results/e102x_extended_s90_20261115_01；报告notes/e102x_extended_eval_20261115.md。结果：{json.dumps(decision,ensure_ascii=False)}；配对差详PAIRED.json。GPUh上界{status['GPUh_upper']:.6f}，落盘约{disk/1024**3:.4f}GiB。状态COMPLETE；无训练/选点/自动部署。\n"),(T/'notes/worklog.md',f"\n- {brt()} BRT E102-X COMPLETE：27parent/211窗三组扩展评估，+100 26parent/208窗；GPUh≤{status['GPUh_upper']:.6f}，结果results/e102x_extended_s90_20261115_01，报告notes/e102x_extended_eval_20261115.md，判定{json.dumps(decision,ensure_ascii=False)}。\n")]:
  with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(line)
 print(json.dumps(dict(decision=decision,paired=pair,resource=receipt,replay=replay),ensure_ascii=False))
if __name__=='__main__':main()
