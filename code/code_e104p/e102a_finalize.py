import collections,json,numpy as np
from e99w2_common import *
GROUPS=('A0','A1','A2')
def quant(x):return np.quantile(x,[.025,.975]).tolist()
def action_summary(rows):
 parents=sorted({r['parent'] for r in rows});by=[[r for r in rows if r['parent']==p] for p in parents];idx=np.random.default_rng(90).integers(len(parents),size=(2000,len(parents)));out=dict(parents=len(parents),windows=len(rows),joints={},events={})
 for h in ('8','50'):
  arr={k:np.array([np.mean([r['joints'][h][k] for r in rr],0) for rr in by]) for k in ('l1','mse','copy_l1','copy_mse')};l1=arr['l1'].mean(1);cl=arr['copy_l1'].mean(1);mse=arr['mse'].mean(1);cm=arr['copy_mse'].mean(1);improve=cl-l1
  out['joints'][h]=dict(l1=float(l1.mean()),copy_l1=float(cl.mean()),improvement=float(improve.mean()),improvement_ci95=quant(improve[idx].mean(1)),mse_skill=float(1-mse.mean()/cm.mean()),mse_skill_ci95=quant(1-mse[idx].mean(1)/cm[idx].mean(1)),dimensions=[dict(dim=d,l1=float(arr['l1'][:,j].mean()),copy_l1=float(arr['copy_l1'][:,j].mean()),mse_skill=float(1-arr['mse'][:,j].mean()/arr['copy_mse'][:,j].mean()),improvement_ci95=quant((arr['copy_l1'][:,j]-arr['l1'][:,j])[idx].mean(1))) for j,d in enumerate(['L1','L2','L3','L4','L5','L6','R1','R2','R3','R4','R5','R6'])])
 for tol in ('5','10'):
  metrics=[];errors=[];raw=[]
  for rr in by:
   tp=sum(r['events'][tol]['tp'] for r in rr);pr=sum(r['events'][tol]['pred'] for r in rr);gt=sum(r['events'][tol]['true'] for r in rr);mins=sum(r['events'][tol]['minutes'] for r in rr);p=tp/pr if pr else 0.;rec=tp/gt if gt else 0.;f1=2*tp/(pr+gt) if pr+gt else 0.;fake=(pr-tp)/mins;metrics.append([p,rec,f1,fake]);raw.append(dict(parent=rr[0]['parent'],tp=tp,pred=pr,true=gt,minutes=mins));errors.extend(e for r in rr for e in r['events'][tol]['errors'])
  a=np.array(metrics);out['events'][tol]=dict(precision=float(a[:,0].mean()),recall=float(a[:,1].mean()),f1=float(a[:,2].mean()),f1_ci95=quant(a[idx,2].mean(1)),false_switches_per_minute=float(a[:,3].mean()),false_switches_ci95=quant(a[idx,3].mean(1)),timing_abs_median=float(np.median(np.abs(errors))) if errors else None,timing_abs_p90=float(np.quantile(np.abs(errors),.9)) if errors else None,timing_signed_median=float(np.median(errors)) if errors else None,raw=raw,parent_metrics=a.tolist(),matched_events=len(errors))
 for name in ('state_accuracy_by_gripper','copy_accuracy_by_gripper'):
  a=np.array([np.mean([r[name] for r in rr],0) for rr in by]);out[name]=dict(per_step=a.mean(0).tolist(),per_gripper=a.mean((0,1)).tolist())
 for name in ('state_accuracy','copy_accuracy'):
  a=np.array([np.mean([r[name] for r in rr],0) for rr in by]);out[name]=dict(mean=float(a.mean()),per_step=a.mean(0).tolist(),ci95=quant(a.mean(1)[idx].mean(1)))
 out['hold_events']=dict(precision=0,recall=0,f1=0,false_switches_per_minute=0)
 return out

def main():
 allout={};action_dec={};world_dec={};event_dec={}
 for g in GROUPS:
  art=CONTROL/f'{g}_s90';x=json.loads((art/'evaluation.json').read_text());summary={s:action_summary([r for r in x['actions'] if r['split']==s]) for s in ('dev','train64')};wr=x['world'];parents=sorted({r['parent'] for r in wr});idx=np.random.default_rng(90).integers(6,size=(2000,6));world={}
  for h in (25,50,100):
   a={}
   for method in ('gt_action','no_traj','pred_N1','oracle'):
    rows=[r for r in wr if r['h']==h and r['method']==method];assert len(rows)==48 and len({r['parent'] for r in rows})==6
    a[method]=np.array([[np.mean([r[k] for r in rows if r['parent']==p]) for k in ('model','copy')] for p in parents])
   copy=a['no_traj'][:,1];den=copy.mean();delta=(a['no_traj'][:,0]-a['pred_N1'][:,0]);bs=delta[idx].mean(1)/copy[idx].mean(1);od=(a['no_traj'][:,0]-a['oracle'][:,0]).mean();world[str(h)]=dict(skill={k:float(1-v[:,0].mean()/den) for k,v in a.items()},delta_skill=float(delta.mean()/den),delta_ci95=quant(bs),retention=float(delta.mean()/od) if od>0 else None)
  summary.update(world=world,latency=x['latency']);atomic(art/'summary.json',summary);allout[g]=summary;action_dec[g]=all(summary['dev']['joints'][str(h)]['improvement_ci95'][0]>0 for h in (8,50));world_dec[g]=all(world[str(h)]['delta_ci95'][0]>0 for h in (25,50,100))
 for g in ('A1','A2'):
  a=np.array(allout[g]['dev']['events']['10']['parent_metrics']);base=np.array(allout['A0']['dev']['events']['10']['parent_metrics']);idx=np.random.default_rng(90).integers(6,size=(2000,6));diff=a[:,2]-base[:,2];fdiff=a[:,3]-base[:,3];ci=quant(diff[idx].mean(1));event_dec[g]=dict(delta_f1=float(diff.mean()),ci95=ci,delta_false_switches_per_minute=float(fdiff.mean()),false_switch_delta_ci95=quant(fdiff[idx].mean(1)),effective=bool(ci[0]>0 and fdiff.mean()<0))
 decision=dict(joint_fix_effective=action_dec['A2'],joint_by_group=action_dec,gripper_fix_effective=all(x['effective'] for x in event_dec.values()),gripper_by_group=event_dec,deployable_world=world_dec,single_seed_limitation=True,dev_parents=6,brt=brt());atomic(CONTROL/'DECISION.json',decision);atomic(CONTROL/'SUMMARY.json',allout)
 lines=['## 固定2000终点评估与预注册判定','parent等权；同parent配对bootstrap2000，seed90。前8/H50是前缀均值。skill=1−parent等权MSE/复制MSE。仅固定终点；无dev选点。', '|组|前8关节L1/复制|改善95%CI|H50关节L1/复制|改善95%CI|前8/H50 MSE skill|','|---|---|---|---|---|---|']
 for g,s in allout.items():
  a=s['dev']['joints']['8'];b=s['dev']['joints']['50'];lines.append(f"|{g}|{a['l1']:.6f}/{a['copy_l1']:.6f}|{a['improvement_ci95']}|{b['l1']:.6f}/{b['copy_l1']:.6f}|{b['improvement_ci95']}|{a['mse_skill']:.4f}/{b['mse_skill']:.4f}|")
 lines+=['\n|组|状态准确率/保持|F1±5/±10|Precision/Recall±10|假切换/分钟±10|时误差median/P90帧±10|','|---|---|---|---|---|---|']
 for g,s in allout.items():
  e=s['dev']['events'];x=e['10'];lines.append(f"|{g}|{s['dev']['state_accuracy']['mean']:.4f}/{s['dev']['copy_accuracy']['mean']:.4f}|{e['5']['f1']:.4f}/{x['f1']:.4f}|{x['precision']:.4f}/{x['recall']:.4f}|{x['false_switches_per_minute']:.3f}|{x['timing_abs_median']}/{x['timing_abs_p90']}|")
 lines+=['\n|组|h|GT/pred N1/no-traj/oracle skill|N1−no-traj CI|保留率|','|---|---|---|---|---|']
 for g,s in allout.items():
  for h,w in s['world'].items():lines.append(f"|{g}|{h}|{w['skill']}|{w['delta_skill']:.5f} {w['delta_ci95']}|{w['retention']}|")
 for split in ('dev','train64'):
  for g,s in allout.items():
   lines+=['\n### '+g+' '+split+'逐维关节表','|维|前8 L1/复制|前8 MSE skill|H50 L1/复制|H50 MSE skill|','|---|---|---|---|---|']
   for a,b in zip(s[split]['joints']['8']['dimensions'],s[split]['joints']['50']['dimensions']):lines.append(f"|{a['dim']}|{a['l1']:.6f}/{a['copy_l1']:.6f}|{a['mse_skill']:.4f}|{b['l1']:.6f}/{b['copy_l1']:.6f}|{b['mse_skill']:.4f}|")
   e=s[split];lines.append('夹爪完整逐步准确率、±5/±10 P/R/F1、事件误差、假切换及CI见各组summary.json；本集合摘要：'+json.dumps({k:e[k] for k in ('state_accuracy','copy_accuracy','events')},ensure_ascii=False))
 lines+=['\n### 延迟（缓存特征，batch1，100次）',json.dumps({g:s['latency'] for g,s in allout.items()},ensure_ascii=False),'\n### 预注册机械判定',json.dumps(decision,ensure_ascii=False),'\n事件时刻是反馈运动起点代理，非实发命令真值。事件按每个50帧预测窗、每爪同方向做最大基数/最小时误差的一一匹配；分钟为预测窗暴露时长（50/30秒），重叠窗可能重复统计。预测从当前实测阈值状态起算，真事件从anchor的起点标签起算；已在进行中的运动可能造成首帧边界假切换，已原样计入，未按dev调整。保持基线事件P/R/F1=0、假切换=0。', '\n未完成：无（若调度器随后失败则以SCHEDULER.json为准）。单seed与6parent不估计seed噪声；结论仅限固定dev。用户决定：本轮无待决事项；不自动部署。']
 report('\n'.join(lines));print(json.dumps(decision,ensure_ascii=False))
if __name__=='__main__':main()
