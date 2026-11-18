import json,numpy as np
from e103d_runtime import *
from e104p_metrics import worlds,action_summary,bootstrap,quant
GROUPS=('A2','F1-pre','F0','F1','F2');METHODS=('gt_action','pred_N1','no_traj','oracle')
def main():
 lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());mapping={e['index']:e for e in lock['entries']};raw={g:json.loads((CONTROL/('eval_'+g)/'evaluation.json').read_text()) for g in GROUPS};summary={};pairs={}
 for g,x in raw.items():
  assert x['checkpoint_sha']==sha(Path(lock['checkpoints'][g]));assert len(x['actions'])==211 and len(x['world'])==(211*3+208)*4;summary[g]={}
  for scope in ('all27','original48'):
   aa=[r for r in x['actions'] if scope=='all27' or mapping[r['index']]['original48']];ww=[r for r in x['world'] if scope=='all27' or mapping[r['index']]['original48']];ac=action_summary(aa);ac['event_window_count']=sum(r['events']['10']['true'] for r in aa);ac['unique_physical_events']=len({tuple(e) for r in aa for e in r['true_event_ids']});summary[g][scope]=dict(actions=ac,world=worlds(ww))
 for target,base in [('F1','F0'),('F1','F2'),('F1-pre','A2'),('F0','A2'),('F2','F0'),('F2','A2')]:
  key=target+'-'+base;pairs[key]={}
  for scope in ('all27','original48'):
   pair=dict(world={},joint={},events={});keep=lambda r:scope=='all27' or mapping[r['index']]['original48']
   for h in (10,25,50,100):
    pair['world'][str(h)]={}
    for method in METHODS:
     rr={g:[r for r in raw[g]['world'] if keep(r) and r['h']==h and r['method']==method] for g in (target,base)};ps=sorted({r['parent'] for r in rr[base]});idx=bootstrap(len(ps));a={g:np.array([[np.mean([r[k] for r in rr[g] if r['parent']==p]) for k in ('model','copy')] for p in ps]) for g in rr};assert np.array_equal(a[target][:,1],a[base][:,1]);d=a[base][:,0]-a[target][:,0];copy=a[base][:,1];pair['world'][str(h)][method]=dict(delta_skill=float(d.mean()/copy.mean()),ci95=quant(d[idx].mean(1)/copy[idx].mean(1)),parents=len(ps))
   aa={g:[r for r in raw[g]['actions'] if keep(r)] for g in (target,base)};ps=sorted({r['parent'] for r in aa[base]});idx=bootstrap(len(ps))
   for h in ('8','50'):
    a={g:np.array([np.mean([r['joints'][h]['l1'] for r in aa[g] if r['parent']==p]) for p in ps]) for g in aa};d=a[target]-a[base];pair['joint'][h]=dict(delta_l1=float(d.mean()),ci95=quant(d[idx].mean(1)),relative_degradation=float(a[target].mean()/a[base].mean()-1),relative_ci95=quant(a[target][idx].mean(1)/a[base][idx].mean(1)-1))
   for tol in ('5','10'):
    a={g:np.array(summary[g][scope]['actions']['events'][tol]['parent_metrics']) for g in aa};d=a[target][:,2]-a[base][:,2];fd=a[target][:,3]-a[base][:,3];pair['events'][tol]=dict(delta_f1=float(d.mean()),ci95=quant(d[idx].mean(1)),relative_degradation=float(1-a[target][:,2].mean()/a[base][:,2].mean()) if a[base][:,2].mean()>0 else None,delta_false_switches=float(fd.mean()),false_switch_ci95=quant(fd[idx].mean(1)))
   pairs[key][scope]=pair
 primary=[pairs[k]['all27'] for k in ('F1-F0','F1-F2')];same=[h for h in ('25','50','100') if all(p['world'][h]['no_traj']['ci95'][0]>0 for p in primary)];checks={}
 for k,p in zip(('F1-F0','F1-F2'),primary):checks[k]=dict(positive_no_traj=[h for h in ('25','50','100') if p['world'][h]['no_traj']['ci95'][0]>0],gt_noninferior=all(p['world'][h]['gt_action']['ci95'][0]>-.01 for h in ('25','50','100')),joint_noninferior=all(p['joint'][h]['relative_degradation']<=.05 for h in ('8','50')),gripper_noninferior=all(p['events'][h]['relative_degradation'] is not None and p['events'][h]['relative_degradation']<=.05 for h in ('5','10')))
 effective=len(same)>=2 and all(all(c[k] for k in ('gt_noninferior','joint_noninferior','gripper_noninferior')) for c in checks.values());only0=len(checks['F1-F0']['positive_no_traj'])>=2 and len(checks['F1-F2']['positive_no_traj'])<2
 verdict='外部预训练有效' if effective else '增益可由多训练步数解释' if only0 else '未满足外部预训练有效判据'
 decision=dict(verdict=verdict,effective=effective,same_positive_horizons=same,checks=checks,bootstrap=2000,seed=90,main='27parent; h10026parent',endpoint_selection=False)
 atomic(CONTROL/'SUMMARY.json',summary);atomic(CONTROL/'PAIRED.json',pairs);atomic(CONTROL/'DECISION.json',decision)
 lines=['## 固定终点评估结果','主集211窗/27parent；+100为208窗/26parent。附表为原48窗/6parent。所有汇总parent等权，配对bootstrap2000 seed90。pred_N1为每个checkpoint的A2确定性动作，参数化及脚本一致。状态/事件为原训练拟合合同和反馈运动起点代理；49个窗口真事件、48个去重物理事件。无模型语言、无新阈值、无选点。']
 for scope in ('all27','original48'):
  lines+=['\n### '+scope,'|组|h|GT / pred / no_traj / oracle skill %|','|---|---:|---|']
  for g in GROUPS:
   for h,w in summary[g][scope]['world'].items():lines.append(f'|{g}|{h}|'+ '/'.join(f"{100*w['skill'][m]:.3f}" for m in METHODS)+'|')
  lines+=['\n|组|前8/H50关节L1|相对复制改善% 前8/H50|状态准确率|事件F1 ±5/±10|假切换/分钟 ±5/±10|','|---|---|---|---|---|---|']
  for g in GROUPS:
   a=summary[g][scope]['actions'];j=a['joints'];e=a['events'];lines.append(f"|{g}|{j['8']['l1']:.6f}/{j['50']['l1']:.6f}|"+'/'.join(f"{100*j[h]['improvement']/j[h]['copy_l1']:.2f}" for h in ('8','50'))+f"|{a['state_accuracy']['mean']:.4f}|{e['5']['f1']:.4f}/{e['10']['f1']:.4f}|{e['5']['false_switches_per_minute']:.3f}/{e['10']['false_switches_per_minute']:.3f}|")
 lines+=['\n## 配对差（主27parent；skill百分点pp，95%CI）','|比较|h|no_traj差 [CI]|gt_action差 [CI]|','|---|---|---|---|']
 for key in pairs:
  for h in ('10','25','50','100'):
   fmt=lambda m: f"{100*pairs[key]['all27']['world'][h][m]['delta_skill']:.3f} ["+','.join(f'{100*x:.3f}' for x in pairs[key]['all27']['world'][h][m]['ci95'])+']'
   lines.append(f'|{key}|{h}|{fmt("no_traj")}|{fmt("gt_action")}|')
 lines+=['\n### 动作/夹爪非劣诊断','|比较|L1相对劣化% 前8/H50|事件F1相对劣化% ±5/±10|','|---|---|---|']
 for key in pairs:
  p=pairs[key]['all27'];lines.append('|'+key+'|'+'/'.join(f"{100*p['joint'][h]['relative_degradation']:.2f}" for h in ('8','50'))+'|'+'/'.join(f"{100*p['events'][h]['relative_degradation']:.2f}" for h in ('5','10'))+'|')
 lines+=['\n## 预注册判定',json.dumps(decision,ensure_ascii=False),'\n关节复制改善绝对CI、所有方法skill CI、所有动作/事件完整指标见SUMMARY.json；所有配对差含原48附表见PAIRED.json。F2−F0、F2−A2仅描述多训练收益，不选点。F1-pre−A2用于直接迁移和遗忘诊断。单seed，不估计seed变异；仅冻结dev结论，不声称实机成功。']
 report('\n'.join(lines));print(json.dumps(decision,ensure_ascii=False))
if __name__=='__main__':main()
