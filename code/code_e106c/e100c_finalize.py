import json,collections,fcntl,numpy as np
from pathlib import Path
from e99w2_common import T,CONTROL,ROOT,REPORT,atomic,brt,sha,report,INIT
GROUPS=('C0','C1','C2');H=(10,25,50,100)
def arr(rows,keys):
 ps=sorted({r['parent'] for r in rows});a=np.array([[np.mean([r[k] for r in rows if r['parent']==p]) for k in keys] for p in ps]);assert np.isfinite(a).all();return ps,a

def boot(a):return a[np.random.default_rng(90).integers(len(a),size=(2000,len(a)))].mean(1)
def ci(x):return np.quantile(x,[.025,.975]).tolist()
def fmt(v,c=None,scale=1):return f'{v*scale:.4f}'+(f' [{c[0]*scale:.4f}, {c[1]*scale:.4f}]' if c else '')
def main():
 assert not (CONTROL/'FINAL.json').exists()
 ev={g:json.loads((CONTROL/f'{g}_s90/evaluation.json').read_text()) for g in GROUPS};runs={g:T/f'results/e100c_{g.lower()}_s90_20261115_03' for g in GROUPS};train={g:json.loads((runs[g]/'train_summary.json').read_text()) for g in GROUPS}
 lines=['## 最终结果（每组seed90，固定5000步）','指标沿用E99-W2：先窗内聚合，再parent等权；skill=1−mean_parent_MSE/mean_parent_copy；parent bootstrap2000 seed90，所有配对共用重抽parent。单seed不能估计seed噪声；seed间差不适用，未做双seed平均。','### 单跳主指标及附表','|组|h|条件|窗/parent|skill % [95%CI]|方向cos|','|---|---:|---|---|---|---:|'];summary=[];arrays={}
 for g in GROUPS:
  for h in H:
   for method in ('gt_action','pred_N1','pred_N8','no_traj'):
    rr=[r for r in ev[g]['records'] if r['h']==h and r['method']==method];ps,a=arr(rr,('model','copy','cosine'));arrays[g,h,method]=(ps,a);v=a.mean(0);d=boot(a);skill=1-v[0]/v[1];c=ci(1-d[:,0]/d[:,1]);summary.append(dict(group=g,h=h,method=method,skill=skill,ci95=c,windows=len(rr),parents=len(ps),cosine=v[2]));lines.append(f'|{g}|{h}|{method}|{len(rr)}/{len(ps)}|{fmt(skill,c,100)}|{v[2]:.5f}|')
 lines+=['### 单跳链收益（GT动作，正值为改善）','|比较|h|Δskill pp [95%CI]|','|---|---:|---|'];paired=[]
 for g in ('C1','C2'):
  for h in (25,50,100):
   ps,b=arrays['C0',h,'gt_action'];pp,a=arrays[g,h,'gt_action'];assert ps==pp and np.array_equal(a[:,1],b[:,1]);x=np.stack([b[:,0]-a[:,0],b[:,1]],1);d=boot(x);v=x[:,0].mean()/x[:,1].mean();c=ci(d[:,0]/d[:,1]);paired.append(dict(comparison=g+'-C0',h=h,delta_skill=v,ci95=c));lines.append(f'|{g}−C0|{h}|{fmt(v,c,100)}|')
 lines+=['### 真开环两跳（归一化端点MSE，低为好）','第二跳只读原t及以前的真实信息；k25/50的四历史槽全失效。未来机器人条件来自原计划后缀+末端保持；不把计划状态当观测token。直接2k为相同原锚点和同一计划的单跳预测。','|组|链|窗/parent|开环MSE [CI]|直接MSE [CI]|两跳/直接 [CI]|','|---|---|---|---|---|---|'];chainarr={};chain_summary=[]
 for g in GROUPS:
  for k in (25,50):
   rr=[r for r in ev[g]['chains'] if r['k']==k];ps,a=arr(rr,('model','direct'));assert len(ps)>0;chainarr[g,k]=(ps,a);d=boot(a);v=a.mean(0);row=dict(group=g,k=k,windows=len(rr),parents=len(ps),mse=v[0],direct=v[1],ratio=v[0]/v[1],ci95=ci(d[:,0]),direct_ci95=ci(d[:,1]),ratio_ci95=ci(d[:,0]/d[:,1]));chain_summary.append(row);lines.append(f'|{g}|{k}→{2*k}|{len(rr)}/{len(ps)}|{fmt(v[0],row["ci95"])}|{fmt(v[1],row["direct_ci95"])}|{fmt(row["ratio"],row["ratio_ci95"])}|')
 lines+=['|比较|链|Δ开环MSE [95%CI]（负值改善）|','|---|---|---|'];chainpaired=[]
 for g in ('C1','C2'):
  for k in (25,50):
   ps,b=chainarr['C0',k];pp,a=chainarr[g,k];assert ps==pp;x=(a[:,0]-b[:,0])[:,None];v=float(x.mean());c=ci(boot(x)[:,0]);chainpaired.append(dict(comparison=g+'-C0',k=k,delta_mse=v,ci95=c));lines.append(f'|{g}−C0|{k}→{2*k}|{fmt(v,c)}|')
 lines+=['### 预测重入适配器诊断（不设门、不选点）','NMSE/cos比较reentry(预测图像token)与reentry(真实中间图像token)；self列比较预测输入的adapter与current输出，门仍只依据真实训练输入。','|组|链|窗/parent|预测对真实NMSE|cos|self NMSE|self cos|','|---|---|---|---:|---:|---:|---:|']
 for g in GROUPS:
  for k in (25,50):
   rr=[r for r in ev[g]['diagnostics'] if r['k']==k];ps,a=arr(rr,('nmse','cosine','predicted_adapter_self_nmse','predicted_adapter_self_cosine'));v=a.mean(0);lines.append(f'|{g}|{k}→{2*k}|{len(rr)}/{len(ps)}|'+ '|'.join(f'{x:.5f}' for x in v)+'|')
 old=json.loads((T/'results/e99w2_control_20261115_01/evaluation_02_h100_complete/world_records_merged.json').read_text());lines+=['### C1重跑与E99-W2终点（相同seed）','|h|条件|C1−E99-W2 Δskill pp [CI]|','|---:|---|---|'];oldpaired=[]
 for h in H:
  for method in ('gt_action','pred_N1','pred_N8','no_traj'):
   ps,b=arr([r for r in old if r['camera']=='all' and r['h']==h and r['method']==method],('model','copy'));pp,a=arrays['C1',h,method];assert ps==pp;assert np.allclose(a[:,1],b[:,1],atol=1e-8,rtol=1e-6);x=np.stack([b[:,0]-a[:,0],b[:,1]],1);d=boot(x);v=x[:,0].mean()/x[:,1].mean();c=ci(d[:,0]/d[:,1]);oldpaired.append(dict(h=h,method=method,delta_skill=v,ci95=c));lines.append(f'|{h}|{method}|{fmt(v,c,100)}|')
 lines+=['### 训练损失与适配器门','|组|步|链更新|首32/末32 loss|峰值MiB|门停用|','|---|---:|---:|---|---:|---|'];curves=[]
 for g in GROUPS:
  t=train[g];assert t['steps']==5000;lines.append(f'|{g}|5000|{t["chain_updates"]}|{t["first32"]:.5f}/{t["last32"]:.5f}|{t["peak_mib"]:.2f}|{t["gate_disabled"]}|')
  logs=[json.loads(x) for x in (runs[g]/'updates.jsonl').read_text().splitlines()];assert len(logs)==5000;assert all(np.isfinite(r['loss']) for r in logs)
  if g=='C0':assert all(r['chain_weight']==0 and 'chain_supervised' not in r for r in logs)
  if g=='C2':assert all(r.get('chain_gap',0)==0 for r in logs)
  for k in (25,50):
   for predicted in (False,True):
    for start in range(3001,5001,100):
     rr=[r for r in logs if start<=r['step']<start+100 and (25,50)[r['step']%2]==k and r.get('predicted_reentry')==predicted and r['chain_weight']>0]
     if rr:curves.append(dict(group=g,k=k,predicted=predicted,step=start+49,updates=len(rr),supervised=float(np.mean([r['chain_supervised'] for r in rr])),gap=float(np.mean([r['chain_gap'] for r in rr]))))
 atomic(CONTROL/'chain_curves.json',curves)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(12,7))
 for j,k in enumerate((25,50)):
  for g in ('C1','C2'):
   for pred in (False,True):
    rr=[r for r in curves if r['group']==g and r['k']==k and r['predicted']==pred]
    for row,key in enumerate(('supervised','gap')):axs[row,j].plot([r['step'] for r in rr],[r[key] for r in rr],label=g+(' predicted' if pred else ' teacher'),linestyle='-' if pred else '--');axs[row,j].set_title(f'{k}→{2*k} {key}');axs[row,j].set_xlabel('step')
  for ax in axs[:,j]:ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(CONTROL/'chain_curves.png',dpi=140);plt.close(fig);lines+=['100步分桶曲线：results/e100c_control_20261115_03/chain_curves.png；完整分桶chain_curves.json；C0不执行链因此无曲线。曲线仅训练描述，不作选点依据。']
 single_ok=all(r['ci95'][0]>-.01 for r in paired if r['comparison']=='C2-C0');open_ok=all(r['delta_mse']<0 and r['ci95'][1]<0 for r in chainpaired if r['comparison']=='C2-C0');no_benefit=all(r['ci95'][0]<=0 for r in paired) and all(r['ci95'][1]>=0 for r in chainpaired);adopt=single_ok and open_ok
 conclusion='采用C2' if adopt else ('链对单跳部署无收益，默认去掉' if no_benefit else 'C2未达到采用条件；证据混合，保持本轮对照结论，不按dev追加选型')
 decision=dict(adopt_C2=adopt,single_noninferior=single_ok,open_improved=open_ok,no_chain_default=no_benefit,conclusion=conclusion,single_seed_limitation=True)
 sched=json.loads((CONTROL/'SCHEDULER.json').read_text());assert sched['status']=='complete';prior=json.loads((T/'results/e100c_control_20261115_01/SCHEDULER.json').read_text());prior2=json.loads((T/'results/e100c_control_20261115_02/SCHEDULER.json').read_text());gpu=sched['GPUh_upper']+prior['GPUh_upper']+prior2['GPUh_upper'];assert gpu<=3
 roots=[ROOT,*T.glob('results/e100c_*'),REPORT];disk=sum(p.stat().st_size for root in roots for p in ([root] if root.is_file() else root.rglob('*')) if p.is_file() and not p.is_symlink());assert disk<10*1024**3
 atomic(CONTROL/'statistics.json',dict(summary=summary,paired=paired,chain_summary=chain_summary,chain_paired=chainpaired,old_paired=oldpaired,decision=decision))
 lines+=['## 判定与资源',json.dumps(decision,ensure_ascii=False),f'总GPU时间保守上界（含前两次失败六段、各job全程及30秒清理余量）={gpu:.6f}GPUh；新增落盘={disk/1024**3:.4f}GiB。初始化SHA={sha(INIT)}。', '## 命令、完成边界与决策','命令：PYTHONDONTWRITEBYTECODE=1 .../e29/bin/python -u e100c_launch.py（start_new_session分离，逐run状态文件）；成功run统一_20261115_03，失败_01/_02永久保留。所有评估同e100c_eval.py，训练前SHA锁见PRETRAIN_LOCK.json。','已完成三组单seed90固定终点、四时域单跳、两链开环及直接对照、适配器诊断、原链重跑比较、parent配对CI和训练曲线。用户取消seed91，因此没有seed间噪声估计；dev仅6parent限制泛化。无剩余授权内工作；无需用户追加决定。','首次错误导入的影响和ledger修复见前文事故记录；旧模型与旧指标未覆盖。']
 report('\n'.join(lines));final=dict(status='COMPLETE',brt=brt(),decision=decision,GPUh_upper=gpu,new_bytes=disk,report=str(REPORT),seed=90,groups=3);atomic(CONTROL/'FINAL.json',final)
 for path,s in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',f'\n### E100-C 完成 | {brt()} BRT\n目标/改动：C0无链、C1原链、C2删gap+真开环，同初始化seed90各5000步。命令e100c_launch.py；目录code_e100c/results/e100c_*_03；结果{conclusion}；GPUh≤{gpu:.6f}，新增{disk/1024**3:.4f}GiB；报告notes/e100c_chain_fix_20261115.md。首启路径污染失败已计费并修复，旧ledger撤销本会话active字段，旧ckpt/指标未写。状态COMPLETE。\n'),(T/'notes/worklog.md',f'\n- {brt()} BRT E100-C完成：三组单seed90×5000，{conclusion}；GPUh≤{gpu:.6f}，报告e100c_chain_fix_20261115.md。\n')]:
  with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(s)
 print(json.dumps(final,ensure_ascii=False))
if __name__=='__main__':main()
