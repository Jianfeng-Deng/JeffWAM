import os,sys,json,collections,time,hashlib,fcntl
from pathlib import Path
from datetime import datetime,timezone,timedelta
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');sys.dont_write_bytecode=True
import numpy as np
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e108g_diag_20261116_01';R=T/'notes/e108g_gripper_diagnosis_20261116.md'
sys.path.insert(0,str(T/'code_e106c'))
from e102a_eval import events,match

def put(n,x):
 p=O/n;assert not p.exists(),p;p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def report(s):
 with R.open('a') as f:f.write('\n'+s+'\n')
def main():
 assert (O/'ANALYSIS_COMPLETE.json').exists()
 rr=json.loads((O/'predictions.json').read_text());dev=[r for r in rr if r['split']=='dev'];g0=json.loads((O/'G0_all27.json').read_text());g1=json.loads((O/'G1_RESULTS.json').read_text());grid=json.loads((O/'TRAIN_SELECTED_LOCK.json').read_text());exposure=[]
 # Validate exact event totals, parent averaging, training-only argmax and bootstrap independently.
 for fam,cfg in grid['selected'].items():
  cand=[v for v in grid['grid'] if v['config']['family']==fam];expected=max(cand,key=lambda v:(v['train_f1'],-v['train_false_per_min']));assert cfg==expected['config']
 for subset,x in g1.items():
  for fam,v in x.items():
   if fam=='baseline':continue
   for tol in ('5','10','20'):
    a=np.array(v['metrics'][tol]['parent_metrics']);b=np.array(x['baseline'][tol]['parent_metrics']);idx=np.random.default_rng(90).integers(len(a),size=(2000,len(a)));delta=(a-b)[idx].mean(1)
    for j,key in enumerate(('f1','false_per_min','recall')):assert np.allclose(np.quantile(delta[:,j],[.025,.975]),v['paired'][tol][key]['ci95'],rtol=0,atol=1e-14)
 for subset,rows in [('all27',dev),('original48',[r for r in dev if r['original48']])]:
  x=json.loads((O/('G0_'+subset+'.json')).read_text())
  for c in x['cells']:
   pars=set()
   for r in rows:
    delta=abs(np.diff(np.array(r['width'])[:,c['gripper']]))*1000
    if c['segment']=='all' or (delta<=c['static_threshold_mm']).any() and c['segment']=='static' or (delta>c['static_threshold_mm']).any() and c['segment']=='moving':pars.add(r['parent'])
   exposure.append(dict(subset=subset,**c,exposure_parents=len(pars)))
 put('G0_EXPOSURE_AUDIT.json',exposure)
 fps=g0['false_events'];duration=dict(total=len(fps),short5=sum(f['return_frames'] is not None and f['return_frames']<=5 for f in fps),known_exceeds5=sum((f['return_frames'] is not None and f['return_frames']>5) or (f['return_frames'] is None and 50-f['time']>=5) for f in fps),unknown5=sum(f['return_frames'] is None and 50-f['time']<5 for f in fps),local_ge2_crossings=sum(f['crossings_local']>=2 for f in fps),static05_local_ge2=sum(f['crossings_local']>=2 and f['static_mm']['0.5'] for f in fps),near_threshold_local_ge2=sum(f['near_crossings_local']>=2 for f in fps))
 put('FALSE_DURATION_AUDIT.json',duration)
 report('## G0 复核与量化解释\n主集±10：TP=16、FN=33、FP=51；左爪6/8/13（真实14，少于20仅报数），右爪10/25/38（真实35）。F1从±3的0.082716、±5的0.149383增至±10的0.245679、±20的0.387860，说明时序偏移确有贡献；即便±20仍漏23/49、假检41/67，不能只归为时序对不准。±10匹配有符号偏移Q25/median/Q75=−4.25/0/7.25帧，中位数0只描述已匹配16项，不能据此声称时序准确。\n51个FP中39个（76.47%）发生于实测单爪静止≤0.5mm帧；39个预测跨阈值前后均在[.4,.6]，但局部[t−5,t+10]含≥2次阈值往返的仅9个（17.65%），其中静止段6个。阈值附近单次跨越不等同于逐帧反复抖动。5帧内回切仅'+str(duration['short5'])+'/51（'+f"{100*duration['short5']/51:.2f}%"+'）；已知超过5帧'+str(duration['known_exceeds5'])+'，窗口尾部不足5帧而未知'+str(duration['unknown5'])+'。共有42个未在窗内回切而右截尾，不能编造完整持续时间；按最悲观未知全部短回切计，上界仍为'+f"{100*(duration['short5']+duration['unknown5'])/51:.2f}%"+'，不支持“大部分假检是短时抖动”。首帧FP=5，其中当前实测状态与起点代理初始状态不一致仅1项，边界定义不是全部假检的解释。\n前文分层表的parent列统一指集合parent/有真事件parent；实际存在该静止/运动帧的暴露parent另列以下复核表。\n|集合|侧|静止阈值mm|分层|实际暴露parent|有真事件parent|真事件数|\n|---|---|---|---|---|---|---|\n'+'\n'.join(f"|{c['subset']}|{c['gripper']}|{c['static_threshold_mm']}|{c['segment']}|{c['exposure_parents']}|{c['event_parents']}|{c['true']}|" for c in exposure if c['tol']==10))
 # Full original event metric for label-jitter sensitivity, including finite-window boundary loss and shared unique-event shifts.
 unique=sorted({(r['cid'],r['anchor']+t,k,d) for r in dev for k in range(2) for t,d in events(np.array(r['labels'])[:,k],r['label_init'][k])});uid={key:i for i,key in enumerate(unique)};parents=sorted({r['parent'] for r in dev});pi={p:i for i,p in enumerate(parents)};prepared=[]
 for r in dev:
  te=[events(np.array(r['labels'])[:,k],r['label_init'][k]) for k in range(2)];ids=[[uid[(r['cid'],r['anchor']+t,k,d)] for t,d in te[k]] for k in range(2)];prepared.append((r,te,ids))
 rng=np.random.default_rng(90);sens=[]
 for jitter in (1,3,5,10):
  values=[]
  for rep in range(500):
   noise=rng.integers(-jitter,jitter+1,len(unique));counts=np.zeros((4,len(parents),3))
   for r,te,ids in prepared:
    for k in range(2):
     pe=sorted((t+int(noise[ind]),d) for (t,d),ind in zip(te[k],ids[k]) if 1<=t+int(noise[ind])<=50)
     for j,tol in enumerate((3,5,10,20)):counts[j,pi[r['parent']]]+=np.array([len(match(pe,te[k],tol)),len(pe),len(te[k])])
   den=counts[:,:,1]+counts[:,:,2];f=np.divide(2*counts[:,:,0],den,out=np.zeros_like(den),where=den>0);values.append(f.mean(1))
  a=np.array(values);sens.append(dict(jitter=jitter,f1_mean=a.mean(0).tolist(),simulation_q025_q975=np.quantile(a,[.025,.975],axis=0).tolist()))
 put('LABEL_JITTER_EXACT_METRIC.json',dict(tolerances=[3,5,10,20],replicates=500,seed=90,unique_events=len(unique),results=sens,method='perturb real onset timestamps, same direction DP matching, drop outside-window predictions, shared jitter for repeated unique events, original parent equal zero conventions'))
 report('## 标签抖动补充：严格复用原事件F1\n为避免将孤立事件命中概率与原parent等权F1混淆，另做完整事件级重算：同一唯一事件在重叠窗共享整数均匀抖动，超出1..50的合成预测事件直接缺失；保留方向并按原DP匹配/原零事件parent=0约定计算，seed90每档500次。真实onset测试标签保持不变；仅构造诊断参照预测。前面的孤立事件表仅是乐观命中率近似，本表为同口径敏感性。无抖动的理想onset预测，在本27parent均有事件的情况下F1=1。\n|时间抖动帧|F1±3|F1±5|F1±10|F1±20|\n|---|---|---|---|---|\n'+'\n'.join('|±'+str(v['jitter'])+'|'+'|'.join(f'{x:.6f}' for x in v['f1_mean'])+'|' for v in sens)+'\n这不是已知标签噪声幅度，也不是模型可达性能的估计。完整模拟分位数见LABEL_JITTER_EXACT_METRIC.json，非bootstrap置信区间。')
 # Final integrity, resource receipt, and append-only shared logs with locks.
 ck=T/'results/e106c_f2_s90_20261115_02/ckpt_20000.pt'
 with ck.open('rb') as f:cksha=hashlib.file_digest(f,'sha256').hexdigest()
 assert cksha==json.loads((O/'COLLECT_COMPLETE.json').read_text())['checkpoint_sha']
 assert not any(p.suffix=='.pt' for p in O.rglob('*'))
 txt=R.read_text().replace('状态：审计与基线复现。','状态：COMPLETE（仅诊断与规则评估，无训练/部署）。').replace('待完成。','已完成，结果见后续对应章节。').replace('暴露parent/有真事件parent','集合parent/有真事件parent')
 # Verify all parents have a true event before making the ideal-reference statement.
 eventpars={r['parent'] for r,te,ids in prepared if any(te)}
 if len(eventpars)!=len(parents):txt=txt.replace('在本27parent均有事件的情况下F1=1',f'按零事件parent=0口径F1={len(eventpars)/len(parents):.6f}（{len(eventpars)}/{len(parents)}个parent有真事件）')
 R.write_text(txt)
 now=datetime.now(timezone(timedelta(hours=-3))).isoformat();size=sum(p.stat().st_size for root in (T/'code_e108g',O) for p in root.rglob('*') if p.is_file())+R.stat().st_size;assert size<2*1024**3
 report(f'## 完成凭据\n完成时间：{now} BRT。新增落盘{size} bytes（约{size/1024**2:.2f}MiB），≤2GiB；GPUh=0（失败段亦0）；只CPU默认模型推理81.65秒，无训练/新checkpoint/下载。ckpt SHA256复核仍为{cksha}。\n命令：`setsid nohup env PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u code_e108g/collect.py > results/e108g_diag_20261116_01/collect.log 2>&1 < /dev/null &`；同样分离执行analyze.py和audit.py，各写完成/状态文件。目录不存在检查在首次创建前完成；旧代码和缓存仅只读import，禁写pyc。只新增code_e108g、results/e108g_diag_20261116_01、本报告及两处共享日志追加，不写E98-C目录、不碰他人进程。原AGENTS旧天机/MuJoCo条款与用户本轮明确Piper规则冲突处按用户规则执行。\n复核：全部23配置前缀截断/未来概率扰动因果测试通过；事件索引归因DP与原match偏移逐项相同；四家族训练选参argmax与原parent bootstrap2000 seed90区间独立复算通过；211窗基线事件和状态逐项一致。')
 line=f'\n## E108-G | {now} BRT | COMPLETE\n目标：默认20k夹爪G0误差分解、G1模型外因果解码；无训练/架构改动。\n改动/目录：code_e108g/、results/e108g_diag_20261116_01/；报告notes/e108g_gripper_diagnosis_20261116.md。\n命令：setsid nohup .../venvs/e29/bin/python -u code_e108g/collect.py / analyze.py / audit.py（分别执行，日志与状态在新结果目录）。\n结果：211窗/27parent与48窗附表；训练2320窗/145parent锁定23配置选参；基线逐项复现F1±5=.149383、±10=.245679，FP/min±10=8.5；TP/FN/FP=16/33/51；5帧内回切4/51。四家族均失败预注册；dwell8/EMA.8的ΔF1@10=0 [0,0]、假检减少3.92%；滞回.45/.55及组合召回相对降6.67%。\n结论：瓶颈不在因果解码，需要在原架构内通过目标/数据层面解决。仅给候选不实施。局限：无人工阶段/实发命令，标签上限不可识别；窗内寿命截尾。\n资源：GPUh=0，新增约{size/1024**2:.2f}MiB<2GiB；旧ckpt SHA复核一致，无新模型/部署。状态：COMPLETE。\n'
 for path,s in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',line),(T/'notes/worklog.md',f'\n- {now} BRT E108-G COMPLETE：夹爪G0/G1，基线F1@10=.245679、FP/min=8.5；四家族因果解码均未达预注册；GPUh=0，新增≈{size/1024**2:.2f}MiB；报告 notes/e108g_gripper_diagnosis_20261116.md。\n')]:
  with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(s);f.flush();fcntl.flock(f,fcntl.LOCK_UN)
 put('FINAL_STATUS.json',dict(status='complete',finished_brt=now,GPUh=0,new_bytes=size,checkpoint_sha=cksha,report=str(R),independent_numeric_audit='passed',shared_logs_appended=True))
 (O/'status.json').write_text(json.dumps(dict(status='complete',finished_brt=now,GPUh=0)))
 print(json.dumps(dict(status='complete',new_bytes=size,GPUh=0,label_jitter=sens),ensure_ascii=False))
if __name__=='__main__':main()
