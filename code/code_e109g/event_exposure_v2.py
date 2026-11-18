import os,sys,json,collections
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';R=T/'notes/e109g_gripper_generalization_20261116.md';import numpy as np

def build():
 ready=json.loads((O/'INSPECT.json').read_text())['ready'];rows=json.loads(Path(ready['windows']['P3']).read_text());counts=np.load(O/'sampling_counts.npz');weights=counts['p3']+counts['a2'];by=collections.defaultdict(list)
 for r,w in zip(rows,weights):by[r['canonical_id']].append((r['anchor'],int(w)))
 full=json.loads((O/'FULL_CAPTURE_STATS.json').read_text());out={}
 for cid,f in full.items():
  if f['split']=='dev':continue
  aa=sorted(by[cid]);anchors=np.array([a for a,w in aa]);cs=np.r_[0,np.cumsum([w for a,w in aa])]
  for t,k,d in f['events']:
   lo=np.searchsorted(anchors,t-50,side='left');hi=np.searchsorted(anchors,t,side='left');out[f'{cid}:{t}:{k}:{d}']=int(cs[hi]-cs[lo])
 path=O/'EVENT_EXPOSURE_COUNTS.json';assert not path.exists();path.write_text(json.dumps(dict(counts=out,definition='exact A2+Piper20k P3 sampled windows containing this onset at future offset1..50; direct BCE target incidence, not inferred observer memory'),indent=2));print('counted',len(out),flush=True)
def score():
 sys.path.insert(0,str(T/'code_e108g'));import analyze as ev
 ex=json.loads((O/'EVENT_EXPOSURE_COUNTS.json').read_text())['counts'];truth=json.loads((O/'WINDOW_TRUTH.json').read_text());out={};lines=['## 3 补充：事件本身是否被监督过','利用已核验的逐窗口精确采样计数，累计每个物理onset处于P3未来1..50帧目标中的次数（A2+Piper20k，未把P1算作直接监督）。seen定义为次数>0，事先固定；不按模型预测选阈值。相同事件在重叠评估窗共享身份，但下表匹配TP/召回按原窗口计数，另报唯一真事件数。FP无法唯一归给真实事件的seen状态，因此此分层只给recall，完整F1沿用主表。','|集合/20k|事件监督|真事件窗口数|唯一事件|TP@10|窗口微召回|有事件parent|','|---|---|---:|---:|---:|---:|---:|']
 for s in ['train16','train_aligned']:
  logits=np.load(O/f'logits_20000_{s}.npz')['logits'];rec={k:dict(true=0,tp=0,ids=set(),parents=set(),exposures=[]) for k in ['seen','unseen']}
  for r,log in zip(truth[s],logits):
   for k,(pe,te,mm) in enumerate(ev.eventrow(r,(log>0).astype(int),10,True)['details']):
    hit={j for i,j in mm}
    for j,(t,d) in enumerate(te):
     key=f"{r['cid']}:{r['anchor']+t}:{k}:{d}";assert key in ex, ('event missing physical source',key);count=ex[key];v=rec['seen' if count>0 else 'unseen'];v['true']+=1;v['tp']+=int(j in hit);v['ids'].add(key);v['parents'].add(r['parent']);v['exposures'].append(count)
  out[s]={}
  for cat,v in rec.items():
   d=dict(true=v['true'],unique=len(v['ids']),tp=v['tp'],micro_recall=v['tp']/v['true'] if v['true'] else None,parents=len(v['parents']),exposure_quantiles=np.quantile(v['exposures'],[0,.25,.5,.75,1]).tolist() if v['exposures'] else None);out[s][cat]=d;lines.append(f"|{s}|{cat}|{d['true']}|{d['unique']}|{d['tp']}|{d['micro_recall']}|{d['parents']}|")
 lines+=['此检验是额外的训练内诊断，不是held-out测试真值，也不把seen/unseen分层用于dev标签、阈值或checkpoint选择。seen意味着目标曾受直接监督，不意味着已证明记忆；unseen事件仍可能在同一录制其他上下文中被观测，不能等同完全未见录制。']
 (O/'EVENT_EXPOSURE_RESULT.json').write_text(json.dumps(out,indent=2))
 with R.open('a') as f:f.write('\n'+'\n'.join(lines)+'\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':build() if sys.argv[1]=='build' else score()
