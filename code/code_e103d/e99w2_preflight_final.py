import sys
sys.dont_write_bytecode=True
import json,collections,numpy as np
from e99w2_common import *
from e99w2_data import load
p,ready,tr,dev=load();coverage={};world=[]
for h in (10,25,50,100):
 ids=[];parents=set();why=collections.Counter()
 for i,r in enumerate(dev.rows):
  rr,reason=dev.index.resolve(r,h)
  if rr is not None:ids.append(i);parents.add(r['parent_component'])
  else:why[reason]+=1
 coverage[str(h)]=dict(indices=ids,windows=len(ids),parents=len(parents),excluded=dict(why))
assert coverage['10']['windows']==48 and coverage['25']['windows']==48 and coverage['50']['windows']==48
# Quantify H100 terminal-hold gap on ALL image-valid P3 training windows, not dev tuning.
by=collections.defaultdict(list);maxerr=np.zeros(14);n=0
for r in tr.rows:
 rr,_=tr.index.resolve(r,100)
 if rr is None:continue
 z,q,_,g=tr.states.get(tr,r['canonical_id']);a=r['anchor'];err=z['action'][a+49].astype('f8')-q[a+100];assert np.isfinite(err).all();by[r['parent_component']].append(abs(err));maxerr=np.maximum(maxerr,abs(err));n+=1
out=dict(train_windows=n,train_parents=len(by),parent_equal_mae=np.mean([np.mean(v,0) for v in by.values()],0).tolist(),maximum_error=maxerr.tolist(),order='left6 rad,left_gripper m,right6 rad,right_gripper m',not_model_input_or_threshold=True)
atomic(ART/'preflight_dev_coverage.json',coverage);atomic(ART/'plan_vs_oracle_h100.json',out)
contract=json.loads((ART/'action_contract.json').read_text());names=['LJ1','LJ2','LJ3','LJ4','LJ5','LJ6','Lgrip','RJ1','RJ2','RJ3','RJ4','RJ5','RJ6','Rgrip'];lines=['### 时间对齐与GT计划/oracle差异补充','全34,777块 ×50步=1,738,850对；动作与next-frame raw反馈逐元素完全相等。下表关节rad、夹爪mm；q(t)与q(t+2)仅诊断相邻时间对齐，未改标签。','|维度|对q(t) MAE|对q(t+1) MAE|对q(t+2) MAE|H100保持末端对oracle MAE（训练parent等权）|','|---|---:|---:|---:|---:|']
for j,name in enumerate(names):
 scale=1000 if j in (6,13) else 1;vs=[contract['by_q_offset'][str(off)]['mae'][j]*scale for off in (0,1,2)];vs.append(out['parent_equal_mae'][j]*scale);lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in vs)+'|')
lines+=['帧间隔秒[min,median,p95,max]='+str(contract['frame_dt_s_quantiles']),f'H100末端保持差异采用{n}个合法训练窗口、{len(by)}个parent；不作为阈值、选点或训练目标。',f'P3冻结dev预检（无模型前向）={json.dumps(coverage,ensure_ascii=False)}。']
report('\n'.join(lines));print(json.dumps(dict(dev=coverage,H100_train_windows=n,H100_train_parents=len(by)),ensure_ascii=False))
