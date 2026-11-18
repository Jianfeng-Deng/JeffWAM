import sys,os,json,time,collections,numpy as np
from pathlib import Path
sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');SRC=T/'code_e104p';OUT=T/'results/e102x_extended_s90_20261115_01';REPORT=T/'notes/e102x_extended_eval_20261115.md'
sys.path.insert(0,str(SRC))
from e99w2_common import atomic,sha,brt,CONTROL
from e99w2_data import load,PlannedData
from e86.data import stratum
E93=T/'results/e93_dev_diag_20261114_01'
def datasets():
 p,ready,tr,old=load();caps=json.loads((E93/'dev/captures_full.json').read_text());rows=json.loads((E93/'dev/windows_full.json').read_text());rows=[dict(r,_stage='P3',source='piper') for r in rows]
 dev=PlannedData(rows,caps,E93/'features',tr.base.stats,{'scale':tr.norm},{'__human_default__':old.base.text_cache['__human_default__']},stage='P3')
 assert len(rows)==211 and len({r['parent_component'] for r in rows})==27
 assert all(p.split[r['parent_component']]=='dev' and r['canonical_id'] not in p.excluded for r in rows)

 # E93 numeric caches omit action arrays; direct next-frame feedback view in RAM.
 receipts=json.loads((T/'results/e97a_robot_env_20261114_01/dev_state_restore_receipt.json').read_text())
 receipts={x['canonical_id']:x for x in receipts}
 for cid in dev.index.captures:
  z,q,pose,good=dev.states.get(dev,cid)
  rec=receipts[cid];assert rec['status']=='restored_exact_identity_and_timing';assert sha(Path(rec['supplement']))==rec['sha256']
  if 'action' not in z:
   assert len(z['original_action_index'])==len(q)-1
   assert np.array_equal(z['original_action_index'],np.arange(1,len(q)))
   z['action']=q[1:] # canonical source index i+1, no shift of any existing action/label
 return p,ready,tr,dev

def prepare():
 from e96a.train import configure
 configure();p,ready,tr,dev=datasets();original=json.loads((CONTROL/'fixed_windows.json').read_text())['dev'];oldset={(r['cid'],r['anchor'],r['parent']) for r in original};selected=[];rejected=[];counts=collections.defaultdict(lambda:dict(candidates=0,accepted=0,horizons=collections.Counter()))
 for i,r in enumerate(dev.rows):
  counts[r['parent_component']]['candidates']+=1
  try:
   z,q,pose,good=dev.states.get(dev,r['canonical_id']);a=r['anchor'];assert q is not None,'missing_state'
   assert a+50<len(q),'short_action';assert good[a:a+51].all(),'invalid_action_state'
   assert (z['continuous_segment'][a:a+51]==z['continuous_segment'][a]).all(),'action_state_seam'
   assert z['p1_source_mask'][a:a+51].all(),'action_source_mask'
   import numpy as np
   assert np.array_equal(z['original_action_index'][a:a+50],np.arange(a+1,a+51)),'action_index'
   assert np.array_equal(z['action'][a:a+50],q[a+1:a+51]),'action_not_next_feedback'
   b=dev.get(i,(10,25,50,100),'short');valid=[];why={}
   for j,h in enumerate((10,25,50,100)):
    rr,reason=dev.index.resolve(r,h)
    if bool(b['world_valid'][b['world_horizons']==h].all()):
     _,ov=dev.oracle(i);assert bool(ov[h]),'invalid_oracle_state'
     valid.append(h);counts[r['parent_component']]['horizons'][h]+=1
    else:why[h]=reason or 'world_invalid'
   assert all(h in valid for h in (10,25,50)),'short_world_invalid'
   entry=dict(index=i,cid=r['canonical_id'],parent=r['parent_component'],anchor=a,horizons=valid,invalid_horizons=why,original48=(r['canonical_id'],a,r['parent_component']) in oldset,stratum=next(k for k in ('demo3','demo4','demo6') if k in dev.index.captures[r['canonical_id']]['source_collection']))
   selected.append(entry);counts[r['parent_component']]['accepted']+=1
  except Exception as e:rejected.append(dict(index=i,parent=r['parent_component'],anchor=r['anchor'],reason=str(e),type=type(e).__name__))
 assert sum(x['original48'] for x in selected)==48
 lock=dict(brt=brt(),rule='E93 immutable windows_full.json, all211 candidates; no new anchors; action legality and state checks before any model output; original48 subset',windows_sha=sha(E93/'dev/windows_full.json'),captures_sha=sha(E93/'dev/captures_full.json'),contract_sha=sha(CONTROL/'GRIPPER_CONTRACT.json'),entries=selected,rejected=rejected,per_parent=dict(counts),bootstrap=dict(unit='parent_equal',replicates=2000,seed=90),horizons=[10,25,50,100],flow_steps=4,inference_seed=90,language='short human default only',peak_estimate_mib=6144,gpu_budget_s=1800,disk_limit_bytes=10*1024**3,feature_extraction_bytes=0,decision='E102-A unchanged joint both L1 improvement CI lower>0; gripper both A1/A2 F1@10 minus A0 CI lower>0 and lower mean false switches; world N1-no CI lower>0 at25/50/100',checkpoints={g:dict(path=str(T/f'results/e102a_{g.lower()}_s90_20261115_03/ckpt_02000.pt'),sha=sha(T/f'results/e102a_{g.lower()}_s90_20261115_03/ckpt_02000.pt')) for g in ('A0','A1','A2')})
 atomic(OUT/'BEFORE_FORWARD_LOCK.json',lock)
 with REPORT.open('a') as f:
  f.write('\n## 窗口锁定（任何模型前向前）\n'+json.dumps({k:v for k,v in lock.items() if k not in ('entries','per_parent','checkpoints')},ensure_ascii=False)+'\n\n|parent|候选|动作合法|+10/+25/+50/+100|\n|---|---:|---:|---|\n')
  for par,c in sorted(counts.items()):f.write(f"|{par}|{c['candidates']}|{c['accepted']}|"+'/'.join(str(c['horizons'].get(h,0)) for h in (10,25,50,100))+'|\n')
  f.write('\n原48窗完整包含；所有评估条件沿用E102-A，包括训练拟合夹爪合同、起点代理、一一事件匹配。状态/动作失败记录见锁文件；世界时域按有效mask，缺+100不伪造。单seed，仅dev结论，无部署操作。\n')
 print(json.dumps(dict(accepted=len(selected),rejected_count=len(rejected),reasons=collections.Counter(x['reason'] for x in rejected),parents=len(counts),strata=collections.Counter(x['stratum'] for x in selected)),ensure_ascii=False))
if __name__=='__main__':prepare()
