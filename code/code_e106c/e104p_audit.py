import json,hashlib,torch,numpy as np
from e103d_runtime import *
from e99w2_data import load
from e103d_model import WorldModel
from e104p_pipeline import disk,shared

def same_state(a,b):
 return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k] for k in a)
def main():
 assert not (CONTROL/'FINAL_AUDIT.json').exists();status=json.loads((CONTROL/'PIPELINE.json').read_text());assert status['status']=='complete'
 lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());checks={}
 checks['locked_code_unchanged']=all(sha(ROOT/k)==v for k,v in lock['code_hashes'].items());assert checks['locked_code_unchanged']
 checks['locked_inputs_unchanged']=all(sha(Path(k))==v for k,v in lock['input_hashes'].items());assert checks['locked_inputs_unchanged']
 e93=T/'results/e93_dev_diag_20261114_01';oldlock=json.loads((T/'results/e102x_extended_s90_20261115_01/BEFORE_FORWARD_LOCK.json').read_text());assert sha(e93/'dev/windows_full.json')==oldlock['windows_sha'];assert sha(e93/'dev/captures_full.json')==oldlock['captures_sha'];assert sha(CONTROL/'GRIPPER_CONTRACT.json')==oldlock['contract_sha'];checks['E102X_windows_captures_gripper_unchanged']=True
 core=['e99w2_model.py','e99w2_data.py','e99w2_objective.py','e100c_chain.py','e102a_contract.py','e102a_eval.py']
 checks['old_A2_core_matches_initial_copy']=all(sha(T/'code_e102a'/k)==lock['code_hashes'][k] for k in core);assert checks['old_A2_core_matches_initial_copy']
 assert all(json.loads(p.read_text())['status']=='closed' for p in CONTROL.glob('gpu_*.json'));checks['all_GPU_segments_closed']=True
 stream=json.loads((CONTROL/'PAIRED_STREAM_AUDIT.json').read_text());assert all(stream['checks'].values());checks['paired_Piper_stream']=True
 cp={g:torch.load(path,map_location='cpu',weights_only=False) for g,path in lock['checkpoints'].items()}
 _,_,tr,_=load();m=WorldModel(tr.norm);names=[n for n,p in m.named_parameters()];ids=cp['A2']['optimizer']['param_groups'][0]['params'];assert len(names)==len(ids);byid=dict(zip(names,ids))
 for g,x in cp.items():assert x['optimizer']['param_groups'][0]['params']==ids and x['optimizer']['param_groups'][0]['lr']==3e-4
 frozen=json.loads((T/'results/e104p_f1_s90_20261115_03/EXTERNAL_FREEZE_AUDIT.json').read_text())['frozen_names']
 for n in frozen:
  assert torch.equal(cp['A2']['model'][n],cp['F1-pre']['model'][n]),n
  assert same_state(cp['A2']['optimizer']['state'].get(byid[n],{}),cp['F1-pre']['optimizer']['state'].get(byid[n],{})),n
 checks['external_frozen_parameters_AND_AdamW_state_exact']=True
 def step(g,n):return int(cp[g]['optimizer']['state'][byid[n]]['step'])
 counts={}
 for n in ['action_head.weight','delta_head.weight','grip_head.weight','mean_head.0.weight','robot_proj.weight']:
  counts[n]={g:step(g,n) for g in cp};b=step('A2',n)
  ext=0 if n in frozen else 5000
  assert step('F1-pre',n)==b+ext
  for g,s in [('F0',5000),('F1',5000),('F2',10000)]:
   if n=='robot_proj.weight':
    summary=json.loads((T/f"results/e104p_{'f2' if g in ('F0','F2') else 'f1'}_s90_20261115_03/summary_{s:05d}.json").read_text());inc=s-summary['modes'].get('action_only',0)
   else:inc=s
   assert step(g,n)==b+(ext if g=='F1' else 0)+inc,(n,g,counts[n])
 checks['continuous_optimizer_step_counts']=True
 now=json.loads((CONTROL/'eval_A2/evaluation.json').read_text());old=json.loads((T/'results/e102x_extended_s90_20261115_01/A2_retry02/evaluation.json').read_text());oldw={(r['index'],r['h'],r['method']):r for r in old['world']};olda={r['index']:r for r in old['actions']};wmax=max(abs(r['model']-oldw[(r['index'],r['h'],r['method'])]['model']) for r in now['world']);amax=0.;events=True
 for r in now['actions']:
  a=olda[r['index']];events &= r['events']==a['events']
  for h in ('8','50'):
   for k in ('l1','mse','copy_l1','copy_mse'):amax=max(amax,float(np.max(abs(np.asarray(r['joints'][h][k])-a['joints'][h][k]))))
 assert wmax==0 and amax==0 and events,(wmax,amax,events);checks['A2_replays_E102X_exactly']=True
 relative={}
 for group in cp:
  rows=json.loads((CONTROL/('eval_'+group)/'evaluation.json').read_text())['actions'];mapping={e['index']:e for e in lock['entries']};relative[group]={}
  for scope in ('all27','original48'):
   aa=[r for r in rows if scope=='all27' or mapping[r['index']]['original48']];ps=sorted({r['parent'] for r in aa});idx=np.random.default_rng(90).integers(len(ps),size=(2000,len(ps)));relative[group][scope]={}
   for h in ('8','50'):
    l=np.array([np.mean([r['joints'][h]['l1'] for r in aa if r['parent']==p]) for p in ps]);c=np.array([np.mean([r['joints'][h]['copy_l1'] for r in aa if r['parent']==p]) for p in ps]);q=lambda v:np.quantile(v,[.025,.975]).tolist();relative[group][scope][h]=dict(l1=float(l.mean()),l1_ci95=q(l[idx].mean(1)),copy_l1=float(c.mean()),copy_l1_ci95=q(c[idx].mean(1)),relative_copy_improvement=float(1-l.mean()/c.mean()),relative_copy_ci95=q(1-l[idx].mean(1)/c[idx].mean(1)))
 atomic(CONTROL/'RELATIVE_COPY.json',relative)
 lines=['## 关节L1与相对复制改善补充CI','仅既定指标的CPU汇总；parent bootstrap2000 seed90，无新推理、阈值或判据。','|集|组|前缀|L1 rad [95%CI]|相对复制改善% [95%CI]|','|---|---|---|---|---|']
 for group,scopes in relative.items():
  for scope,hs in scopes.items():
   for h,v in hs.items():lines.append(f"|{scope}|{group}|{h}|{v['l1']:.6f} {v['l1_ci95']}|{100*v['relative_copy_improvement']:.2f} {[round(100*x,2) for x in v['relative_copy_ci95']]}|")
 report('\n'.join(lines))
 upper=(used_seconds()+30*len(list(CONTROL.glob('gpu_*.json'))))/3600;assert upper<=3 and disk()<=10e9

 out=dict(status='passed',brt=brt(),checks=checks,optimizer_steps=counts,A2_replay=dict(world_max_abs=wmax,action_max_abs=amax,events_equal=events),GPUh_upper=upper,new_bytes=disk(),new_bytes_upper_including_report_temporary_and_local_artifacts=disk()+(T/'notes/e106c_longer_training_20261115.md').stat().st_size+10*1024**2,audit_sha256=sha(Path(__file__)),audit='CPU only; no additional inference/training/checkpoint selection');atomic(CONTROL/'FINAL_AUDIT.json',out)
 report('## 最终CPU审计\n'+json.dumps(out,ensure_ascii=False)+'\n所有锁定代码/输入SHA未变；原A2在27parent的动作与全部四时域世界MSE逐位复现E102-X；外部阶段冻结权重和AdamW状态逐位不变；关键头步数验证连续优化器未重置。仅CPU，无新增推理、训练或选点。有效主调度命令nohup e29/bin/python -u e104p_parallel.py（start_new_session=True），PID1963577；训练代码与初始锁一致。\n')
 reportpath=T/'notes/e106c_longer_training_20261115.md';s=reportpath.read_text().replace('状态：PREPARING','状态：COMPLETE（有效run03，终态及审计见后文）').replace('待追加。','见后文追加的完整审计。').replace('待源码核对后锁定；','已在训练前锁定；').replace('待追加。旧目录','见后文。旧目录');reportpath.write_text(s)
 shared('AUDIT_PASSED',f'固定终点全流程完成；连续AdamW/外部冻结状态/Piper成对流/A2重放/输入源码只读审计通过；GPUh上界{upper:.6f}，新增{disk()/1e9:.4f}GB；有效主调度e104p_parallel.py。')
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
