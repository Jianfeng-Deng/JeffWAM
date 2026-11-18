import sys
sys.dont_write_bytecode=True
import json,numpy as np,collections
from e99w2_common import *
ev=ART/'evaluation_01';assert (ART/'FINAL_STATUS.json').exists();locks=json.loads((ART/'EVAL_SOURCE_LOCK.json').read_text());assert sha(ROOT/'e99w2_eval.py')==locks['eval_sha256'];assert sha(ROOT/'e99w2_finalize.py')==locks['finalizer_sha256'];rms=np.load(ev/'train_only_noise_rms.npy');actions=json.loads((ev/'action_records.json').read_text());noise=[]
for i in range(48):noise.append(np.random.default_rng(990000+i).normal(size=(50,14))*rms)
noise=np.stack(noise);out=[]
for h in (8,50):
 for indices,name,scale in [(list(range(6))+list(range(7,13)),'joint',1.),([6,13],'gripper',1000.)]:
  train_rms=float(np.sqrt((rms[:h,indices]**2).mean())*scale);actual=float(np.sqrt((noise[:,:h,indices]**2).mean())*scale);rr=[r for r in actions if r['method']=='action_only_N1' and r['h']==h];ps=sorted({r['parent'] for r in rr});key='joint_RMSE_rad' if name=='joint' else 'gripper_RMSE_mm';pred=float(np.mean([np.mean([r[key] for r in rr if r['parent']==p]) for p in ps]));out.append(dict(h=h,dimension=name,unit='rad' if name=='joint' else 'mm',train_calibration_RMS=train_rms,realized_dev_Gaussian_RMS=actual,dev_pred_N1_parent_mean_window_RMSE=pred,noise_to_training_target_RMS_ratio=actual/train_rms))
atomic(ev/'noise_amplitude_audit.json',out)
ledger=json.loads((ART/'gpu_ledger.json').read_text());assert not ledger.get('active');updates=[json.loads(s) for s in (RUN/'updates.jsonl').read_text().splitlines()];assert len(updates)==5000 and [r['step'] for r in updates]==list(range(1,5001));assert all(np.isfinite(r['loss']) for r in updates);assert all(r['chain_weight']==0 for r in updates[:3000]);assert all(0<=r['chain_weight']<=.1 for r in updates);assert all(r['chain_weight']==0 for r in updates if r['mode']=='action_only');assert all(r['chain_weight']==.1 for r in updates[4000:] if r['chain_weight']>0)
summary=json.loads((RUN/'train_summary.json').read_text());cp=json.loads((ART/'FINAL_STATUS.json').read_text());assert summary['sha256']==sha(RUN/'ckpt_05000.pt');s=summary['sampler'];child=s['children']['piper'];assert child['demo3_count']/child['count']<=.45
receipt=dict(status='passed',brt=brt(),training_steps=5000,finite_updates=True,chain_schedule_and_action_only_deletion=True,demo3_fraction=child['demo3_count']/child['count'],eval_and_finalizer_match_pre_dev_source_lock=True,no_active_GPU_segment=True,noise_amplitude_audit=out)
atomic(ART/'tail_audit.json',receipt);report('## 最终尾部审计\n'+json.dumps(receipt,ensure_ascii=False)+'\n同幅噪声按训练内RMS标定；实际dev预测误差幅度也单列，仅作描述、不重新调节噪声。噪声与预测误差的维度/时序相关性不同，不能据两者差异单独识别全部误差机制。');print(json.dumps(receipt,ensure_ascii=False))
