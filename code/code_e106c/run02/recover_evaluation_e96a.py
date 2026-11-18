"""CPU-only recovery of JSON serialization tail; no new inference or selection."""
import os,json,time,traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
from e96a.common import T,ART,atomic,brt,report
from eval_e96a import H,summary,E93,M1,M2
started=time.monotonic()
old=json.loads((ART/'evaluation_status.json').read_text());assert old['status']=='stopped' and 'not JSON serializable' in old['error']
backup=ART/'evaluation_status_failed_01.json';assert not backup.exists();backup.write_text(json.dumps(old,indent=2))
try:
 allrs={};audit={}
 for method in ('E96a','M1','M2','ridge_E93'):
  for pool,count in [('train64',1917),('dev',841)]:
   rs=json.loads((ART/(method+'_'+pool+'_records.json')).read_text());assert len(rs)==count
   allrs[method,pool]=rs
   if method in ('M1','M2'):
    path=M1 if method=='M1' else M2
    old=[r for r in json.loads((path/(pool+'_records.json')).read_text()) if r['method']==method+'_mean']
   elif method=='ridge_E93':old=json.loads(((T/'results/e94a_control_20261114_01/ridge_train64_records.json') if pool=='train64' else E93/'linear_records.json').read_text())
   else:old=[]
   maxerr=0.
   if old:
    lookup={(r['index'],r['h'],r['parent']):r for r in old};assert len(lookup)==len(rs)
    for r in rs:
     a=lookup[r['index'],r['h'],r['parent']];assert abs(a['copy']-r['copy'])<1e-5
     maxerr=max(maxerr,abs(a['model']-r['model']))
    assert maxerr<2e-5
   audit[method+'_'+pool]=dict(records=len(rs),old_native_max_abs_mse_difference=maxerr)
   for suffix in ('summaries','decomposition'):
    d=json.loads((ART/(method+'_'+pool+'_'+suffix+'.json')).read_text());assert set(d)=={'normalized','fine_raw','global_raw'}
   if method!='ridge_E93':assert len(json.loads((ART/(method+'_'+pool+'_joint_records.json')).read_text()))==count
 paired=json.loads((ART/'paired_E96a_minus_M1.json').read_text());checks=[]
 for p in ('train64','dev'):
  x=summary(allrs['E96a',p]);ridge=summary(allrs['ridge_E93',p])
  for h in (10,25,50):
   a=x[str(h)];b=ridge[str(h)]['values']['skill_copy'];skill=a['values']['skill_copy'];lo=a['ci95']['skill_copy'][0]
   checks.append(dict(pool=p,h=h,skill=skill,ci_lower=lo,ridge_skill=b,ridge_fraction=skill/b,positive_point=skill>0,positive_ci=lo>0,at_least_half_ridge=skill>=.5*b,passed=lo>0 and skill>=.5*b))
 improvement=bool(paired['dev']['normalized']['10']['ci95'][0]>0);local=bool(paired['dev']['fine_raw']['10']['delta_skill']>0)
 verdict=dict(pre_registered_before_results=True,dev10_paired_improvement_ci_lower_positive=improvement,local_fine_raw_improves=local,six_short_h_checks=checks,six_short_h_all_pass=all(r['passed'] for r in checks),passed=improvement and local and all(r['passed'] for r in checks),no_extension_or_checkpoint_selection=True)
 assert not (ART/'preregistered_verdict.json').exists();atomic(ART/'preregistered_verdict.json',verdict);atomic(ART/'evaluation_audit.json',audit)
 ledger=json.loads((ART/'gpu_ledger.json').read_text());assert 'active' not in ledger
 evalsegment=[s for s in ledger['segments'] if s['phase']=='E96a_evaluation'][0]
 import torch
 assert not torch.cuda.is_initialized()
 atomic(ART/'evaluation_recovery_02.json',dict(status='completed',brt=brt(),GPUh=0,seconds=time.monotonic()-started,all_inference_records_preserved=True,thresholds_unchanged=True,cuda_initialized=False,reason='NumPy bool converted to builtin bool only',failed_evaluation_GPU_segment_seconds=evalsegment['elapsed_s']))
 atomic(ART/'evaluation_status.json',dict(status='completed',brt=brt(),bootstrap=2000,seed=90,main_layout='single_h',joint='supplement only',recovery='CPU JSON-tail recovery only; full metrics saved before failure',GPU_segment_s=evalsegment['elapsed_s'],all_inference_complete=True,raw_scores='raw residuals; per-camera average before square for global'))
 report('\n## 统一评估完成与失败段说明\n'+brt()+'：全部E96a/M1/M2均值与冻结岭的推理、分解和配对bootstrap完成。首次尾部保存判定JSON因NumPy布尔值不能被json序列化而退出；所有逐窗、summary/decomposition/joint与配对文件已经落盘。失败GPU段完整计入gpu_ledger。修正仅将比较结果转Python bool；CPU读取已存记录核验全部计数和旧表一致性，重建完全相同预注册判定；不重新推理/训练，不改阈值，CUDA未初始化。原失败状态保存在evaluation_status_failed_01.json，日志evaluation_01.log保留，恢复见evaluation_recovery_02.json。')
 print(json.dumps(dict(status='completed',verdict=verdict['passed'],dev10=paired['dev']['normalized']['10'],GPUh_recovery=0)))
except BaseException as e:
 atomic(ART/'evaluation_recovery_02.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc(),brt=brt()));raise
