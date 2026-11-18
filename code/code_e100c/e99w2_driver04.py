import os,time,json,traceback,torch
from e99w2_common import *
from e99w2_data import load
from e99w2_train import train,contracts
from e99w2_eval import aggregate
from e96a.train import configure

def main():
 configure();_,_,tr,_=load()
 # The previous audit is immutable evidence; do not overwrite it.
 assert (ART/'action_contract.json').exists() and (ART/'contracts.json').exists()
 from e99w2_model import WorldModel
 m=WorldModel(tr.norm);m.load_state_dict(torch.load(INIT,map_location='cpu',weights_only=False)['model']);m.eval();checked=[]
 for k in (25,50):
  valid=0
  for i in list(range(8))+list(range(len(tr.rows)-8,len(tr.rows))):
   if tr.index.resolve(tr.rows[i],2*k)[0] is None:continue
   b=tr.shifted(i,k)
   if b is None:continue
   orig=tr.get(i,(k,));suffix=torch.cat([orig['action'][k:],orig['action'][-1:].expand(k,-1)])
   assert torch.equal(b['action'],suffix)
   assert torch.isfinite(b['robot']).all();valid+=1
  assert valid>0;checked.append(dict(k=k,windows=valid))
 del m
 # Parent resampling uses one shared index for model/copy, rejects nonfinite.
 rr=[]
 for method in ('oracle','gt_action','pred_N1','pred_N8','no_traj','noisy_gt'):
  for p,n in [('a',1),('b',9)]:
   for i in range(n):rr.append(dict(h=25,camera='all',method=method,parent=p,model=1.,copy=2.,cosine=0.))
 s,paired,_=aggregate(rr);assert all(r['delta_skill']==0 and r['ci95']==[0.,0.] for r in paired)
 bad=[dict(rr[0],model=float('nan'))]
 try:aggregate(bad)
 except AssertionError:pass
 else:raise AssertionError('Nonfinite aggregate not rejected')
 atomic(ART/'chain_and_eval_contracts04.json',dict(status='passed',chain_suffix_only=checked,paired_parent_imbalance_zero=True,nonfinite_rejection=True,CUDA_initialized=torch.cuda.is_initialized(),brt=brt()))
 report('### 链与统计补充合同\n原动作块后缀逐元素一致，k25/k50近首尾窗口通过；parent窗口1:9不均衡的相同预测配对CI精确为0，NaN拒绝。run04保留既有动作合同与CPU合同文件，无dev评分。')
 atomic(ART/'STATUS.json',dict(status='training',brt=brt()));train(tr);atomic(ART/'STATUS.json',dict(status='trained',brt=brt()))
 # CUDA allocations are released by train; evaluation in a fresh env-pinned subprocess.
 import subprocess,sys
 with (ART/'eval_driver01.log').open('x') as log:subprocess.run([sys.executable,'-u','e99w2_eval.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (ART/'finalize01.log').open('x') as log:subprocess.run([sys.executable,'-u','e99w2_finalize.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(ART/'driver04_failure.json',dict(brt=brt(),traceback=traceback.format_exc()));atomic(ART/'STATUS.json',dict(status='stopped',brt=brt()));raise
