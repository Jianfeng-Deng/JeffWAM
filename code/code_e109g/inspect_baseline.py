import os,sys,json,traceback,time
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';R=T/'notes/e109g_gripper_generalization_20261116.md'
sys.path.insert(0,str(T/'code_e108g'));import collect as old
old.common.CONTROL=T/'results/e106c_control_20261115_02'
import analyze as ev
import numpy as np,torch
old.configure();torch.backends.mha.set_fastpath_enabled(False)
p,ready,tr,dev=old.datasets();c=old.contract();cp=torch.load(T/'results/e106c_f2_s90_20261115_02/ckpt_20000.pt',map_location='cpu',weights_only=False);m=old.WorldModel(tr.norm);m.load_state_dict(cp['model']);del cp;m.eval().requires_grad_(False)
rr=json.loads((T/'results/e108g_diag_20261116_01/predictions.json').read_text());ds=[r for r in rr if r['split']=='dev'];ts=[r for r in rr if r['split']=='train']
try:
 with torch.inference_mode():
  for r in ds:
   b=dev.batch([r['index']],(10,),'short');tt=torch.zeros(1);log=m(b,torch.zeros(1,50,14),None,tt,tt)['grip_logits'][0].numpy();assert np.array_equal(log>0,np.array(r['logits'])>0)
   for tol in (5,10):assert ev.eventrow(r,(log>0).astype(int),tol)==ev.eventrow(r,(np.array(r['logits'])>0).astype(int),tol)
 v=ev.score(ds,[(np.array(r['logits'])>0).astype(int) for r in ds]);assert abs(v['10']['f1']-.24567901234567902)<1e-12 and abs(v['5']['f1']-.14938271604938272)<1e-12
 (O/'BASELINE_REPRODUCED.json').write_text(json.dumps(dict(exact211_binary_events=True,f1_10=v['10']['f1'],f1_5=v['5']['f1'],false10=v['10']['false_per_min'],GPUh=0)))
 with R.open('a') as f:f.write('\n## 基线复现通过\n本轮默认20k模型CPU重新前向211窗；逐帧二值预测与±5/±10事件逐项复现E108-G。F1@5=0.149382716049，@10=0.245679012346，假检/分钟@10=8.5；GPUh=0。\n')
 print('BASELINE PASS',flush=True)
except BaseException:
 (O/'STOPPED.json').write_text(traceback.format_exc());raise
lock=json.loads((old.common.CONTROL/'PRETRAIN_LOCK.json').read_text())
summary=dict(train_parents=len({r['parent_component'] for r in tr.rows}),train_caps=len(tr.index.captures),rows=len(tr.rows),capture_example=list(tr.index.captures.values())[:2],train_row=tr.rows[0],dev_row=dev.rows[0],checkpoints=lock['checkpoints'],ready=ready)
(O/'INSPECT.json').write_text(json.dumps(summary,default=str,indent=2));print(json.dumps(summary,default=str)[:17000],flush=True)
for d in (tr,dev):
 cid=d.rows[0]['canonical_id'];z,q,_,g=d.states.get(d,cid);print('STATE',cid,len(q),[(k,np.shape(v)) for k,v in z.items()],flush=True)
(O/'INSPECT_COMPLETE.json').write_text(json.dumps(dict(status='complete')))
