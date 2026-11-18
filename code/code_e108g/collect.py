import os,sys,json,time,traceback,hashlib
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e108g_diag_20261116_01';R=T/'notes/e108g_gripper_diagnosis_20261116.md'
sys.path.insert(0,str(T/'code_e106c'))
import numpy as np,torch
import e99w2_common as common
common.CONTROL=T/'results/e106c_control_20261115_02'
from e102x_setup import datasets
from e103d_model import WorldModel
from e102a_contract import contract,onset_labels,states
from e102a_eval import events,match
from e102a_finalize import action_summary
from e96a.train import configure

def put(n,x):
 p=O/n;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2));os.replace(tmp,p)
def report(s):
 with R.open('a') as f:f.write('\n'+s+'\n')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 start=time.monotonic();configure();torch.backends.mha.set_fastpath_enabled(False)
 ck=T/'results/e106c_f2_s90_20261115_02/ckpt_20000.pt';oldsha=sha(ck)
 p,ready,tr,dev=datasets();c=contract();lock=json.loads((common.CONTROL/'PRETRAIN_LOCK.json').read_text());entries=lock['entries']
 assert len(dev.base.text_cache)==1 and '__human_default__' in dev.base.text_cache
 assert all(r['canonical_id']!='piper_4dc107335389f5711a0f65e0' for d in (tr,dev) for r in d.rows)
 old=json.loads((common.CONTROL/'eval_S20000/evaluation.json').read_text());oldrows={r['index']:r for r in old['actions']};labels={};result=[]
 cp=torch.load(ck,map_location='cpu',weights_only=False);m=WorldModel(tr.norm);m.load_state_dict(cp['model']);del cp;m.eval().requires_grad_(False)
 # Preselect training parents/windows before scoring any decoder, no model labels as truth.
 groups={}
 for i,r in enumerate(tr.rows):groups.setdefault(r['parent_component'],[]).append(i)
 trainentries=[]
 for par,ids in sorted(groups.items()):
  ids=sorted(ids,key=lambda i:(tr.rows[i]['canonical_id'],tr.rows[i]['anchor']))
  for j in sorted(set(np.linspace(0,len(ids)-1,min(16,len(ids)),dtype=int))):
   i=ids[j];r=tr.rows[i];trainentries.append(dict(index=i,cid=r['canonical_id'],parent=par,anchor=r['anchor'],original48=False))
 assert not {e['parent'] for e in trainentries}&{e['parent'] for e in entries}
 put('TRAIN_WINDOW_LOCK.json',dict(entries=trainentries,rule='all training parents, 16 equispaced ordered windows each; fixed before decoder scores',checkpoint_sha=oldsha))
 report('## 数据与概率提取锁定\n训练侧使用全部训练parent，每parent按(cid,anchor)排序取16个等距窗（去重），窗口已锁定TRAIN_WINDOW_LOCK.json。原缓存仅有二值正确性和事件统计，无可复用夹爪概率；E107-P世界token缓存不含夹爪概率。本轮CPU float32只读默认模型t=0动作分支原有grip_logits，sigmoid取得概率；不调用训练，无CUDA。事件和静止分层仍来自原位实测反馈。')
 for split,data,ee in [('dev',dev,entries),('train',tr,trainentries)]:
  with torch.inference_mode():
   for j,e in enumerate(ee):
    i=e['index'];r=data.rows[i];assert (r['canonical_id'],r['anchor'],r['parent_component'])==(e['cid'],e['anchor'],e['parent'])
    b=data.batch([i],(10,),'short');z,q,pose,good=data.states.get(data,e['cid']);a=e['anchor'];key=(split,e['cid'])
    if key not in labels:labels[key]=onset_labels(q,z['continuous_segment'],good,c)[0]
    lab=labels[key];truth=b['grip_label'][0].numpy().astype('i1');initial=b['grip_current'][0].numpy().astype('i1');init=lab[a]
    assert np.array_equal(truth,lab[a+1:a+51])
    t=torch.zeros(1);heads=m(b,torch.zeros(1,50,14),None,t,t);logits=heads['grip_logits'][0].numpy();prob=1/(1+np.exp(-logits));state=(logits>0).astype('i1');assert np.isfinite(prob).all()
    item=dict(split=split,**e,current=initial.tolist(),label_init=init.tolist(),labels=truth.tolist(),prob=prob.tolist(),logits=logits.tolist(),width=q[a:a+51,[6,13]].tolist(),phase=r.get('phase',r.get('stage',None)))
    if split=='dev':
     oldr=oldrows[i];assert np.array_equal((state==truth).astype(float),np.array(oldr['state_accuracy_by_gripper'])),f'BASELINE_STATE_MISMATCH index={i}'
     for tol in (5,10):
      pe=[events(state[:,k],initial[k]) for k in range(2)];te=[events(truth[:,k],init[k]) for k in range(2)];errs=sum([match(pe[k],te[k],tol) for k in range(2)],[]);orig=oldr['events'][str(tol)]
      assert (len(errs),sum(map(len,pe)),sum(map(len,te)),errs)==(orig['tp'],orig['pred'],orig['true'],orig['errors']),f'BASELINE_EVENT_MISMATCH {i} tol={tol}'
    result.append(item)
    if (j+1)%25==0:
     put('status.json',dict(status='collect',split=split,done=j+1,total=len(ee),elapsed_s=time.monotonic()-start,GPUh=0));print(split,j+1,len(ee),round(time.monotonic()-start,1),flush=True)
  if split=='dev':
   summary=action_summary(old['actions']);repro={str(t):{k:summary['events'][str(t)][k] for k in ('f1','false_switches_per_minute','recall')} for t in (5,10)}
   put('BASELINE_REPRODUCED.json',dict(exact_all211_state_and_events=True,summary=repro,checkpoint_sha=oldsha));report('## 原样基线复现通过\nCPU提取211窗逐帧二值状态、每窗±5/±10的TP/预测数/真值数/匹配偏移均与旧GPU评估逐元素一致。'+json.dumps(repro,ensure_ascii=False))
   put('dev_predictions.json',result)
 assert sha(ck)==oldsha and not torch.cuda.is_initialized()
 put('predictions.json',result);put('COLLECT_COMPLETE.json',dict(status='complete',windows=len(result),train_windows=len(trainentries),train_parents=len(groups),elapsed_s=time.monotonic()-start,GPUh=0,checkpoint_sha=oldsha))
if __name__=='__main__':
 try:main()
 except BaseException:put('STOPPED.json',dict(traceback=traceback.format_exc(),GPUh=0));raise
