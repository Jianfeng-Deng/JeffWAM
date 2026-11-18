import sys
sys.dont_write_bytecode=True
import json,copy,traceback,time,collections,numpy as np,torch
from e99w2_common import *
from e99w2_data import load,ImageDataset
from e99w2_model import WorldModel,trajectory
from e99w2_eval import aggregate,finite
from e96a.train import configure
from e86.data import move,CAMS
from e88.audit import HorizonIndex
EXTRA=T/'results/e93_dev_diag_20261114_01/features';OLD=ART/'evaluation_01';OUT=ART/'evaluation_02_h100_complete'
class UnionImage(ImageDataset):
 def features(self,cid,cam,ids):
  if not hasattr(self,'supp_maps'):self.supp_maps={}
  result=[]
  for ix in ids:
   try:v=ImageDataset.features(self,cid,cam,[ix])[0]
   except KeyError:
    key=(cid,cam)
    if key not in self.supp_maps:self.supp_maps[key]=(np.load(EXTRA/cid/(CAMS[cam]+'_rows.npy')),np.load(EXTRA/cid/(CAMS[cam]+'.npy'),mmap_mode='r'))
    rows,arr=self.supp_maps[key];j=int(np.searchsorted(rows,ix));assert j<len(rows) and rows[j]==ix;v=arr[j].astype('f4')
   result.append(v)
  return np.stack(result)
def main():
 OUT.mkdir(exist_ok=False);configure();torch.backends.mha.set_fastpath_enabled(False);_,_,tr,dev=load();old=json.loads((OLD/'world_records.json').read_text());done={r['index'] for r in old if r['h']==100};assert done=={9};original=json.loads((C/'artifacts/dev_v1/windows_v2.json').read_text());assert [(r['canonical_id'],r['anchor'],r['parent_component']) for r in dev.rows]==[(r['canonical_id'],r['anchor'],r['parent_component']) for r in original]
 c0=json.loads((dev.base.cache/'contract.json').read_text());c1=json.loads((EXTRA/'contract.json').read_text());assert c0['preprocess']==c1['preprocess'] and c0['encoder_files']==c1['encoder_files'];replay=json.loads((T/'results/e93_dev_diag_20261114_01/cache_replay.json').read_text());assert replay['max_abs_difference']==0
 dev.base.__class__=UnionImage;dev.index=HorizonIndex(EXTRA,list(dev.index.captures.values()));selected=[]
 for i,r in enumerate(dev.rows):
  rr,why=dev.index.resolve(r,100);assert rr is not None,(i,why)
  if i not in done:selected.append(i)
 assert len(selected)==47
 lock=dict(brt=brt(),checkpoint_sha256=sha(RUN/'ckpt_05000.pt'),original_window_file=str(C/'artifacts/dev_v1/windows_v2.json'),original_windows_sha256=sha(C/'artifacts/dev_v1/windows_v2.json'),new_indices=selected,preserve_original_indices=sorted(done),source=str(EXTRA),E93_replay_zero=True,cache_contracts_match=True,short_predictions_preserved=True,reason='Complete same fixed48 P3 windows using already-existing read-only E93 cache; no new features/windows/labels, no rescoring existing predictions',disclosure='Coverage source discovered after initial cached-dev scores; all methods and all47 missing windows completed without selection; primary N1 rule unchanged',script_sha256=sha(ROOT/'e99w2_complete_h100.py'))
 atomic(OUT/'BEFORE_FORWARD_LOCK.json',lock);report('## +100覆盖补全预告（评分前）\n'+json.dumps(lock,ensure_ascii=False));atomic(ART/'STATUS.json',dict(status='completing_original_fixed_h100',brt=brt()));cp=torch.load(RUN/'ckpt_05000.pt',map_location='cpu',weights_only=False);m=WorldModel(tr.norm);m.load_state_dict(cp['model']);del cp;guard=None;records=[];vectors={};rms=np.load(OLD/'train_only_noise_rms.npy')
 try:
  from e96a.resource import Guard
  guard=Guard('E99W2_H100_fixed47_completion',600);m=m.to('cuda');m.eval()
  with torch.inference_mode():
   for i in selected:
    guard.check();r=dev.rows[i];b=move(dev.batch([i],(100,),'short'),'cuda');assert b['world_valid'].all();oracle,ov=dev.oracle(i);z,qstate,_,good=dev.states.get(dev,r['canonical_id']);a=r['anchor'];idx=np.minimum(a+np.arange(101),len(qstate)-1);ok=(a+np.arange(101)<len(qstate))&z['p1_source_mask'][idx];dt=np.diff(z['timestamp'][a:min(a+101,len(qstate))]);dg=np.ones(101,bool);dg[1:len(dt)+1]=(dt>0)&(dt<=.050001);ov &= torch.tensor(np.logical_and.accumulate(ok&dg));assert ov[100];oracle[~ov]=0
    noise=np.random.default_rng(990000+i).normal(size=(50,14))*rms;noisy=b['action']+torch.tensor(noise[None],device='cuda',dtype=torch.float32)
    sources=dict(oracle=dict(b,robot=oracle[None].to('cuda'),robot_valid=ov[None].to('cuda')),gt_action=b,no_traj=dict(b,robot=torch.zeros_like(b['robot']),robot_valid=torch.zeros_like(b['robot_valid'])),noisy_gt=dict(b,robot=trajectory(b['context'],noisy)))
    pp={k:finite(m.forward_mean(bb)['mean']).cpu().numpy() for k,bb in sources.items()};samples=[]
    for n in range(8):
     act,_=m.infer(b,'action_only',4,90+n);samples.append(finite(m.forward_mean(dict(b,robot=trajectory(b['context'],act)))['mean']).cpu().numpy())
    pp['pred_N1']=samples[0];pp['pred_N8']=np.mean(samples,0);Y=finite(b['world']).cpu().numpy();vectors[f'{i}_target']=Y
    for method,P in pp.items():
     finite(P);vectors[f'{i}_{method}']=P
     for ci,cam in [(None,'all'),(0,'top'),(1,'left_wrist'),(2,'right_wrist')]:
      sl=slice(None) if ci is None else slice(ci,ci+1);y=Y[:,sl].astype('f8');p=P[:,sl].astype('f8');records.append(dict(index=i,parent=r['parent_component'],h=100,camera=cam,method=method,model=float(((p-y)**2).mean()),copy=float((y*y).mean()),cosine=float((p*y).sum()/max(np.sqrt((p*p).sum()*(y*y).sum()),1e-12))))
    if len(records)%192==0:print('completed missing windows',len(records)//24,'/47',flush=True)
   torch.cuda.synchronize();del m;torch.cuda.empty_cache();guard.finish();guard=None
 except BaseException:
  if guard:guard.finish()
  raise
 # Append only new47*h100; all original predictions and metrics remain byte-identical in evaluation_01.
 assert len(records)==47*24;allrecords=old+records;summary,paired,ret=aggregate(allrecords);atomic(OUT/'new_h100_records.json',records);atomic(OUT/'world_records_merged.json',allrecords);atomic(OUT/'world_summary.json',summary);atomic(OUT/'world_paired.json',paired);atomic(OUT/'retention.json',ret);np.savez_compressed(OUT/'new_h100_vectors.npz',**vectors)
 decision={str(h):next(r for r in paired if r['h']==h and r['camera']=='all' and r['method']=='pred_N1') for h in (25,50,100)};passed=all(r['ci95'][0]>0 for r in decision.values());assert passed==False # +25/+50 preserved and already failed, no coverage-based rescue.
 for r in summary:assert r['windows']==48 and r['parents']==6
 d=dict(deployable_condition_effective=bool(passed),by_h=decision,limited_parent_coverage=False,small_dev_parent_count=6,no_selection=True,checkpoint_sha256=lock['checkpoint_sha256'],brt=brt());atomic(OUT/'decision.json',d)
 # Confirm original short-h summaries exactly unchanged.
 previous=json.loads((OLD/'world_summary.json').read_text());assert [r for r in summary if r['h']!=100]==[r for r in previous if r['h']!=100]
 atomic(OUT/'COMPLETE.json',dict(status='completed',new_windows=47,all_horizons_fixed48_parents6=True,existing_short_and_h100_predictions_preserved=True,brt=brt()))
 lines=['## 最终有效结果：原P3固定48窗/6parent全部时域（替代前文+100受限表）','补全只从E93既有只读缓存读取原窗口+100帧。原+10/+25/+50和原+100单窗的所有预测不重算；只增加47个缺失+100窗口。缓存contract同一encoder/preprocess，E93既有全6parent原缓存重放差为0。此前结果已看见，因此披露覆盖源发现晚；模型/终点/窗口/方法/种子/阈值均未改变。','|h|oracle skill %|GT-action %|pred N1 %|pred N8 %|no-traj %|noisy-GT %|N1−no pp [CI]|N1/N8保留率 %|','|---:|---:|---:|---:|---:|---:|---:|---|---|']
 for h in (10,25,50,100):
  ss={r['method']:r for r in summary if r['h']==h and r['camera']=='all'};p=next(r for r in paired if r['h']==h and r['camera']=='all' and r['method']=='pred_N1');rt={r['method']:r for r in ret if r['h']==h and r['camera']=='all'};fmt=lambda x:'NA' if x is None else f'{x:.2f}';lines.append('|'+ '|'.join([str(h),*[f"{100*ss[k]['skill']:.3f}" for k in ('oracle','gt_action','pred_N1','pred_N8','no_traj','noisy_gt')],f"{100*p['delta_skill']:.3f} [{100*p['ci95'][0]:.3f},{100*p['ci95'][1]:.3f}]",fmt(rt['pred_N1']['percent'])+'/'+fmt(rt['pred_N8']['percent'])])+'|')
 lines+=['所有h均48窗/6parent，parent bootstrap2000 seed90；完整CI/方向余弦/分相机与配对见evaluation_02_h100_complete/world_summary.json、world_paired.json和retention.json。原action/latency/noise_calibration沿用evaluation_01；joint动作诊断仍沿原联合查询有效mask，不作新的联合布局比较。',f"预注册有效={passed}。N8不替代预注册N1判定。单seed、6parent仍限制泛化证据；本轮不存在+100单parent退化CI的未完成项。"]
 ledger=json.loads((ART/'gpu_ledger.json').read_text());assert not ledger.get('active');upper=ledger['elapsed_s']+30*len(ledger['segments']);assert upper<=14400
 roots=[ROOT,*sorted((T/'results').glob('e99w2_*')),REPORT];disk=sum(p.stat().st_size for root in roots for p in ([root] if root.is_file() else root.rglob('*')) if p.is_file() and not p.is_symlink());assert disk<10*1024**3
 lines += [f'最终GPU时间保守上界={upper/3600:.6f}GPUh；新增落盘约{disk/1024**3:.6f}GiB。无新编码/新数据副本/语言标签。','本轮指定训练与评估已全部完成。动作仍未超过复制当前状态的L1基线；并未证明部署条件在预注册全部时域有效。需要用户决定：当前无待决事项；若后续开展更广parent/多seed或动作质量改进，另轮预注册。']
 report('\n'.join(lines));oldstatus=json.loads((ART/'FINAL_STATUS.json').read_text());atomic(ART/'FINAL_STATUS_cache_limited01_preserved.json',oldstatus);final=dict(status='COMPLETE_FIXED_P3_48WINDOWS_ALL_HORIZONS',brt=brt(),decision=d,GPUh_upper=upper/3600,new_bytes=disk,checkpoint=str(RUN/'ckpt_05000.pt'),report=str(REPORT),valid_world_results=str(OUT),valid_action_results=str(OLD),all_horizons_windows=48,all_horizons_parents=6,coverage_completion_disclosed=True)
 atomic(ART/'FINAL_STATUS.json',final);atomic(ART/'STATUS.json',final)
 # Update report skeleton status only; all dated audit/history sections remain preserved.
 text=REPORT.read_text().replace('状态：动作合同核查与实现中。','状态：COMPLETE_FIXED_P3_48WINDOWS_ALL_HORIZONS；最终有效世界表见末尾覆盖补全节。').replace('待完成。','已完成，详见后续审计与最终有效结果。');REPORT.write_text(text)
 for path,line in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',f'\n### E99-W2 +100覆盖补全与最终状态 | {brt()} BRT\n改动：只读复用E93已有特征，原48窗口/6parent不变；只对47个未评分+100窗口推理，原预测保留。命令：setsid nohup .../python e99w2_complete_h100.py；结果目录results/e99w2_control_20261115_01/evaluation_02_h100_complete；预注册有效={passed}；GPUh≤{upper/3600:.6f}，新增约{disk/1024**3:.3f}GiB。结论：原单parent限制已解决；N1规则仍未通过。状态COMPLETE_FIXED_P3_48WINDOWS_ALL_HORIZONS。\n'),(T/'notes/worklog.md',f'\n- {brt()} BRT E99-W2最终补全：原48窗/6parent四时域齐全，只读E93缓存；N1预注册未通过，GPUh≤{upper/3600:.6f}；报告末尾为最终有效口径。\n')]:
  with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(line)
 print(json.dumps(final,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(ART/'h100_completion_failure.json',dict(brt=brt(),traceback=traceback.format_exc()));raise
